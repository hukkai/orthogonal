import argparse
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import (
    AverageMeter,
    BlockStiefelAdam,
    accuracy,
    build_imagenet_datasets,
    cosine_lr,
    get_param_groups,
    init_distributed,
    is_main_process,
    save_checkpoint,
    set_seed,
)

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("ImageNet ViT Training")
    parser.add_argument("--data", type=str, required=True, help="Path to ImageNet root")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024, help="Global batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override base LR")
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)

    parser.add_argument("--drop", type=float, default=0.0)
    parser.add_argument("--attn-drop", type=float, default=0.0)
    parser.add_argument("--drop-path", type=float, default=0.2)

    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=0.8)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch")

    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1")
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--reprob", type=float, default=0.25)
    parser.add_argument("--remode", type=str, default="pixel")
    parser.add_argument("--recount", type=int, default=1)
    parser.add_argument("--interpolation", type=str, default="bicubic")
    parser.add_argument("--crop-pct", type=float, default=0.875)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--output", type=str, default="./output")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=999)
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--orthogonal-type", type=str, choices=["none", "all", "atten", "mlp"], default="none")
    parser.add_argument("--orth-dim", type=int, default=None)
    parser.add_argument("--so-lr", type=float, default=0.5)
    parser.add_argument("--orth-beta1", type=float, default=0.9)
    parser.add_argument("--orth-beta2", type=float, default=0.999)
    parser.add_argument("--orth-eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-orth-project-last",
        dest="orth_project_last",
        action="store_false",
        help="Deprecated no-op kept for script compatibility",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--model-ema-decay", type=float, default=0.9999)

    return parser.parse_args()


def resolve_orth_dim(args: argparse.Namespace) -> int | None:
    if args.orthogonal_type == "none":
        return None
    return args.orth_dim if args.orth_dim is not None else args.embed_dim


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    from models import rout_model

    chunk_type = args.orthogonal_type

    common_kwargs = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        num_classes=args.num_classes,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop,
        drop_path_rate=args.drop_path
    )

    model_kwargs = dict(
        **common_kwargs,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        init_values=1e-4,
        chunk_type=chunk_type,
    )
    if chunk_type != "none":
        model_kwargs["orth_dim"] = args.orth_dim

    model = rout_model(**model_kwargs)
    return model


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    exclude = ["chunk_weights"] if args.orthogonal_type != "none" else []
    param_groups = get_param_groups(model, args.weight_decay, exclude_names=exclude)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    return optimizer


def train_one_epoch(
    args: argparse.Namespace,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    steps_per_epoch: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    mixup_fn,
    criterion,
    orth_opt: BlockStiefelAdam | None,
    ema_model,
) -> tuple[float, float]:
    model.train()
    loss_meter = AverageMeter("loss")
    acc_meter = AverageMeter("acc1")

    for i, (images, targets) in enumerate(loader):
        step = epoch * steps_per_epoch + i
        lr = cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )

        with autocast_ctx:
            outputs = model(images)
            loss = criterion(outputs, targets)

        loss.backward()

        if args.clip_grad and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        is_last = i == len(loader) - 1
        if orth_opt is not None:
            orth_opt.step(lr=lr * args.so_lr, is_last=is_last)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if ema_model is not None:
            ema_model.update(model.module if hasattr(model, "module") else model)

        loss_meter.update(loss.item(), images.size(0))
        if mixup_fn is None:
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc_meter.update(acc1.item(), images.size(0))

        if is_main_process() and (i % args.log_interval == 0 or i == len(loader) - 1):
            if mixup_fn is None:
                print(
                    f"Epoch [{epoch}] Step [{i}/{len(loader)}] "
                    f"LR {lr:.6f} Loss {loss_meter.avg:.4f} Acc@1 {acc_meter.avg:.2f}"
                )
            else:
                print(
                    f"Epoch [{epoch}] Step [{i}/{len(loader)}] "
                    f"LR {lr:.6f} Loss {loss_meter.avg:.4f}"
                )

    return loss_meter.avg, acc_meter.avg


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion,
) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = targets.size(0)
        loss_sum += loss.item() * batch_size
        top1_sum += acc1.item() * batch_size / 100.0
        top5_sum += acc5.item() * batch_size / 100.0
        total += batch_size

    if dist.is_initialized():
        tensor = torch.tensor([loss_sum, top1_sum, top5_sum, total], device=device)
        dist.all_reduce(tensor)
        loss_sum, top1_sum, top5_sum, total = tensor.tolist()

    loss_avg = loss_sum / total
    top1 = 100.0 * top1_sum / total
    top5 = 100.0 * top5_sum / total
    return loss_avg, top1, top5


def main() -> None:
    args = parse_args()
    args.orth_dim = resolve_orth_dim(args)
    distributed, local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    set_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    train_set, val_set = build_imagenet_datasets(args)

    if distributed:
        train_sampler = DistributedSampler(train_set)
        val_sampler = DistributedSampler(val_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    if args.batch_size % world_size != 0:
        raise ValueError("Global batch size must be divisible by world size")
    per_gpu_batch = args.batch_size // world_size

    train_loader = DataLoader(
        train_set,
        batch_size=per_gpu_batch,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=per_gpu_batch,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = build_model(args).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])

    if args.lr is None:
        args.lr = args.base_lr * args.batch_size / 512.0

    optimizer = create_optimizer(args, model)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs

    orth_opt = None
    if args.orthogonal_type != "none":
        module = model.module if hasattr(model, "module") else model
        orth_opt = BlockStiefelAdam(
            module.chunk_weights,
            lr=args.lr * args.so_lr,
            betas=(args.orth_beta1, args.orth_beta2),
            eps=args.orth_eps,
        )

    ema_model = None
    if args.model_ema:
        ema_model = ModelEmaV2(model.module if hasattr(model, "module") else model, decay=args.model_ema_decay)

    mixup_fn = None
    if args.mixup > 0.0 or args.cutmix > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )

    if mixup_fn is not None:
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    val_criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_acc1 = 0.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc1 = checkpoint.get("best_acc1", 0.0)
        if ema_model is not None and "ema" in checkpoint and checkpoint["ema"] is not None:
            ema_model.load_state_dict(checkpoint["ema"])
        if is_main_process():
            print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        base_loss, base_acc1, base_acc5 = evaluate(model, val_loader, device, val_criterion)
        if ema_model is not None:
            ema_loss, ema_acc1, ema_acc5 = evaluate(ema_model.module, val_loader, device, val_criterion)
        if is_main_process():
            print(f"Eval Loss {base_loss:.4f} Acc@1 {base_acc1:.2f} Acc@5 {base_acc5:.2f}")
            if ema_model is not None:
                print(f"Eval EMA Loss {ema_loss:.4f} Acc@1 {ema_acc1:.2f} Acc@5 {ema_acc5:.2f}")
        return

    for epoch in range(start_epoch, args.epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        start = time.time()
        train_loss, train_acc = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            steps_per_epoch,
            args.lr,
            args.min_lr,
            warmup_steps,
            total_steps,
            mixup_fn,
            criterion,
            orth_opt,
            ema_model,
        )
        base_loss, base_acc1, base_acc5 = evaluate(model, val_loader, device, val_criterion)
        ema_loss = ema_acc1 = ema_acc5 = None
        if ema_model is not None:
            ema_loss, ema_acc1, ema_acc5 = evaluate(ema_model.module, val_loader, device, val_criterion)
        val_loss = ema_loss if ema_model is not None else base_loss
        val_acc1 = ema_acc1 if ema_model is not None else base_acc1
        val_acc5 = ema_acc5 if ema_model is not None else base_acc5
        elapsed = time.time() - start

        if is_main_process():
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} Acc@1 {train_acc:.2f} | "
                f"Val Loss {base_loss:.4f} Acc@1 {base_acc1:.2f} Acc@5 {base_acc5:.2f} | "
                + (
                    f"EMA Loss {ema_loss:.4f} Acc@1 {ema_acc1:.2f} Acc@5 {ema_acc5:.2f} | "
                    if ema_model is not None else ""
                )
                + f"Time {elapsed:.1f}s"
            )

            best_acc1 = max(best_acc1, val_acc1)
        should_save = epoch % args.save_freq == 0 or epoch == args.epochs - 1

        if is_main_process() and should_save:
            save_checkpoint(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_acc1": best_acc1,
                    "ema": ema_model.state_dict() if ema_model is not None else None,
                    "args": vars(args),
                },
                args.output,
                filename=f"checkpoint_{epoch:03d}.pth",
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
