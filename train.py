from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SimpleActionModel

ACTION_COLUMNS = [
    "dx_local_left",
    "dy_local_left",
    "dz_local_left",
    "dx_local_right",
    "dy_local_right",
    "dz_local_right",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frame-pair to action model.")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--train-split", default="trainset")
    parser.add_argument("--val-split", default="valset")
    parser.add_argument("--output-dir", default="checkpoints/simple_action/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sync-timing", action="store_true")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU/PyTorch CUDA install.")
    return device


def configure_gpu(args: argparse.Namespace, device: torch.device) -> None:
    if device.type != "cuda":
        print(f"device={device}")
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    print(f"device={device} name={torch.cuda.get_device_name(device)}")
    print(f"amp={args.amp} tf32={args.tf32} channels_last={args.channels_last}")


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda" and enabled:
        return torch.amp.autocast("cuda")
    return nullcontext()


def move_images(
    start_image: torch.Tensor,
    end_image: torch.Tensor,
    device: torch.device,
    channels_last: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    start_image = start_image.to(device, non_blocking=True)
    end_image = end_image.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        start_image = start_image.contiguous(memory_format=torch.channels_last)
        end_image = end_image.contiguous(memory_format=torch.channels_last)
    return start_image, end_image


def frame_name(frame_idx: int) -> str:
    return f"frame_{frame_idx:06d}.png"


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    data = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return image_to_tensor(image, image_size)


def load_manifest_items(split_dir: Path) -> list[dict]:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


def resolve_manifest_path(split_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else split_dir / path


def read_action_rows(action_csv: Path) -> list[dict[str, str]]:
    with action_csv.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class ActionFrameDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str | Path, image_size: int = 224) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.split = self.split_dir.name
        self.samples: list[dict] = []

        for item in load_manifest_items(self.split_dir):
            action_csv = resolve_manifest_path(self.split_dir, item["action_csv"])
            frame_dir = resolve_manifest_path(self.split_dir, item["frame_dir"])
            for row_idx, row in enumerate(read_action_rows(action_csv)):
                # CSV rows define the exact frame pair used for each action.
                start_idx = int(row["from_sampled_frame_idx"])
                end_idx = int(row["to_sampled_frame_idx"])
                task = item["key"].split("/")[0]
                self.samples.append(
                    {
                        "start_path": frame_dir / frame_name(start_idx),
                        "end_path": frame_dir / frame_name(end_idx),
                        "action": [float(row[name]) for name in ACTION_COLUMNS],
                        "id": f"{item['key']}:{row_idx}",
                        "task": task,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "start_image": load_image(sample["start_path"], self.image_size),
            "end_image": load_image(sample["end_path"], self.image_size),
            "action": torch.tensor(sample["action"], dtype=torch.float32),
            "id": sample["id"],
            "task": sample["task"],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "start_image": torch.stack([item["start_image"] for item in batch]),
        "end_image": torch.stack([item["end_image"] for item in batch]),
        "action": torch.stack([item["action"] for item in batch]),
        "id": [item["id"] for item in batch],
        "task": [item["task"] for item in batch],
    }


def create_loader(
    dataset: torch.utils.data.Dataset,
    args: argparse.Namespace,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate,
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**kwargs)


def run_epoch(
    model: SimpleActionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    channels_last: bool,
    sync_timing: bool,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    data_time_sum = 0.0
    step_time_sum = 0.0

    iterator = tqdm(loader, leave=False, desc="train" if is_train else "val")
    last_time = time.perf_counter()
    for batch in iterator:
        data_time = time.perf_counter() - last_time
        step_start = time.perf_counter()

        start_image, end_image = move_images(
            batch["start_image"],
            batch["end_image"],
            device,
            channels_last,
        )
        action = batch["action"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with autocast_context(device, use_amp):

                loss = model.regression_loss(start_image, end_image, action)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        if sync_timing and device.type == "cuda":
            torch.cuda.synchronize()

        batch_size = action.shape[0]
        total_loss += loss.item() * batch_size
        total_count += batch_size
        step_time = time.perf_counter() - step_start
        data_time_sum += data_time
        step_time_sum += step_time
        avg_data = data_time_sum / max(total_count / batch_size, 1)
        avg_step = step_time_sum / max(total_count / batch_size, 1)
        iterator.set_postfix(
            loss=f"{loss.item():.5f}",
            data=f"{avg_data:.3f}s",
            step=f"{avg_step:.3f}s",
        )
        last_time = time.perf_counter()

    return total_loss / max(total_count, 1)


@torch.no_grad()
def run_validation(
    model: SimpleActionModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    channels_last: bool,
) -> tuple[float, float, dict[str, tuple[float, float, int]]]:
    model.eval()
    l1_sum = 0.0
    l2_sum = 0.0
    value_count = 0

    # per-task accumulators
    from collections import defaultdict
    task_l1: dict[str, float] = defaultdict(float)
    task_l2: dict[str, float] = defaultdict(float)
    task_n: dict[str, int] = defaultdict(int)

    iterator = tqdm(loader, leave=False, desc="val")
    for batch in iterator:
        start_image, end_image = move_images(
            batch["start_image"],
            batch["end_image"],
            device,
            channels_last,
        )
        target = batch["action"].to(device, non_blocking=True)

        with autocast_context(device, use_amp):
            pred = model.sample_actions(start_image, end_image)
        diff = pred - target  # [B, D]

        # overall
        l1_sum += diff.abs().sum().item()
        l2_sum += diff.pow(2).sum().item()
        value_count += diff.numel()

        # per-task
        per_sample_l1 = diff.abs().mean(dim=1)  # [B]
        per_sample_l2 = diff.pow(2).mean(dim=1)  # [B]
        for i, t in enumerate(batch["task"]):
            task_l1[t] += per_sample_l1[i].item()
            task_l2[t] += per_sample_l2[i].item()
            task_n[t] += 1

        iterator.set_postfix(
            l1=f"{l1_sum / max(value_count, 1):.5f}",
            l2=f"{l2_sum / max(value_count, 1):.5f}",
        )

    average_l1 = l1_sum / max(value_count, 1)
    average_l2 = l2_sum / max(value_count, 1)

    # per-task averages
    per_task = {
        t: (task_l1[t] / task_n[t], task_l2[t] / task_n[t], task_n[t])
        for t in task_n
    }
    return average_l1, average_l2, per_task


def create_model(args: argparse.Namespace) -> SimpleActionModel:
    return SimpleActionModel(action_dim=len(ACTION_COLUMNS))
    


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    configure_gpu(args, device)
    pin_memory = device.type == "cuda"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = Path(args.dataset_root)
    train_dataset = ActionFrameDataset(dataset_root / args.train_split, image_size=args.image_size)
    val_dataset = ActionFrameDataset(dataset_root / args.val_split, image_size=args.image_size)

    (output_dir / "action_columns.json").write_text(
        json.dumps(ACTION_COLUMNS, indent=2),
        encoding="utf-8",
    )

    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"batch_size={args.batch_size} num_workers={args.num_workers} prefetch_factor={args.prefetch_factor}")

    train_loader = create_loader(train_dataset, args, shuffle=True, pin_memory=pin_memory)
    val_loader = create_loader(val_dataset, args, shuffle=False, pin_memory=pin_memory)

    model = create_model(args).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            args.amp,
            args.channels_last,
            args.sync_timing,
        )
        val_loss, val_l2, per_task = run_validation(
            model,
            val_loader,
            device,
            args.amp,
            args.channels_last,
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss_l1={val_loss:.6f} val_l2={val_l2:.6f}"
        )
        # per-task summary: best 3, worst 3
        ranked = sorted(per_task.items(), key=lambda x: x[1][0])
        best3 = ranked[:3]
        worst3 = ranked[-3:]
        print(f"  best tasks:   " + " | ".join(
            f"{t}: L1={l1:.4f}" for t, (l1, _, _) in best3))
        print(f"  worst tasks:  " + " | ".join(
            f"{t}: L1={l1:.4f}" for t, (l1, _, _) in worst3))
        # spread
        l1s = [v[0] for v in per_task.values()]
        print(f"  task spread:  min={min(l1s):.4f} max={max(l1s):.4f} "
              f"ratio={max(l1s)/max(min(l1s),1e-8):.1f}x")

        # Convert per_task dict keys to strings for JSON serialisation
        per_task_serialisable = {
            t: {"l1": float(l1), "l2": float(l2), "n": int(n)}
            for t, (l1, l2, n) in per_task.items()
        }
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_l2": val_l2,
            "per_task": per_task_serialisable,
            "args": vars(args),
            "action_columns": ACTION_COLUMNS,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
