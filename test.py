from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import  SimpleActionModel

DEFAULT_ACTION_COLUMNS = [
    "dx_local_left",
    "dy_local_left",
    "dz_local_left",
    "droll_left_rad",
    "dpitch_left_rad",
    "dyaw_left_rad",
    "dleft_gripper",
    "dx_local_right",
    "dy_local_right",
    "dz_local_right",
    "droll_right_rad",
    "dpitch_right_rad",
    "dyaw_right_rad",
    "dright_gripper",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict actions for dataset image pairs only.")
    parser.add_argument("--checkpoint", default="checkpoints/v2_magweight/best.pt")
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument("--test-split", default="testset")
    parser.add_argument("--output-csv", default="result/test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU/PyTorch CUDA install.")
    return device


def frame_index(path: str | Path) -> int:
    stem = Path(path).stem
    return int(stem.replace("frame_", ""))


def load_manifest_items(split_dir: Path) -> list[dict]:
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))["items"]


def resolve_manifest_path(split_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else split_dir / path


def image_to_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    image = image.convert("RGB").resize((image_size, image_size))
    data = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (data - mean) / std


def load_image(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as image:
        return image_to_tensor(image, image_size)


class RealTestImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, split_dir: str | Path, image_size: int = 224) -> None:
        self.split_dir = Path(split_dir)
        self.image_size = image_size
        self.samples: list[dict] = []

        for item in load_manifest_items(self.split_dir):
            frames = [resolve_manifest_path(self.split_dir, path) for path in item["frames"]]
            frames = sorted(frames, key=frame_index)
            for pair_index, (start_path, end_path) in enumerate(zip(frames[:-1], frames[1:])):
                # Only image paths are used here; action CSV files are intentionally ignored.
                self.samples.append(
                    {
                        "id": f"{item['key']}:{pair_index}",
                        "key": item["key"],
                        "pair_index": pair_index,
                        "start_path": start_path,
                        "end_path": end_path,
                        "start_idx": frame_index(start_path),
                        "end_idx": frame_index(end_path),
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "start_image": load_image(sample["start_path"], self.image_size),
            "end_image": load_image(sample["end_path"], self.image_size),
            "id": sample["id"],
            "key": sample["key"],
            "pair_index": sample["pair_index"],
            "start_frame": sample["start_path"].as_posix(),
            "end_frame": sample["end_path"].as_posix(),
            "from_sampled_frame_idx": sample["start_idx"],
            "to_sampled_frame_idx": sample["end_idx"],
        }


def collate(batch: list[dict]) -> dict:
    return {
        "start_image": torch.stack([item["start_image"] for item in batch]),
        "end_image": torch.stack([item["end_image"] for item in batch]),
        "id": [item["id"] for item in batch],
        "key": [item["key"] for item in batch],
        "pair_index": [item["pair_index"] for item in batch],
        "start_frame": [item["start_frame"] for item in batch],
        "end_frame": [item["end_frame"] for item in batch],
        "from_sampled_frame_idx": [item["from_sampled_frame_idx"] for item in batch],
        "to_sampled_frame_idx": [item["to_sampled_frame_idx"] for item in batch],
    }


def load_model(
    args: argparse.Namespace,
    device: torch.device,
):
    checkpoint = torch.load(args.checkpoint, map_location=device)
    action_columns = checkpoint.get("action_columns", DEFAULT_ACTION_COLUMNS)

    model = SimpleActionModel(action_dim=len(action_columns)).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, action_columns


@torch.no_grad()
def predict(
    args: argparse.Namespace,
    model: SimpleActionModel,
    action_columns: list[str],
    device: torch.device,
) -> list[dict]:
    dataset = RealTestImagePairDataset(
        Path(args.dataset_root) / args.test_split,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate,
    )

    rows: list[dict] = []
    for batch in tqdm(loader, desc=args.test_split):
        start_image = batch["start_image"].to(device, non_blocking=True)
        end_image = batch["end_image"].to(device, non_blocking=True)
        pred = model.sample_actions(start_image, end_image).cpu()

        for i in range(pred.shape[0]):
            row = {
                "id": batch["id"][i],
                "key": batch["key"][i],
                "pair_index": batch["pair_index"][i],
                "start_frame": batch["start_frame"][i],
                "end_frame": batch["end_frame"][i],
                "from_sampled_frame_idx": batch["from_sampled_frame_idx"][i],
                "to_sampled_frame_idx": batch["to_sampled_frame_idx"][i],
            }
            for name, value in zip(action_columns, pred[i].tolist()):
                row[name] = value
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model, action_columns = load_model(args, device)
    rows = predict(args, model, action_columns, device)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "key",
        "pair_index",
        "start_frame",
        "end_frame",
        "from_sampled_frame_idx",
        "to_sampled_frame_idx",
        *action_columns,
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} predicted actions to {output_csv}")


if __name__ == "__main__":
    main()
