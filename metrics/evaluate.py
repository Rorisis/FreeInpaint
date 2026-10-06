"""Object-bounding-box mPSNR, mSSIM, mLPIPS, mFID and mKID."""

import argparse
import json
from pathlib import Path
import re
import tempfile

METRICS = ("mpsnr", "mssim", "mlpips", "mfid", "mkid")
ROOT = Path(__file__).resolve().parents[1]


def crop_bounds(mask, min_size):
    """Preserve the existing IMFine-style largest-contour bbox convention."""
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    # Do not recenter, pad, union components, or multiply the crop by its mask.
    # NumPy/PyTorch slicing naturally clips the right/bottom to the image edge.
    return x, y, x + max(w, min_size), y + max(h, min_size)


def triplets(folder):
    predictions = sorted(
        (p for p in folder.iterdir() if re.fullmatch(r"\d+\.png", p.name)),
        key=lambda p: int(p.stem),
    )
    if not predictions:
        raise ValueError(f"No numbered predictions in {folder}")
    if len({int(p.stem) for p in predictions}) != len(predictions):
        raise ValueError(f"Duplicate numeric prediction IDs in {folder}")
    result = []
    for pred in predictions:
        gt = folder / f"{pred.stem}_gt.png"
        mask = folder / f"{pred.stem}_gt_mask.png"
        if not mask.exists():
            mask = folder / f"{pred.stem}_mask.png"
        if not gt.exists() or not mask.exists():
            raise FileNotFoundError(f"Missing GT or mask for {pred}")
        result.append((pred, gt, mask))
    return result


class MetricSuite:
    def __init__(self, device, distributions=True):
        import pyiqa
        import torch

        self.device = torch.device(device)
        self.paired = {
            name: pyiqa.create_metric(name, device=self.device)
            for name in ("psnr", "ssim", "lpips")
        }
        self.fid = (
            pyiqa.create_metric("fid", device=self.device) if distributions else None
        )

    def distribution_scores(self, pred, gt, args):
        import numpy as np
        from pyiqa.archs.fid_arch import (
            get_folder_features,
            frechet_distance,
            maximum_mean_discrepancy,
        )

        # Same pyiqa clean-mode Inception and preprocessing as the prior evaluator.
        # Each variable-sized bbox is resized independently inside ResizeDataset.
        options = dict(
            model=self.fid.net.model,
            mode="clean",
            device=self.device,
            batch_size=8,
            num_workers=0,
            test_img_size=(299, 299),
            verbose=False,
        )
        a = get_folder_features(str(pred), **options)
        b = get_folder_features(str(gt), **options)
        mfid = frechet_distance(
            a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False)
        )
        np.random.seed(args.seed)
        mkid = maximum_mean_discrepancy(
            a,
            b,
            kernel_type="polynomial",
            num_subsets=args.kid_subsets,
            max_subset_size=args.kid_subset_size,
        )
        return float(mfid), float(mkid)


def evaluate_scene(folder, suite, args):
    import numpy as np
    import torch
    from PIL import Image

    receipt = folder / "result.json"
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        if metadata.get("clean_gt_available") is False:
            raise ValueError(
                "This result has no clean removal GT; reference-based metrics are not meaningful"
            )
        if metadata.get("status") not in {"complete", "success"}:
            raise ValueError("Inference did not complete successfully")
    pairs = triplets(folder)
    values = {name: [] for name in METRICS[:3]}
    skipped = []
    with tempfile.TemporaryDirectory(prefix="freeinpaint_bbox_") as temporary:
        pred_dir, gt_dir = Path(temporary) / "pred", Path(temporary) / "gt"
        pred_dir.mkdir()
        gt_dir.mkdir()
        for pred_path, gt_path, mask_path in pairs:
            with Image.open(pred_path) as image:
                pred = np.array(
                    image.convert("RGB").resize(
                        (args.width, args.height), Image.Resampling.LANCZOS
                    )
                )
            with Image.open(gt_path) as image:
                gt = np.array(
                    image.convert("RGB").resize(
                        (args.width, args.height), Image.Resampling.LANCZOS
                    )
                )
            with Image.open(mask_path) as image:
                mask = np.array(
                    image.convert("L").resize(
                        (args.width, args.height), Image.Resampling.NEAREST
                    )
                )
            bounds = crop_bounds(mask, args.min_crop_size)
            if bounds is None:
                skipped.append(pred_path.stem)
                continue
            x0, y0, x1, y1 = bounds
            pred, gt = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1]
            a = (
                torch.from_numpy(pred.copy())
                .permute(2, 0, 1)[None]
                .float()
                .to(suite.device)
                / 255
            )
            b = (
                torch.from_numpy(gt.copy())
                .permute(2, 0, 1)[None]
                .float()
                .to(suite.device)
                / 255
            )
            with torch.no_grad():
                for name, metric in suite.paired.items():
                    values["m" + name].append(float(metric(a, b).mean().item()))
            # Save native bbox sizes; no additional resize before the metric backend.
            Image.fromarray(pred).save(pred_dir / pred_path.name)
            Image.fromarray(gt).save(gt_dir / pred_path.name)
        if not values["mpsnr"]:
            raise ValueError("All evaluation masks are empty")
        scores = {key: float(np.mean(v)) for key, v in values.items()}
        scores.update(mfid=None, mkid=None)
        if not args.no_distribution_metrics and len(values["mpsnr"]) >= 2:
            with torch.no_grad():
                scores["mfid"], scores["mkid"] = suite.distribution_scores(
                    pred_dir, gt_dir, args
                )
    return dict(
        metrics=scores,
        num_images=len(pairs),
        num_crops=len(values["mpsnr"]),
        empty_mask_ids=skipped,
        protocol="imfine_bbox_v1",
        resize=[args.width, args.height],
        min_crop_size=args.min_crop_size,
        fid_mode="pyiqa_clean",
        kid_subset_size=args.kid_subset_size,
        kid_subsets=args.kid_subsets,
        seed=args.seed,
    )


def save_report(folder, result):
    def display(value):
        return "N/A" if value is None else f"{value:.6f}"

    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "metrics.txt").write_text(
        "protocol: imfine_bbox_v1\n"
        + "\n".join(f"{name}: {display(result['metrics'][name])}" for name in METRICS)
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root",
        required=True,
        help="One scene directory or a root searched recursively",
    )
    parser.add_argument("--dataset", required=True, choices=["spinnerf", "usid"])
    parser.add_argument("--config", default=str(ROOT / "config/test.yaml"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--min_crop_size", type=int)
    parser.add_argument("--kid_subset_size", type=int)
    parser.add_argument("--kid_subsets", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no_distribution_metrics", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    import yaml

    cfg = yaml.safe_load(Path(args.config).read_text())["metrics"]
    args.width = args.width or cfg["resize_width"]
    args.height = args.height or cfg["resize_height"]
    args.min_crop_size = (
        cfg["min_crop_size"][args.dataset]
        if args.min_crop_size is None
        else args.min_crop_size
    )
    args.kid_subset_size = args.kid_subset_size or cfg["kid_subset_size"]
    args.kid_subsets = args.kid_subsets or cfg["kid_subsets"]
    args.seed = cfg["seed"] if args.seed is None else args.seed
    if args.min_crop_size < 1 or args.kid_subset_size < 2 or args.kid_subsets < 1:
        parser.error("Invalid crop size or KID sampling parameters")
    root = Path(args.results_root).resolve()
    folders = sorted({p.parent for p in root.rglob("*_gt.png")})
    if not folders:
        parser.error("No result folders found")
    if args.dry_run:
        for folder in folders:
            print(f"{folder}: {len(triplets(folder))} image pairs")
        return 0
    suite = MetricSuite(args.device, not args.no_distribution_metrics)
    results, failures = {}, {}
    for folder in folders:
        try:
            result = evaluate_scene(folder, suite, args)
            save_report(folder, result)
            results[str(folder.relative_to(root))] = result
            print(f"[Saved] {folder / 'metrics.txt'}")
        except Exception as error:
            failures[str(folder)] = f"{type(error).__name__}: {error}"
            print(f"[Failed] {folder}: {error}")
    averages = {}
    for metric in METRICS:
        values = [
            r["metrics"][metric]
            for r in results.values()
            if r["metrics"][metric] is not None
        ]
        averages[metric] = sum(values) / len(values) if values else None
    report = dict(scene_average=averages, scenes=results, failures=failures)
    (root / "evaluation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [f"successful_scenes: {len(results)}", f"failed_scenes: {len(failures)}"]
    for scene, result in results.items():
        lines.append(f"\n[{scene}]")
        lines.extend(f"{key}: {result['metrics'][key]}" for key in METRICS)
    lines.append("\n[scene average]")
    lines.extend(f"{key}: {value}" for key, value in averages.items())
    lines.extend(f"FAILED {scene}: {error}" for scene, error in failures.items())
    (root / "evaluation_summary.txt").write_text("\n".join(lines) + "\n")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
