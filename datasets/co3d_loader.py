"""CO3D image/mask loader for qualitative FreeInpaint evaluation.

This loader intentionally does not share code with the GS25/COLMAP loader.
FreeInpaint does not require input camera poses, and the locally downloaded
CO3D sequences may not include the optional CO3D camera annotation archives.
Consequently, this dataset returns only RGB inputs and CO3D's native object
masks. Camera parameters used for visualization are predicted by DA3.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _natural_key(path: Path) -> List[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path.name)
    ]


def _frame_key(path: Path) -> str:
    # CO3D normally uses frame000001.jpg and frame000001.png.  Repeatedly
    # stripping suffixes also tolerates names such as frame.jpg.mask.png.
    name = path.name
    while Path(name).suffix:
        name = Path(name).stem
    return name


def _list_media(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        return {}
    files = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )
    return {_frame_key(path): path for path in files}


def _uniform_indices(length: int, count: int) -> List[int]:
    if count > length:
        raise ValueError(
            f"Requested {count} views, but only {length} valid frames are available."
        )
    if count == 1:
        return [length // 2]
    # round(linspace) is deterministic and includes both ends of the sequence.
    indices = np.rint(np.linspace(0, length - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError(
            f"Uniform sampling unexpectedly produced duplicate indices: {indices.tolist()}"
        )
    return indices.tolist()


class CO3DSequenceDataset(Dataset):
    """One deterministic CO3D sequence from one object category.

    ``num_input_views`` includes the reference view.  When ``sequence_name`` is
    omitted, the sequence with the largest number of matched image/mask frames
    is selected (ties are resolved lexicographically).  This rule is automatic,
    reproducible, and does not cherry-pick based on model output.
    """

    def __init__(
        self,
        root: str,
        category: str,
        sequence_name: Optional[str] = None,
        num_input_views: int = 16,
        target_resolution: Tuple[int, int] = (280, 504),
        min_mask_ratio: float = 1e-4,
        max_mask_ratio: float = 0.98,
    ) -> None:
        self.root = Path(root).expanduser()
        self.category = category
        self.num_input_views = int(num_input_views)
        self.target_h, self.target_w = target_resolution
        self.min_mask_ratio = float(min_mask_ratio)
        self.max_mask_ratio = float(max_mask_ratio)

        if self.num_input_views < 2:
            raise ValueError(
                "CO3D evaluation requires at least 2 input views (1 reference + sources)."
            )
        category_dir = self.root / category
        if not category_dir.is_dir():
            available = self.available_categories(self.root)
            preview = ", ".join(available[:12])
            raise FileNotFoundError(
                f"CO3D category directory not found: {category_dir}. "
                f"Available categories include: {preview}"
            )

        if sequence_name:
            sequence_dir = category_dir / sequence_name
            if not sequence_dir.is_dir():
                raise FileNotFoundError(
                    f"CO3D sequence directory not found: {sequence_dir}"
                )
        else:
            sequence_dir = self._select_sequence(category_dir)

        self.sequence_name = sequence_dir.name
        pairs = self._matched_pairs(sequence_dir)
        valid_pairs = [pair for pair in pairs if self._mask_is_valid(pair[1])]
        if len(valid_pairs) < self.num_input_views:
            raise ValueError(
                f"{category}/{self.sequence_name} has {len(valid_pairs)} valid image/mask pairs; "
                f"{self.num_input_views} views were requested. "
                "Choose another --sequence_name or reduce --num_input_views."
            )

        selected = [
            valid_pairs[index]
            for index in _uniform_indices(len(valid_pairs), self.num_input_views)
        ]
        self.image_paths = [pair[0] for pair in selected]
        self.mask_paths = [pair[1] for pair in selected]
        self.frame_names = [_frame_key(path) for path in self.image_paths]
        print(
            f"[CO3D] category={self.category}, sequence={self.sequence_name}, "
            f"views={self.num_input_views}, available_valid_frames={len(valid_pairs)}"
        )

    @staticmethod
    def available_categories(root: Path | str) -> List[str]:
        root = Path(root).expanduser()
        if not root.is_dir():
            return []
        return sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith("_")
        )

    @staticmethod
    def _matched_pairs(sequence_dir: Path) -> List[Tuple[Path, Path]]:
        images = _list_media(sequence_dir / "images")
        masks = _list_media(sequence_dir / "masks")
        keys = sorted(
            set(images).intersection(masks), key=lambda key: _natural_key(Path(key))
        )
        return [(images[key], masks[key]) for key in keys]

    def _select_sequence(self, category_dir: Path) -> Path:
        candidates = []
        for sequence_dir in sorted(
            (path for path in category_dir.iterdir() if path.is_dir()), key=_natural_key
        ):
            count = len(self._matched_pairs(sequence_dir))
            if count >= self.num_input_views:
                candidates.append((count, sequence_dir.name, sequence_dir))
        if not candidates:
            raise ValueError(
                f"No sequence in {category_dir} has at least {self.num_input_views} matched image/mask frames."
            )
        # Prefer maximum usable coverage; the secondary key makes the choice stable.
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return candidates[0][2]

    def _mask_is_valid(self, mask_path: Path) -> bool:
        with Image.open(mask_path) as image:
            mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        ratio = float((mask > 0.5).mean())
        return self.min_mask_ratio <= ratio <= self.max_mask_ratio

    def __len__(self) -> int:
        return 1

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (self.target_w, self.target_h), Image.Resampling.BILINEAR
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_mask(self, path: Path) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("L").resize(
                (self.target_w, self.target_h), Image.Resampling.NEAREST
            )
            array = (np.asarray(image, dtype=np.float32) / 255.0 > 0.5).astype(
                np.float32
            )
        return torch.from_numpy(array).unsqueeze(0).contiguous()

    def __getitem__(self, index: int) -> Dict[str, object]:
        if index not in (0, -1):
            raise IndexError(index)
        images = torch.stack([self._load_image(path) for path in self.image_paths])
        masks = torch.stack([self._load_mask(path) for path in self.mask_paths])
        return {
            "scene_id": f"{self.category}/{self.sequence_name}",
            "category": self.category,
            "sequence_name": self.sequence_name,
            "frame_names": list(self.frame_names),
            "image_paths": [str(path) for path in self.image_paths],
            "mask_paths": [str(path) for path in self.mask_paths],
            "ref_img": images[0],
            "ref_mask": masks[0],
            "src_imgs": images[1:],
            "src_masks": masks[1:],
            "all_imgs": images,
            "all_masks": masks,
        }
