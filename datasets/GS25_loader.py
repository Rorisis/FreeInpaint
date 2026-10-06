"""GS25 / Omni3DEdit RGB inputs, read-only from directories or scene ZIPs."""

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GS25Dataset(Dataset):
    """Uniformly select input RGBs without depending on COLMAP registration.

    Accept root/scene/input/ or root/scene.zip containing scene/input/.
    Optional scene/masks/<stem>.png are white for removal. Otherwise test.py
    generates masks from --mask_prompt. Archives are never extracted.
    """

    def __init__(
        self, root, scene_name, target_resolution=(280, 504), num_input_views=8
    ):
        self.root, self.scene_name = Path(root), scene_name
        self.height, self.width = target_resolution
        self.archive = None
        folder = self.root / scene_name / "input"
        paths = sorted(
            p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(paths) < num_input_views and (self.root / f"{scene_name}.zip").is_file():
            self.archive = self.root / f"{scene_name}.zip"
            with ZipFile(self.archive) as archive:
                paths = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith(f"{scene_name}/input/")
                    and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                )
        if num_input_views < 2 or len(paths) < num_input_views:
            raise ValueError(
                f"{scene_name}: requested {num_input_views} views, found {len(paths)} input images"
            )
        indices = np.linspace(0, len(paths) - 1, num_input_views, dtype=int)
        self.paths = [paths[i] for i in indices]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError(index)
        images, masks = [], []
        archive = ZipFile(self.archive) if self.archive else None
        try:
            members = set(archive.namelist()) if archive else set()
            for path in self.paths:
                with archive.open(str(path)) if archive else open(path, "rb") as handle:
                    with Image.open(handle) as image:
                        array = np.array(
                            image.convert("RGB").resize(
                                (self.width, self.height), Image.Resampling.BILINEAR
                            )
                        )
                images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255)
                mask_name = f"{self.scene_name}/masks/{Path(path).stem}.png"
                exists = (
                    mask_name in members
                    if archive
                    else (self.root / mask_name).is_file()
                )
                if exists:
                    with (
                        archive.open(mask_name)
                        if archive
                        else open(self.root / mask_name, "rb")
                    ) as handle:
                        with Image.open(handle) as image:
                            array = np.array(
                                image.convert("L").resize(
                                    (self.width, self.height), Image.Resampling.NEAREST
                                )
                            )
                    masks.append(torch.from_numpy(array > 127).float()[None])
        finally:
            if archive:
                archive.close()
        if masks and len(masks) != len(images):
            raise ValueError("GS25 masks must cover every selected input, or none")
        images = torch.stack(images)
        item = dict(
            scene_id=self.scene_name,
            ref_img=images[0],
            src_imgs=images[1:],
            all_imgs=images,
            input_names=[Path(p).name for p in self.paths],
        )
        if masks:
            item.update(ref_mask=masks[0], src_masks=torch.stack(masks[1:]))
        return item
