"""Deterministic DL3DV inference inputs; no training split or synthetic mask generator."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageFolderDataset(Dataset):
    """Pose-free RGB inputs and optional user masks (white means remove).

    Layout: root/scene/images/* and root/scene/masks/<same-stem>.png.
    No masks means that a text segmentation prompt must be supplied to test.py.
    """

    def __init__(
        self,
        root,
        scene_name,
        num_input_views=8,
        target_resolution=(280, 504),
        image_dir="images",
        image_paths=None,
    ):
        self.scene_name = scene_name
        self.scene = Path(root) / scene_name
        self.height, self.width = target_resolution
        paths = image_paths or sorted(
            p
            for p in (self.scene / image_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        if num_input_views < 2 or len(paths) < num_input_views:
            raise ValueError(
                f"Need {num_input_views} distinct views; found {len(paths)}"
            )
        indices = np.rint(np.linspace(0, len(paths) - 1, num_input_views)).astype(int)
        self.paths = [Path(paths[i]) for i in indices]

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index not in (0, -1):
            raise IndexError(index)
        images, masks = [], []
        for path in self.paths:
            with Image.open(path) as image:
                rgb = np.array(
                    image.convert("RGB").resize(
                        (self.width, self.height), Image.Resampling.BILINEAR
                    )
                )
            images.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255)
            mask_path = self.scene / "masks" / f"{path.stem}.png"
            if mask_path.exists():
                with Image.open(mask_path) as mask:
                    array = np.array(
                        mask.convert("L").resize(
                            (self.width, self.height), Image.Resampling.NEAREST
                        )
                    )
                masks.append(torch.from_numpy(array > 127).float()[None])
        if masks and len(masks) != len(images):
            raise ValueError(
                "Masks must be provided for every selected input, or for none of them"
            )
        images = torch.stack(images)
        item = {
            "scene_id": self.scene_name,
            "ref_img": images[0],
            "src_imgs": images[1:],
            "input_names": [p.name for p in self.paths],
            "all_imgs": images,
        }
        if masks:
            masks = torch.stack(masks)
            item.update(ref_mask=masks[0], src_masks=masks[1:])
        return item


class DL3DVSceneDataset(ImageFolderDataset):
    """Read one scene, e.g. scene_name='11K/<scene>', in transforms.json order.

    Images are loaded from images_8. Supply masks/<image-stem>.png or use
    --mask_prompt. Camera annotations are not consumed by the model.
    """

    def __init__(
        self, root, scene_name, num_input_views=4, target_resolution=(280, 504)
    ):
        scene = Path(root) / scene_name
        meta = json.loads((scene / "transforms.json").read_text())
        paths = []
        for frame in meta["frames"]:
            path = scene / "images_8" / Path(frame["file_path"]).name
            if not path.is_file():
                matches = [
                    p
                    for p in path.parent.glob(path.stem + ".*")
                    if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                ]
                if len(matches) != 1:
                    raise FileNotFoundError(path)
                path = matches[0]
            paths.append(path)
        super().__init__(
            root, scene_name, num_input_views, target_resolution, image_paths=paths
        )
