import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from datasets.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
)
from typing import Tuple

# =============================================================================
# USID Dataset Loader
# =============================================================================


class USIDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
        target_resolution: Tuple[int, int] = (378, 504),
        num_target_views: int = 20,
        scene_name="cone",
    ):

        self.root = root
        self.target_h, self.target_w = target_resolution
        self.num_target_views = num_target_views
        self.scene_name = scene_name
        # USID typically has serveral scenes
        all_scenes = sorted([d for d in os.listdir(root)])

        if scene_name is not None:
            self.scenes = [scene for scene in all_scenes if scene == scene_name]
        else:
            self.scenes = all_scenes

        print(f"[USID] Split: {split}, Scenes: {self.scenes}")

    def __len__(self):
        # One item is one complete benchmark scene.
        return len(self.scenes)

    def __getitem__(self, idx):
        # Map global idx to scene idx
        scene_id = self.scenes[idx % len(self.scenes)]
        scene_path = os.path.join(self.root, scene_id)

        # 1. Parse COLMAP
        colmap_dir = os.path.join(scene_path, "sparse", "0")
        if not os.path.exists(colmap_dir):
            # Fallback if structure is slightly different (some versions have just 'sparse')
            colmap_dir = os.path.join(scene_path, "sparse")

        try:
            cam_intrinsics = read_intrinsics_binary(
                os.path.join(colmap_dir, "cameras.bin")
            )
            cam_extrinsics = read_extrinsics_binary(
                os.path.join(colmap_dir, "images.bin")
            )
        except:
            cam_intrinsics = read_intrinsics_text(
                os.path.join(colmap_dir, "cameras.txt")
            )
            cam_extrinsics = read_extrinsics_text(
                os.path.join(colmap_dir, "images.txt")
            )

        # 2. Sort images by name to separate "No Object" (GT) and "With Object" (Input)
        # USID structure: sorted by timestamp
        name_to_id = {img.name: img_id for img_id, img in cam_extrinsics.items()}

        # all_image_names = sorted([f for f in os.listdir(images_ori_dir) if f.endswith(('.JPG', '.png'))])
        images_dir = os.path.join(scene_path, "images")
        reference_dir = os.path.join(scene_path, "reference")
        masks_dir = os.path.join(scene_path, "object_masks")
        test_images_dir = os.path.join(scene_path, "test_images")

        valid_image_names = []
        for name in name_to_id.keys():
            img_path = os.path.join(images_dir, os.path.basename(name))
            mask_path = os.path.join(masks_dir, os.path.basename(name))
            if os.path.exists(img_path) and os.path.exists(mask_path):
                valid_image_names.append(name)

        reference_image_names = sorted(os.listdir(reference_dir))[0]
        test_image_names = sorted(os.listdir(test_images_dir))

        if len(valid_image_names) == 0:
            raise RuntimeError(
                f"No valid images found in images_2 for scene {scene_id}"
            )
        all_image_names = sorted(valid_image_names)

        # Heuristic from user description:
        num_frames = len(all_image_names)
        clean_names = test_image_names
        object_names = all_image_names

        num_input_views = 8
        stride = max(1, len(object_names) // num_input_views)
        # 3. Sampling Strategy
        # Input: Sample 4 views from "With Object" set
        # - 1 Ref, 3 Src
        if len(object_names) < num_input_views:
            raise ValueError(f"Scene {scene_id} has too few object images")
        # input_names = random.sample(object_names, 4)
        input_names = sorted(object_names)[::stride][
            :num_input_views
        ]  # For reproducibility during testing
        ref_name = reference_image_names
        src_names = input_names[: (num_input_views - 1)]

        # Target: Sample N views from "No Object" set
        # tgt_names = clean_names[:self.num_target_views]
        tgt_names = clean_names

        # 4. Load Data Helper
        def load_view(
            img_name, img_path, load_mask=False, is_ref=False, is_tgt=False, idx=0
        ):
            if img_name not in name_to_id:
                # Handle filename extension differences (jpg vs png).
                base = os.path.splitext(img_name)[0]
                found = False
                for k in name_to_id.keys():
                    if os.path.splitext(k)[0] == base:
                        img_name = k
                        found = True
                        break
                if not found:
                    raise ValueError(f"Image {img_name} not found in COLMAP data")

            img_id = name_to_id[img_name]
            extr = cam_extrinsics[img_id]
            intr = cam_intrinsics[extr.camera_id]

            height = intr.height
            width = intr.width

            if is_ref:
                self.target_h = int(
                    self.target_w / width * height
                )  # Maintain aspect ratio
                self.target_h = self.target_h - (self.target_h % 14)
            # print(f"[USID] Loading View: {extr.name}, Original Size: ({width}, {height})")
            x_scale = self.target_w / width
            y_scale = self.target_h / height

            if intr.model == "SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[0]  # SIMPLE_PINHOLE uses a single focal length.
                cx = intr.params[1]
                cy = intr.params[2]

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                cx = intr.params[2]
                cy = intr.params[3]

            else:
                assert False, (
                    "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
                )

            # cx = (cx - width / 2) / width * 2
            # cy = (cy - height / 2) / height * 2

            K = np.array(
                [
                    [focal_length_x * x_scale, 0.0, cx * x_scale],
                    [0.0, focal_length_y * y_scale, cy * y_scale],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            # --- Image Path (Use images_2) ---
            image_name = os.path.basename(img_path).split(".")[0]

            pil_img = Image.open(img_path).convert("RGB")

            # --- Resize Image ---
            pil_img = pil_img.resize(
                (self.target_w, self.target_h), Image.Resampling.BILINEAR
            )
            img_tensor = (
                torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            )

            # --- Intrinsics (K) ---
            # COLMAP params are for ORIGINAL resolution
            # We are using images_4 (1/4 scale) AND resizing to (target_w, target_h)

            R_w2c = qvec2rotmat(extr.qvec)
            t_w2c = extr.tvec

            w2c = np.eye(4)
            w2c[:3, :3] = R_w2c
            w2c[:3, 3] = t_w2c

            c2w = np.linalg.inv(w2c)

            c2w_tensor = torch.from_numpy(c2w).float()
            ext_tensor = torch.from_numpy(w2c).float()
            K = torch.from_numpy(K).float()

            mask_tensor = None
            if load_mask:
                if not is_tgt:
                    final_mask_path = os.path.join(
                        scene_path, "object_masks", os.path.basename(extr.name)
                    )
                else:
                    final_mask_path = os.path.join(
                        scene_path,
                        f"{self.scene_name}_masks",
                        str(idx).zfill(5) + ".png",
                    )
                    # print(f"[USID] Loading Target Mask: {final_mask_path}")

                if final_mask_path:
                    mask_pil = Image.open(final_mask_path).convert("L")  # Grayscale
                    mask_pil = mask_pil.resize(
                        (self.target_w, self.target_h), Image.Resampling.NEAREST
                    )
                    mask_tensor = torch.from_numpy(np.array(mask_pil)).float() / 255.0
                    mask_tensor = (mask_tensor > 0.5).float().unsqueeze(0)  # [1, H, W]
                else:
                    # Fallback: No mask found (maybe object is out of frame?), return zeros
                    mask_tensor = torch.zeros(1, self.target_h, self.target_w)

            # Fake depth (USID doesn't provide dense depth easily, sparse is hard to map)
            depth_tensor = torch.zeros(1, self.target_h, self.target_w)

            return img_tensor, c2w_tensor, ext_tensor, K, depth_tensor, mask_tensor

        # 5. Load Batches
        # Ref
        ref_img, ref_c2w, ref_ext, ref_K, ref_depth, ref_mask = load_view(
            ref_name,
            img_path=os.path.join(reference_dir, ref_name),
            load_mask=True,
            is_ref=True,
        )

        # Src
        src_data = [
            load_view(n, img_path=os.path.join(images_dir, n), load_mask=True)
            for n in src_names
        ]
        src_imgs = torch.stack([x[0] for x in src_data])
        src_c2ws = torch.stack([x[1] for x in src_data])
        src_exts = torch.stack([x[2] for x in src_data])
        src_Ks = torch.stack([x[3] for x in src_data])
        src_depths = torch.stack([x[4] for x in src_data])
        src_masks = torch.stack([x[5] for x in src_data])  # GT Object Masks

        # Target
        tgt_data = [
            load_view(
                n,
                img_path=os.path.join(test_images_dir, n),
                load_mask=True,
                is_tgt=True,
                idx=idx,
            )
            for idx, n in enumerate(tgt_names)
        ]
        tgt_imgs = torch.stack([x[0] for x in tgt_data])
        tgt_c2ws = torch.stack([x[1] for x in tgt_data])
        tgt_exts = torch.stack([x[2] for x in tgt_data])
        tgt_Ks = torch.stack([x[3] for x in tgt_data])
        tgt_depths = torch.stack([x[4] for x in tgt_data])
        tgt_masks = torch.stack([x[5] for x in tgt_data])

        return {
            "scene_id": scene_id,
            "input_names": [ref_name] + src_names,
            "target_names": tgt_names,
            "ref_img": ref_img,
            "ref_c2w": ref_c2w,
            "ref_ext": ref_ext,
            "ref_K": ref_K,
            "ref_depth": ref_depth,
            "ref_mask": ref_mask,
            "src_imgs": src_imgs,
            "src_c2ws": src_c2ws,
            "src_exts": src_exts,
            "src_Ks": src_Ks,
            "src_depths": src_depths,
            "src_masks": src_masks,
            "tgt_imgs": tgt_imgs,
            "tgt_c2ws": tgt_c2ws,
            "tgt_exts": tgt_exts,
            "tgt_Ks": tgt_Ks,
            "tgt_depths": tgt_depths,
            "tgt_masks": tgt_masks,
        }
