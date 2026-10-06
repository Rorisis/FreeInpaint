import argparse
import gc
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["spinnerf", "usid", "gs25", "co3d", "dl3dv"],
    )
    parser.add_argument("--config", default=str(ROOT / "config/test.yaml"))
    parser.add_argument("--checkpoint", "--ckpt_path", dest="checkpoint")
    parser.add_argument("--data_root")
    parser.add_argument("--output_root")
    parser.add_argument("--scene_name")
    parser.add_argument("--category")
    parser.add_argument("--sequence_name")
    parser.add_argument("--num_input_views", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument(
        "--context_json", help="SPIn-NeRF ordered input stems; first is the reference"
    )
    parser.add_argument(
        "--reference_image",
        help="Optional already-inpainted RGB for the selected reference camera",
    )
    parser.add_argument("--refresh_reference", action="store_true")
    parser.add_argument("--positive_prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--mask_prompt",
        default="",
        help="LangSAM prompt for inputs without supplied masks",
    )
    parser.add_argument(
        "--task",
        choices=["object-removal", "object-replacement"],
        default="object-removal",
    )
    parser.add_argument(
        "--num_refinement_iters", type=int, help="Total forward passes (1 disables STR)"
    )
    parser.add_argument(
        "--support_canvas", choices=["source", "rendered", "composite"], default="source",
        help="STR canvas; composite uses source background and rendered masked pixels",
    )
    parser.add_argument(
        "--ref_view_strategy", choices=["default", "first", "saddle_balanced"]
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--save_gs", action="store_true", help="Export the final Gaussian field as PLY"
    )
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--trajectory", choices=["orbit", "spiral"], default="orbit")
    parser.add_argument("--n_frames", type=int, default=120)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--radius", type=float, default=1.1)
    parser.add_argument(
        "--check_config", action="store_true", help="No model imports or downloads"
    )
    parser.add_argument(
        "--check_data",
        action="store_true",
        help="Read/validate inputs only; no inference",
    )
    return parser


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rgb(path, tensor):
    import numpy as np
    from PIL import Image

    array = tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(path)


def save_mask(path, tensor):
    import numpy as np
    from PIL import Image

    Image.fromarray(
        (tensor.detach().cpu().squeeze().numpy() > 0.5).astype(np.uint8) * 255
    ).save(path)


def load_scene(args, cfg):
    section = cfg[f"{args.dataset}_dataset"]
    common = dict(root=args.data_root, target_resolution=(args.height, args.width))
    if args.dataset == "spinnerf":
        from datasets.SPInNeRF_loader import SPInNeRFSceneDataset

        loader = SPInNeRFSceneDataset(
            **common, scene_name=args.scene_name, context_json=section["context_json"]
        )
    elif args.dataset == "usid":
        from datasets.USID_loader import USIDDataset

        loader = USIDDataset(**common, scene_name=args.scene_name)
    elif args.dataset == "co3d":
        from datasets.co3d_loader import CO3DSequenceDataset

        loader = CO3DSequenceDataset(
            **common,
            category=args.category,
            sequence_name=args.sequence_name,
            num_input_views=args.num_input_views,
        )
    elif args.dataset == "dl3dv":
        from datasets.DL3DV_loader import DL3DVSceneDataset

        loader = DL3DVSceneDataset(
            **common, scene_name=args.scene_name, num_input_views=args.num_input_views
        )
    elif args.dataset == "gs25":
        from datasets.GS25_loader import GS25Dataset

        loader = GS25Dataset(
            **common,
            scene_name=args.scene_name,
            num_input_views=args.num_input_views,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if len(loader) != 1:
        raise ValueError(
            f"Expected one scene; found {len(loader)}. Check --scene_name."
        )
    item = loader[0]
    item.setdefault(
        "input_names",
        item.get("frame_names") or [item.get("ref_name")] + item.get("src_names", []),
    )
    # The USID adapter derives height from the reference aspect ratio, rounded to patches.
    args.height, args.width = item["ref_img"].shape[-2:]
    if args.height % 14 or args.width % 14:
        raise ValueError("DA3 input height and width must be divisible by 14")
    item["clean_gt_available"] = (
        args.dataset in {"spinnerf", "usid"} and args.task == "object-removal"
    )
    return item


def prepare_masks(item, args, cfg):
    import cv2
    import numpy as np
    import torch

    if "src_masks" not in item:
        if not args.mask_prompt:
            raise ValueError("Provide masks for every input or specify --mask_prompt")
        from lang_sam import LangSAM
        from torchvision.transforms.functional import to_pil_image

        segmentor = LangSAM()
        masks = []
        for img in torch.cat([item["ref_img"][None], item["src_imgs"]]):
            result = segmentor.predict([to_pil_image(img)], [args.mask_prompt])[0][
                "masks"
            ]
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            elif isinstance(result, list):
                result = [
                    m.detach().cpu().numpy() if isinstance(m, torch.Tensor) else m
                    for m in result
                ]
            result = np.asarray(result)
            if result.size == 0:
                raise ValueError(
                    f"No object detected for {args.mask_prompt!r}; provide explicit masks"
                )
            mask = np.any(result.reshape(-1, args.height, args.width) > 0.5, axis=0)
            masks.append(torch.from_numpy(mask).float()[None])
        item["ref_mask"], item["src_masks"] = masks[0], torch.stack(masks[1:])
        del segmentor
        gc.collect()
        torch.cuda.empty_cache()
    # Undilated input masks are retained for qualitative output; benchmark target masks
    # are separate and are never modified by the inpainting dilation.
    item["original_input_masks"] = torch.cat(
        [item["ref_mask"][None], item["src_masks"]]
    ).clone()
    size = int(cfg[f"{args.dataset}_dataset"]["mask_dilation_kernel"])
    if size < 1 or size % 2 != 1:
        raise ValueError("mask_dilation_kernel must be positive and odd")
    shape = cv2.MORPH_ELLIPSE if args.dataset == "co3d" else cv2.MORPH_RECT
    kernel = cv2.getStructuringElement(shape, (size, size))
    for key in ("ref_mask", "src_masks"):
        value = item[key]
        arrays = value.numpy().reshape(-1, args.height, args.width)
        masks = np.stack(
            [cv2.dilate((m > 0.5).astype(np.uint8), kernel) for m in arrays]
        )
        item[key] = torch.from_numpy(masks.reshape(value.shape)).float()


def prepare_reference(item, args, cfg):
    import numpy as np
    import torch
    from PIL import Image
    from utils.io import digest_json, file_digest, write_json

    def read(path):
        with Image.open(path) as image:
            image = image.convert("RGB").resize(
                (args.width, args.height), Image.Resampling.BILINEAR
            )
            return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255

    if args.reference_image:
        return read(args.reference_image), {
            "source": str(args.reference_image),
            "sha256": file_digest(args.reference_image),
        }
    if args.dataset == "usid" and args.task == "object-removal":
        return item["ref_img"], {
            "source": "dataset/reference",
            "name": item["input_names"][0],
        }
    import hashlib

    spec = dict(
        dataset=args.dataset,
        scene=item["scene_id"],
        views=item["input_names"],
        task=args.task,
        seed=args.seed,
        positive=args.positive_prompt,
        negative=args.negative_prompt,
        resolution=[args.height, args.width],
        diffusion=cfg["diffusion"],
        image=hashlib.sha256(item["ref_img"].numpy().tobytes()).hexdigest(),
        mask=hashlib.sha256(item["ref_mask"].numpy().tobytes()).hexdigest(),
    )
    folder = (
        Path(cfg["reference"]["cache_dir"])
        / args.dataset
        / item["scene_id"]
        / digest_json(spec)[:20]
    )
    folder.mkdir(parents=True, exist_ok=True)
    path, receipt = folder / "reference.png", folder / "reference.json"
    import fcntl

    with open(folder / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.refresh_reference or not (path.exists() and receipt.exists()):
            from guidance.inpainting import load_diffusion_output

            seed_everything(args.seed)
            with torch.no_grad():
                generated = load_diffusion_output(
                    item["ref_img"][None].cuda(),
                    item["ref_mask"][None].cuda(),
                    args.positive_prompt,
                    args.negative_prompt,
                    cfg,
                    args.task,
                    digest_json(spec)[:20],
                    "cuda",
                )
            save_rgb(path, generated[0])
            write_json(receipt, {"spec": spec, "sha256": file_digest(path)})
            del generated
            gc.collect()
            torch.cuda.empty_cache()
        metadata = json.loads(receipt.read_text())
        if metadata["spec"] != spec or metadata["sha256"] != file_digest(path):
            raise RuntimeError(f"Reference cache changed or corrupted: {folder}")
    # Read back PNG even on first generation so all later runs use the same pixels.
    return read(path), {"source": str(path), **metadata}


def load_model(args, cfg):
    import torch
    from network.freeinpaint import FreeInpaint

    options = cfg["learnable_mask"]
    model = FreeInpaint(
        cfg["model"]["backbone"],
        torch.device("cuda"),
        mask_hidden_dim=options["hidden_dim"],
        mask_alpha=options["alpha"],
        mask_beta=options["beta"],
        mask_epsilon=options["epsilon"],
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    # Fail loudly on an incompatible checkpoint rather than silently dropping LMA weights.
    model.load_state_dict(state, strict=True)
    return model.requires_grad_(False).to("cuda").eval()


def infer_scene(model, item, reference, args, cfg, folder):
    import torch
    from depth_anything_3.model.utils.gs_renderer import render_3dgs
    from depth_anything_3.utils.geometry import as_homogeneous
    from guidance.inpainting import (
        build_diffusion_pipeline1,
        run_diffusion_output1,
        build_diffusion_pipeline2,
        run_diffusion_output2,
        build_support_confidence,
    )
    from utils.io import write_json

    ref, src, masks = (
        reference[None].cuda(),
        item["src_imgs"][None].cuda(),
        item["src_masks"][None].cuda(),
    )
    masked = src * (1 - masks) + 0.5 * masks
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
    norm = lambda value: (value - mean) / std
    supports, support_masks, confidences, anchors, history = [], [], [], [], []
    pipeline = None
    h, w = args.height, args.width
    strategy = (
        {}
        if args.ref_view_strategy == "default"
        else {"ref_view_strategy": args.ref_view_strategy}
    )
    for iteration in range(args.num_refinement_iters):
        print(
            f"[FreeInpaint] pass {iteration + 1}/{args.num_refinement_iters}, supports={len(supports)}",
            flush=True,
        )
        prediction = model(
            ref_img=norm(ref),
            masked_src_imgs=norm(masked),
            src_masks=masks,
            support_imgs=norm(torch.stack(supports, dim=1)) if supports else None,
            support_masks=torch.stack(support_masks, dim=1) if supports else None,
            support_confidences=torch.stack(confidences, dim=1) if supports else None,
            **strategy,
        )
        record = {"forward_pass": iteration + 1, "num_support_views": len(supports)}
        history.append(record)
        if iteration == args.num_refinement_iters - 1:
            record["stop"] = "max_forward_passes"
            break
        scores, depths, alphas, renders = [], [], [], []
        for view in range(src.shape[1]):
            K = prediction["intrinsics"][0, view + 1].clone()
            K[0] /= w
            K[1] /= h
            rgb, depth, alpha = render_3dgs(
                extrinsics=as_homogeneous(
                    prediction["extrinsics"][0, view + 1][None].float()
                ),
                intrinsics=K[None],
                image_shape=(h, w),
                gaussian=prediction["gaussians"],
                use_sh=True,
                color_mode="RGB+ED",
            )
            area = masks[0, view, 0].sum()
            score = (
                (alpha.squeeze() * masks[0, view, 0]).sum() / area
                if area > 10
                else torch.tensor(999.0)
            )
            if not torch.isfinite(score):
                raise RuntimeError("Non-finite opacity score")
            scores.append(score.item())
            depths.append(depth.reshape(1, 1, h, w))
            alphas.append(alpha.reshape(1, 1, h, w))
            if args.support_canvas != "source":
                renders.append(rgb.clamp(0, 1))
        anchor = min(range(len(scores)), key=scores.__getitem__)
        record["opacity_scores"] = scores
        if scores[anchor] > cfg["refinement"]["opacity_threshold"]:
            record["stop"] = "opacity_threshold"
            break
        removal = args.task == "object-removal"
        if pipeline is None:
            pipeline = (
                build_diffusion_pipeline1 if removal else build_diffusion_pipeline2
            )(cfg, "cuda")
        seed_everything(args.seed + 1000 + iteration)
        depth = depths[anchor]
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        image, mask = src[:, anchor], masks[:, anchor]
        if args.support_canvas == "rendered":
            image = renders[anchor]
        elif args.support_canvas == "composite":
            image = image * (1 - mask) + renders[anchor] * mask
        generated = (run_diffusion_output1 if removal else run_diffusion_output2)(
            pipeline,
            image,
            mask,
            depth,
            ref,
            args.positive_prompt,
            args.negative_prompt,
            f"{args.dataset}_{item['scene_id'].replace('/', '_')}_{iteration}",
        )
        confidence = build_support_confidence(mask, rendered_alpha=alphas[anchor])
        if anchor in anchors:
            idx = anchors.index(anchor)
            supports[idx], support_masks[idx], confidences[idx] = (
                generated,
                mask,
                confidence,
            )
        else:
            anchors.append(anchor)
            supports.append(generated)
            support_masks.append(mask)
            confidences.append(confidence)
        record.update(anchor_source_index=anchor, seed=args.seed + 1000 + iteration)
        save_rgb(folder / f"support_{iteration + 1}.png", generated[0])
    write_json(folder / "refinement.json", history)
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    return prediction


def render_outputs(prediction, item, reference, args, folder):
    import numpy as np
    import torch
    from depth_anything_3.model.utils.gs_renderer import render_3dgs
    from depth_anything_3.utils.geometry import as_homogeneous

    count = len(item["src_imgs"]) + 1
    c2ws = torch.linalg.inv(as_homogeneous(prediction["extrinsics"].float()))[0, :count]
    h, w = args.height, args.width
    if args.dataset in {"spinnerf", "usid"}:
        # Cameras are used only AFTER unposed inference to render held-out benchmark views.
        gt_poses = torch.cat([item["ref_c2w"][None], item["src_c2ws"]]).cuda()
        d_gt = torch.linalg.vector_norm(gt_poses[-1, :3, 3] - gt_poses[0, :3, 3])
        d_pred = torch.linalg.vector_norm(c2ws[-1, :3, 3] - c2ws[0, :3, 3])
        scale = torch.where(d_gt < 1e-3, torch.ones_like(d_gt), d_pred / (d_gt + 1e-6))
        if not torch.isfinite(scale) or scale <= 0:
            raise RuntimeError(f"Invalid camera-alignment scale: {scale}")
        target_poses = []
        for pose in item["tgt_c2ws"]:
            relative = torch.linalg.inv(gt_poses[0]) @ pose.cuda()
            relative[:3, 3] *= scale
            target_poses.append(c2ws[0] @ relative)
        target_poses = torch.stack(target_poses)
        Ks, gts, masks = item["tgt_Ks"].cuda(), item["tgt_imgs"], item["tgt_masks"]
        names = item["target_names"]
    else:
        target_poses, Ks = c2ws, prediction["intrinsics"][0, :count]
        gts = torch.cat([item["ref_img"][None], item["src_imgs"]])
        masks, names = item["original_input_masks"], item["input_names"]
    for idx, (pose, intrinsic, gt, mask) in enumerate(
        zip(target_poses, Ks, gts, masks)
    ):
        K = intrinsic.clone()
        K[0] /= w
        K[1] /= h
        rgb, depth, _ = render_3dgs(
            extrinsics=torch.linalg.inv(pose)[None],
            intrinsics=K[None],
            image_shape=(h, w),
            gaussian=prediction["gaussians"],
            use_sh=True,
            color_mode="RGB+ED",
        )
        if not torch.isfinite(rgb).all():
            raise RuntimeError(f"Non-finite render: {idx}")
        save_rgb(folder / f"{idx}.png", rgb[0])
        save_rgb(folder / f"{idx}_gt.png", gt)
        save_mask(folder / f"{idx}_gt_mask.png", mask)
        np.save(
            folder / f"{idx}_depth.npy", depth.detach().float().cpu().numpy().squeeze()
        )
    save_rgb(folder / "reference.png", reference)
    np.savez_compressed(
        folder / "predicted_cameras.npz",
        c2ws=c2ws.cpu().numpy(),
        intrinsics=prediction["intrinsics"][0, :count].cpu().numpy(),
    )
    if args.save_gs:
        from depth_anything_3.utils.gsply_helpers import save_gaussian_ply

        save_gaussian_ply(
            prediction["gaussians"],
            str(folder / "gaussians.ply"),
            ctx_depth=prediction["depth"][0].unsqueeze(-1),
            shift_and_scale=False,
            save_sh_dc_only=False,
            inv_opacity=True,
        )
    if args.render_video:
        import imageio.v2 as imageio
        from utils.trajectory import (
            generate_orbit_trajectory,
            generate_spiral_trajectory,
        )

        gs = prediction["gaussians"]
        valid = gs.opacities[0].reshape(-1) > 0.1
        points = gs.means[0][valid].detach().cpu().numpy()
        if len(points) == 0:
            raise RuntimeError("No visible Gaussians for video trajectory")
        options = dict(
            c2ws_opencv=c2ws.cpu().numpy(),
            pts_3d=points,
            n_frames=args.n_frames,
            radius_mult=args.radius,
        )
        trajectory = (
            generate_orbit_trajectory(**options, elevation_deg=args.elevation)
            if args.trajectory == "orbit"
            else generate_spiral_trajectory(**options)
        )
        K = prediction["intrinsics"][0, 0].clone()
        K[0] /= w
        K[1] /= h
        with imageio.get_writer(
            folder / "video.mp4", fps=30, macro_block_size=2
        ) as writer:
            for pose in trajectory:
                pose = torch.as_tensor(pose, device="cuda", dtype=torch.float32)
                rgb, _, _ = render_3dgs(
                    extrinsics=torch.linalg.inv(pose)[None],
                    intrinsics=K[None],
                    image_shape=(h, w),
                    gaussian=gs,
                    use_sh=True,
                    color_mode="RGB+ED",
                )
                array = rgb[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                writer.append_data((array * 255).round().astype(np.uint8))
    return list(names)


def main():
    parser = build_parser()
    args = parser.parse_args()
    import yaml

    args.config = str(Path(args.config).expanduser().resolve())
    # Explicit CLI paths resolve relative to the caller; config paths relative to the repository.
    for key in (
        "checkpoint",
        "data_root",
        "output_root",
        "reference_image",
        "context_json",
    ):
        if getattr(args, key):
            setattr(args, key, str(Path(getattr(args, key)).expanduser().resolve()))
    cfg = yaml.safe_load(Path(args.config).read_text())
    os.chdir(ROOT)
    section = cfg[f"{args.dataset}_dataset"]
    args.data_root = args.data_root or section["root"]
    args.output_root = args.output_root or str(
        Path(cfg["experiment"]["output_dir"]) / args.dataset
    )
    args.checkpoint = args.checkpoint or cfg["model"]["checkpoint"]
    args.height, args.width = args.height or section["H"], args.width or section["W"]
    args.num_input_views = args.num_input_views or section.get(
        "num_input_views", 4 if args.dataset == "spinnerf" else 8
    )
    args.seed = cfg["experiment"]["seed"] if args.seed is None else args.seed
    args.ref_view_strategy = args.ref_view_strategy or section["ref_view_strategy"]
    if args.num_refinement_iters is None:
        args.num_refinement_iters = section.get(
            "num_forward_passes", cfg["refinement"]["num_forward_passes"]
        )
    if args.context_json:
        section["context_json"] = args.context_json
    if args.num_refinement_iters < 1 or args.num_input_views < 2:
        parser.error("Need at least one forward pass and two input views")
    if args.dataset in {"spinnerf", "usid"} and args.num_input_views != (
        4 if args.dataset == "spinnerf" else 8
    ):
        parser.error(
            "This benchmark release uses 4 SPIn-NeRF / 8 USID views; edit the loader to change the protocol"
        )
    if args.dataset == "co3d" and not args.category:
        parser.error("CO3D requires --category")
    if args.dataset != "co3d" and not args.scene_name:
        parser.error("--scene_name is required")
    if args.check_config:
        print(yaml.safe_dump(vars(args), sort_keys=False))
        return
    import torch
    from utils.io import write_json

    item = load_scene(args, cfg)
    print(f"[Inputs] {item['input_names']}; resolution={args.height}x{args.width}")
    if args.check_data:
        print(
            f"[Data OK] {item['scene_id']}; clean_gt_available={item['clean_gt_available']}"
        )
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is required for Gaussian rendering")
    if not Path(args.checkpoint).is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    folder = Path(args.output_root) / item["scene_id"] / args.task
    folder.mkdir(parents=True, exist_ok=True)
    Path("debug").mkdir(exist_ok=True)
    metadata = dict(
        status="running",
        dataset=args.dataset,
        scene=item["scene_id"],
        clean_gt_available=item["clean_gt_available"],
    )
    if not item["clean_gt_available"]:
        metadata["gt_file_semantics"] = (
            "Object-present input image; not a clean object-removal target. Do not compute reference-based removal metrics."
        )
    write_json(folder / "result.json", metadata)
    try:
        seed_everything(args.seed)
        prepare_masks(item, args, cfg)
        reference, _ = prepare_reference(item, args, cfg)
        model = load_model(args, cfg)
        seed_everything(args.seed)
        with torch.no_grad():
            prediction = infer_scene(model, item, reference, args, cfg, folder)
            render_outputs(prediction, item, reference, args, folder)
        metadata["status"] = "complete"
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(folder / "result.json", metadata)
    print(f"[Saved] {folder}")


if __name__ == "__main__":
    main()
