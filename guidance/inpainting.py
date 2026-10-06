"""Reference inpainting and diffusion support-view generation."""

import os
import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from safetensors.torch import load_model, load_file
from diffusers import (
    LCMScheduler,
    ControlNetModel,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
)
from guidance.powerpaint.pipelines.pipeline_PowerPaint import (
    StableDiffusionInpaintPipeline as Pipeline,
)
from guidance.powerpaint.utils.utils import TokenizerWrapper, add_tokens


def build_support_confidence(
    anchor_mask, rendered_alpha=None, blur_kernel=31, blindspot_gamma=1.5
):
    """
    Build a support confidence map that only activates strongly on true blind spots.

    The support region is still bounded by the anchor mask, but the confidence inside
    that region is modulated by the rendered opacity: regions already explained by the
    current 3D reconstruction should continue to trust the original observations,
    whereas low-opacity blind spots receive higher support confidence.
    """
    mask = anchor_mask.squeeze().detach().float().clamp(0, 1)
    if rendered_alpha is not None:
        alpha = rendered_alpha.squeeze().detach().float().clamp(0, 1).to(mask.device)
        blindspot = (1.0 - alpha).pow(blindspot_gamma) * mask
    else:
        blindspot = mask
    blindspot_np = blindspot.cpu().numpy().astype(np.float32)
    blur_kernel = max(3, blur_kernel)
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blurred = cv2.GaussianBlur(blindspot_np, (blur_kernel, blur_kernel), 0)
    if blurred.max() > 1e-06:
        blurred = blurred / blurred.max()
    confidence = np.clip(blurred * mask.cpu().numpy().astype(np.float32), 0.0, 1.0)
    return torch.from_numpy(confidence).to(anchor_mask.device).unsqueeze(0).unsqueeze(0)


def load_diffusion_output(
    ref_img,
    ref_mask,
    positive_prompt,
    negative_prompt,
    cfg,
    task_type,
    scene_name,
    device="cuda",
):
    os.makedirs("./cache/reference", exist_ok=True)
    os.makedirs("./debug", exist_ok=True)
    sd_pipe = Pipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-inpainting",
        dtype=torch.float16,
        use_safetensors=False,
    )
    sd_pipe.tokenizer = TokenizerWrapper(
        from_pretrained="stable-diffusion-v1-5/stable-diffusion-v1-5",
        subfolder="tokenizer",
        revision=None,
        local_files_only=False,
    )
    add_tokens(
        tokenizer=sd_pipe.tokenizer,
        text_encoder=sd_pipe.text_encoder,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    load_model(
        sd_pipe.unet,
        os.path.join(cfg["diffusion"]["checkpoint_dir"], "unet/unet.safetensors"),
    )
    state_dict = load_file(
        os.path.join(
            cfg["diffusion"]["checkpoint_dir"], "text_encoder/text_encoder.safetensors"
        )
    )
    (missing, unexpected) = sd_pipe.text_encoder.load_state_dict(
        state_dict, strict=False
    )
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    sd_pipe = sd_pipe.to(device)
    (promptA, promptB, negative_promptA, negative_promptB) = add_task(
        positive_prompt, negative_prompt, task_type, "ppt-v1"
    )
    print("Running diffusion inpainting...", ref_img.shape, ref_mask.shape)
    (H, W) = (ref_img.shape[2], ref_img.shape[3])
    ref_img_ = torch.nn.functional.interpolate(
        ref_img,
        size=((H - H % 8) * 2, (W - W % 8) * 2),
        mode="bilinear",
        align_corners=False,
    )
    ref_mask_ = torch.nn.functional.interpolate(
        ref_mask,
        size=((H - H % 8) * 2, (W - W % 8) * 2),
        mode="bilinear",
        align_corners=False,
    )
    inpainted_ref = sd_pipe(
        promptA=promptA,
        promptB=promptB,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        tradoff=0.3,
        tradoff_nag=0.3,
        image=ref_img_ * 2 - 1,
        mask=ref_mask_,
        width=(W - W % 8) * 2,
        height=(H - H % 8) * 2,
        guidance_scale=7.5,
        num_inference_steps=20,
        output_type="pt",
    )["images"]
    inpainted_ref = torch.nn.functional.interpolate(
        inpainted_ref, size=(H, W), mode="bilinear", align_corners=False
    )
    inpainted_ref = inpainted_ref * ref_mask + ref_img * (1 - ref_mask)
    vis_ref_img = ref_img.cpu().permute(0, 2, 3, 1).numpy()[0]
    vis_inpainted = inpainted_ref.float().cpu().permute(0, 2, 3, 1).numpy()[0]
    imageio.imwrite(
        f"./cache/reference/{scene_name}.jpg", (vis_inpainted * 255).astype(np.uint8)
    )
    vis_ref_mask = ref_mask.cpu().permute(0, 2, 3, 1).numpy()[0] * vis_ref_img
    vis_cat = np.concatenate([vis_ref_img, vis_ref_mask, vis_inpainted], axis=1)
    imageio.imwrite(
        "./debug/debug_diffusion_inpaint.png", (vis_cat * 255).astype(np.uint8)
    )
    print("Diffusion output obtained.", inpainted_ref.shape)
    ref_img = inpainted_ref * ref_mask + ref_img * (1 - ref_mask)
    print(ref_img.shape, ref_img.min(), ref_img.max())
    return inpainted_ref


def add_task(prompt, negative_prompt, control_type, version):
    pos_prefix = neg_prefix = ""
    if control_type == "object-removal" or control_type == "image-outpainting":
        if version == "ppt-v1":
            pos_prefix = "empty scene blur " + prompt
            neg_prefix = negative_prompt
        promptA = pos_prefix + " P_ctxt"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + " P_obj"
        negative_promptB = neg_prefix + " P_obj"
    elif control_type == "shape-guided":
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = (
                negative_prompt
                + ", worst quality, low quality, normal quality, bad quality, blurry "
            )
        promptA = pos_prefix + " P_shape"
        promptB = pos_prefix + " P_ctxt"
        negative_promptA = neg_prefix + "P_shape"
        negative_promptB = neg_prefix + "P_ctxt"
    else:
        if version == "ppt-v1":
            pos_prefix = prompt
            neg_prefix = (
                negative_prompt
                + ", worst quality, low quality, normal quality, bad quality, blurry "
            )
        promptA = pos_prefix + " P_obj"
        promptB = pos_prefix + " P_obj"
        negative_promptA = neg_prefix + "P_obj"
        negative_promptB = neg_prefix + "P_obj"
    return (promptA, promptB, negative_promptA, negative_promptB)


def build_diffusion_pipeline2(cfg, device):
    print("[Info] Loading Ultimate SDXL Pipeline (Depth + IP-Adapter + LCM)...")
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16, variant="fp16"
    )
    sd_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        requires_safety_checker=False,
    )
    print(" -> Fusing SDXL LCM LoRA...")
    sd_pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    sd_pipe.fuse_lora()
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    print(" -> Loading SDXL IP-Adapter...")
    sd_pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin"
    )
    sd_pipe.set_ip_adapter_scale(0.8)
    sd_pipe = sd_pipe.to(device)
    print("✅ Ultimate Pipeline Ready!")
    return sd_pipe


def run_diffusion_output2(
    sd_pipe,
    img,
    mask,
    depth_map,
    ref_style_img,
    positive_prompt,
    negative_prompt,
    scene_id,
):
    """
    img: [1, 3, H, W] - DA3-rendered base image with edge structure.
    mask: [1, 1, H, W] - Blind-region mask (1 denotes a hole).
    depth_map: [1, 1, H, W] - Depth map for ControlNet.
    ref_style_img: [1, 3, H, W] - Reference image providing texture guidance.
    """
    (H, W) = (img.shape[2], img.shape[3])
    target_H = H - H % 8
    target_W = W - W % 8
    img_ = torch.nn.functional.interpolate(
        img, size=(target_H, target_W), mode="bilinear"
    )
    mask_ = torch.nn.functional.interpolate(
        mask, size=(target_H, target_W), mode="nearest"
    )
    depth_map_ = torch.nn.functional.interpolate(
        depth_map, size=(target_H, target_W), mode="bilinear"
    )
    ref_style_img_ = torch.nn.functional.interpolate(
        ref_style_img, size=(target_H, target_W), mode="bilinear"
    )
    depth_map_3c = depth_map_.repeat(1, 3, 1, 1)
    ref_style_pil = TF.to_pil_image(ref_style_img.squeeze(0).cpu().clamp(0, 1))
    prompt = "high quality, seamless background, " + positive_prompt
    neg_prompt = "blurry, artifacts, deformed geometry, " + negative_prompt
    inpainted = sd_pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=img_ * 2 - 1,
        mask_image=mask_,
        control_image=depth_map_3c,
        ip_adapter_image=ref_style_pil,
        width=target_W,
        height=target_H,
        num_inference_steps=4,
        guidance_scale=1.5,
        strength=0.85,
        controlnet_conditioning_scale=0.7,
        output_type="pt",
    ).images
    inpainted = torch.nn.functional.interpolate(inpainted, size=(H, W), mode="bilinear")
    final_img = inpainted * mask + img * (1 - mask)
    ref_style_img_vis = ref_style_img.float().cpu().permute(0, 2, 3, 1).numpy()[0]
    final_img_vis = final_img.float().cpu().permute(0, 2, 3, 1).numpy()[0]
    vis_cat = np.concatenate([ref_style_img_vis, final_img_vis], axis=1)
    imageio.imwrite(
        f"./debug/{scene_id}_{positive_prompt}.jpg", (vis_cat * 255).astype(np.uint8)
    )
    return final_img


def build_diffusion_pipeline1(cfg, device):
    print("[Info] Loading Ultra-Fast SD1.5 Pipeline (LCM + IP-Adapter)...")
    sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        variant="fp16",
        requires_safety_checker=False,
        safety_checker=None,
    )
    print(" -> Fusing LCM LoRA...")
    sd_pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    sd_pipe.fuse_lora()
    sd_pipe.scheduler = LCMScheduler.from_config(sd_pipe.scheduler.config)
    sd_pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
    )
    sd_pipe.set_ip_adapter_scale(0.8)
    sd_pipe = sd_pipe.to(device)
    print("✅ Ultra-Fast Pipeline Ready!")
    return sd_pipe


def run_diffusion_output1(
    sd_pipe,
    img,
    mask,
    depth_map,
    ref_style_img,
    positive_prompt,
    negative_prompt,
    scene_id,
):
    """
    img: [1, 3, H, W] - DA3-rendered base image with edge structure.
    mask: [1, 1, H, W] - Blind-region mask (1 denotes a hole).
    depth_map: [1, 1, H, W] - Depth map for ControlNet.
    ref_style_img: [1, 3, H, W] - Reference image providing texture guidance.
    """
    (H, W) = (img.shape[2], img.shape[3])
    target_H = H - H % 8
    target_W = W - W % 8
    img_ = torch.nn.functional.interpolate(
        img, size=(target_H, target_W), mode="bilinear"
    )
    mask_ = torch.nn.functional.interpolate(
        mask, size=(target_H, target_W), mode="nearest"
    )
    depth_map_ = torch.nn.functional.interpolate(
        depth_map, size=(target_H, target_W), mode="bilinear"
    )
    ref_style_img_ = torch.nn.functional.interpolate(
        ref_style_img, size=(target_H, target_W), mode="bilinear"
    )
    depth_map_3c = depth_map_.repeat(1, 3, 1, 1)
    ref_style_pil = TF.to_pil_image(ref_style_img.squeeze(0).cpu().clamp(0, 1))
    prompt = "high quality, seamless background, " + positive_prompt
    neg_prompt = "blurry, artifacts, deformed geometry, " + negative_prompt
    inpainted = sd_pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=img_ * 2 - 1,
        mask_image=mask_,
        width=target_W,
        ip_adapter_image=ref_style_pil,
        height=target_H,
        num_inference_steps=4,
        guidance_scale=1.5,
        strength=0.85,
        output_type="pt",
    ).images
    inpainted = torch.nn.functional.interpolate(inpainted, size=(H, W), mode="bilinear")
    final_img = inpainted * mask + img * (1 - mask)
    ref_style_img_vis = ref_style_img.float().cpu().permute(0, 2, 3, 1).numpy()[0]
    final_img_vis = final_img.float().cpu().permute(0, 2, 3, 1).numpy()[0]
    vis_cat = np.concatenate([ref_style_img_vis, final_img_vis], axis=1)
    imageio.imwrite(
        f"./debug/{scene_id}_{positive_prompt}.jpg", (vis_cat * 255).astype(np.uint8)
    )
    return final_img
