from PIL import Image
import numpy as np
import torchvision.transforms as transforms


import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from CGIC.zextention.loss import LossFns


from CGIC.models.model import CGIC


class NumpyToPIL:
    def __init__(self):
        pass
    def __call__(self, img: np.ndarray) -> Image.Image:
        return Image.fromarray(img)

class ResizeToLongSide:
    """Aspect-ratio preserving resize to a fixed long side.
    Always scales so that max(width, height) == long_side (upscales or downscales)."""

    def __init__(self, short_side=1024,
        interpolation=transforms.InterpolationMode.BICUBIC):
        self.short_side = short_side
        self.interpolation = interpolation

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        short_dim = min(w, h)
        scale = float(self.short_side) / float(short_dim)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resample = Image.BICUBIC if self.interpolation == transforms.InterpolationMode.BICUBIC else Image.BILINEAR
        return img.resize((new_w, new_h), resample=resample)



def _load_config(config_path: Union[str, Path]) -> OmegaConf:
    """Load OmegaConf config file."""
    return OmegaConf.load(str(config_path))


def _build_model(config: OmegaConf, device: Optional[str] = None) -> CGIC:
    """Instantiate CGIC model from config and move to device. Auto-select device.

    Prefers CUDA, then MPS (Apple Silicon), then CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model: CGIC = CGIC(**config.model.params)
    model = model.to(device).eval()
    return model


def _ensure_multiple_of_16(img: Image.Image) -> Image.Image:
    """Center-crop image so both sides are multiples of 16 (required by encoder strides)."""
    w, h = img.size
    hn = h // 16
    wn = w // 16
    return TF.center_crop(img, output_size=[int(16 * hn), int(16 * wn)])


def _to_tensor(img: Image.Image) -> torch.Tensor:
    return T.ToTensor()(img)


@torch.no_grad()
def extract_latents(
    image: Union[str, Path, Image.Image, torch.Tensor],
    *,
    config_path: Union[str, Path] = "./configs/config_inference.yaml",
    device: Optional[str] = None,
    return_numpy: bool = False,
) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor], int]]:
    """Extract quantized latents and codebook indices from an input image.

    Args:
        image: Path to an image, a PIL.Image, or a 3xHxW FloatTensor in [0,1].
        config_path: Path to config that defines the model and checkpoint.
        device: Device to run on; requires CUDA. Defaults to 'cuda' if available.
        return_numpy: If True, returns numpy arrays for 'latents' and 'indices'.

    Returns:
        A dict with keys:
          - 'latents': Quantized latent tensor [1, C, H16, W16]
          - 'indices': Codebook indices grid [1, H16, W16] (int64)
          - 'grain_mask': Tuple of three masks at different grains
          - 'compression_mode': Routing mode (int)
    """
    config = _load_config(config_path)
    model = _build_model(config, device=device)

    # Load and preprocess image
    if isinstance(image, (str, Path)):
        img = Image.open(str(image))
    elif isinstance(image, Image.Image):
        img = image
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:
            tensor = image.unsqueeze(0)
        elif image.dim() == 4 and image.shape[1] == 3:
            tensor = image
        else:
            raise ValueError("Tensor image must be [3,H,W] or [B,3,H,W].")
        pass_tensor = True
    else:
        raise TypeError("image must be a path, PIL.Image, or torch.Tensor")

    if 'img' in locals():
        img = _ensure_multiple_of_16(img)
        tensor = _to_tensor(img).unsqueeze(0)  # [1,3,H,W]

    # Move to device
    tensor = tensor.to(model.device)

    # Encode to obtain quantized latents and indices
    quant, _, _, grain_mask, indices_flat, _, compression_mode = model.encode(tensor)
    indices = indices_flat.view(-1, quant.shape[-2], quant.shape[-1])  # [1,H16,W16]

    if return_numpy:
        result = {
            "latents": quant.detach().cpu().numpy(),
            "indices": indices.long().detach().cpu().numpy(),
            "grain_mask": tuple(m.detach().cpu().numpy() for m in grain_mask),
            "compression_mode": int(compression_mode),
        }
    else:
        result = {
            "latents": quant,
            "indices": indices.long(),
            "grain_mask": tuple(grain_mask),
            "compression_mode": int(compression_mode),
        }

    return result

@torch.no_grad()
def reconstruct_image_from_latents(
    latents: Union[torch.Tensor, "np.ndarray"],
    *,
    config_path: Union[str, Path] = "./configs/config_inference.yaml",
    device: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """Decode an RGB image from quantized latents and optionally visualize/save it.

    Ignores codebook indices; uses a full fine-grain mask for reconstruction.

    Args:
        latents: Quantized latent tensor of shape [1,C,Hq,Wq] or numpy array.
        config_path: Path to the same config used for encoding.
        device: Optional device override ("cuda"/"mps"/"cpu").
        show: If True, displays the reconstructed image with matplotlib.
        save_path: If provided, saves the image to this path.

    Returns:
        PIL.Image.Image: The reconstructed RGB image in [0,255].
    """
    config = _load_config(config_path)
    model = _build_model(config, device=device)

    if isinstance(latents, np.ndarray):
        latents_t = torch.from_numpy(latents)
    else:
        latents_t = latents

    if latents_t.dim() == 3:
        latents_t = latents_t.unsqueeze(0)
    latents_t = latents_t.to(model.device, dtype=torch.float32)

    b, c, hq, wq = latents_t.shape
    # Masks need a channel dimension for broadcasting inside the decoder
    mask_coarse = torch.zeros((b, 1, hq // 4, wq // 4), dtype=torch.int, device=model.device)
    mask_medium = torch.zeros((b, 1, hq // 2, wq // 2), dtype=torch.int, device=model.device)
    mask_fine = torch.ones((b, 1, hq, wq), dtype=torch.int, device=model.device)

    recon = model.decode(latents_t, [mask_coarse, mask_medium, mask_fine])  # [B,3,H,W]
    return recon

def _resize_and_crop(pil_img, crop_sahpe: int = 512):
    # Resize and center-crop to crop_sahpe x crop_sahpe
    resize_op = ResizeToLongSide(short_side=crop_sahpe)
    pil_img = resize_op(pil_img)
    pil_img = TF.center_crop(pil_img, output_size=[crop_sahpe, crop_sahpe])
    return pil_img

def project_image_to_latents(
    image: Union[np.ndarray, Image.Image, torch.Tensor], 
    *, crop_sahpe: int = 512,
    config_path: Union[str, Path] = "./configs/config_inference.yaml",
    device: Optional[str] = None,
    return_numpy: bool = False,
) -> Union[torch.Tensor, "np.ndarray"]:
    """Project an input RGB image into quantized latents.

    Steps:
      1) Resize so the shorter side becomes 1024 while preserving aspect ratio
      2) Center-crop to 1024x1024
      3) Run the model encoder and return quantized latents

    Accepts numpy, PIL.Image, or torch.Tensor inputs. Works with small patches
    (they are upsampled so that min side is 1024 before center-cropping).
    """
    # Convert to PIL image first
    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected numpy image of shape [H,W,3] (RGB)")
        if image.dtype != np.uint8:
            img_u8 = np.clip(image, 0, 255).astype(np.uint8) if image.max() > 1.5 else np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = image
        pil_img = Image.fromarray(img_u8)
    elif isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, torch.Tensor):
        # Expect [H,W,3] or [3,H,W] in [0,1]
        if image.dim() == 3 and image.shape[-1] == 3:
            pil_img = T.ToPILImage()(image.permute(2, 0, 1))
        elif image.dim() == 3 and image.shape[0] == 3:
            pil_img = T.ToPILImage()(image)
        else:
            raise ValueError("Tensor must be [H,W,3] or [3,H,W] for image input")
    else:
        raise TypeError("image must be numpy.ndarray, PIL.Image, or torch.Tensor")

    pil_img = _resize_and_crop(pil_img, crop_sahpe)

    outputs = extract_latents(
        pil_img, config_path=config_path, device=device, return_numpy=return_numpy
    )
    return outputs["latents"]


def project_latents_to_image(
    latents: Union[torch.Tensor, "np.ndarray"],
    *,
    config_path: Union[str, Path] = "./configs/config_inference.yaml",
    device: Optional[str] = None,
) -> Image.Image:
    """Decode an RGB PIL image from quantized latents.

    Accepts latents as torch.Tensor [1,C,Hq,Wq] or numpy array with same shape.
    """

    recon = reconstruct_image_from_latents(
        latents, config_path=config_path, device=device, show=False, save_path=None
    )
    #recon = torch.clamp((recon + 1.0) / 2.0, 0.0, 1.0)
    recon = recon.squeeze(0).cpu()
    recon = recon - recon.min()
    recon = recon / (recon.max() - recon.min()) 
    image = T.ToPILImage()(recon)
    return image

