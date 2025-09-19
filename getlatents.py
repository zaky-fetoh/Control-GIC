import sys
sys.path.append(".")

from PIL import Image
import matplotlib.pyplot as plt
from CGIC.zextention.loss import LossFns
from CGIC.zextention.utiles import project_image_to_latents, project_latents_to_image, _resize_and_crop, _ensure_multiple_of_16
import torchvision.transforms as T
import torch

#aemodel waights on gd is main_ae/4203-850/model.ckpt please copy it to ckpt/model.ckpt

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--config", type=str, default="./configs/config_inference.yaml")
    parser.add_argument("--numpy", action="store_true")
    args = parser.parse_args()

    # Read image and convert to latents
    orig_img = Image.open(args.image)

    orig_img = _resize_and_crop(orig_img, crop_sahpe=512)

    latents = project_image_to_latents(
        orig_img,
        config_path=args.config,
        crop_sahpe=512,
        device=None,
        return_numpy=args.numpy
    )

    print(f"latents shape: {latents.shape}")

    reconimg = project_latents_to_image(latents) 

    # Load and display original image alongside reconstruction
    orig_img = _ensure_multiple_of_16(orig_img)  # Match preprocessing
    

    
    # Convert images to tensors and normalize to [-1,1] range for loss calculation
    orig_tensor = T.ToTensor()(orig_img) * 2.0 - 1.0
    recon_tensor = T.ToTensor()(reconimg) * 2.0 - 1.0
    
    # Add batch dimension
    orig_tensor = orig_tensor.unsqueeze(0)
    recon_tensor = recon_tensor.unsqueeze(0)
    
    # Create loss function instance with dummy values for latent variables 
    # since we only care about image reconstruction metrics
    loss_fn = LossFns(
        x=orig_tensor, xhat=recon_tensor,
        lx=torch.zeros(1,1),mu=torch.zeros(1,1), sig=torch.zeros(1,1), # Dummy values
    )
    
    # Calculate and print reconstruction metrics
    mse = loss_fn.mse_loss().item()
    ssim = loss_fn.ssim_loss().item()
    
    print(f"\nReconstruction metrics:")
    print(f"MSE loss: {mse:.4f}")
    print(f"SSIM loss: {ssim:.4f}")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(reconimg)
    plt.title("Reconstruction") 
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

