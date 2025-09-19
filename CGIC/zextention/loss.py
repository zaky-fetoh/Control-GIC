import torch
import torch.nn as nn
import torch.nn.functional as F

def _gaussian_window(window_size, sigma=1.5, dtype=torch.float32, device='cpu'):
    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size // 2)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g[:, None] @ g[None, :]  # outer product -> 2D Gaussian
    return window_2d

def _ssim_loss(x, y,
               window_size=7,
               sigma=1.5,
               K1=0.01,
               K2=0.03,
               L=1.0, # dynamic range: 1.0 for images scaled to [0,1], 255 for [0,255]
               eps=1e-6,
               use_reflect_pad=True,
               reduction='mean'):
    """
    Returns SSIM loss = 1 - mean(SSIM_map) by default.
    x, y : (B, C, H, W) float Tensors (same dtype & device)
    """

    assert x.shape == y.shape, "x and y must have the same shape"
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    # Constants scaled by dynamic range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Create Gaussian window with same dtype/device as inputs
    window = _gaussian_window(window_size, sigma, dtype=dtype, device=device)
    window = window.unsqueeze(0).unsqueeze(0)        # [1,1,ws,ws]
    window = window.repeat(C, 1, 1, 1)               # [C,1,ws,ws] -> depthwise conv weight

    pad = window_size // 2
    if use_reflect_pad:
        x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        y_pad = F.pad(y, (pad, pad, pad, pad), mode='reflect')
        padding = 0
    else:
        # let conv2d handle zero-padding (less preferred)
        x_pad, y_pad = x, y
        padding = pad

    # local means
    mu_x = F.conv2d(x_pad, window, padding=padding, groups=C)
    mu_y = F.conv2d(y_pad, window, padding=padding, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # local variances / covariance
    sigma_x_sq = F.conv2d(x_pad * x_pad, window, padding=padding, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y_pad * y_pad, window, padding=padding, groups=C) - mu_y_sq
    sigma_xy   = F.conv2d(x_pad * y_pad, window, padding=padding, groups=C) - mu_xy

    # Clamp variances to >= 0 to avoid tiny negative values from numerical error
    sigma_x_sq = torch.clamp(sigma_x_sq, min=0.0)
    sigma_y_sq = torch.clamp(sigma_y_sq, min=0.0)

    # SSIM formula
    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / (denominator + eps)

    if reduction == 'mean':
        return 1.0 - ssim_map.mean()
    elif reduction == 'none':
        return 1.0 - ssim_map  # shape (B,C,H,W)
    elif reduction == 'spatial_mean':
        # average over H,W but keep B,C if caller wants per-channel/per-image statistics
        return 1.0 - ssim_map.flatten(2).mean(-1).mean(-1)  # -> (B,C)
    else:
        raise ValueError("reduction must be 'mean'|'none'|'spatial_mean'")



class LossFns(nn.Module):
    def __init__(self, x, xhat, lx,
                mu: torch.Tensor,
                sig: torch.Tensor):
        super(LossFns, self).__init__()
        self.x, self.xhat, self.lx = x, xhat, lx
        self.mu, self.sig = mu, sig

    def mse_loss(self):
        return F.mse_loss(self.x, self.xhat)

    def l1_loss(self):
        return F.l1_loss(self.x, self.xhat)

    def ssim_loss(self):
        return _ssim_loss(self.x, self.xhat)

    def var_loss(self, gamma: float = 1.0):
        #penalize the less variance
        x = self.mu - self.mu.mean(dim=0)
        std = x.std(dim=0)
        var_loss = F.leaky_relu(gamma - std).mean()
        return var_loss
    
    def cov_loss(self):
        #penalize the less covariance
        x = self.mu - self.mu.mean(dim=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
        return cov_loss
    
    def vic_loss(self):
        pass

    def kl_loss(self) -> torch.Tensor:
        var = self.sig.pow(2)
        kl_per_dim = 0.5 * (var + self.mu.pow(2) - 1.0 - torch.log(var + 1e-8))

        # Sum across latent dims -> per-sample KL
        kl_per_sample = kl_per_dim.sum(dim=1)

        return kl_per_sample.mean()

    def total_loss(self, 
                mse_weight: float = 1.0,
                ssim_weight: float = 1.0,
                var_weight: float = 1.0,
                cov_weight: float = 1.0,
                kl_weight: float = 1.0
                ):
        
        mse_loss_val = self.mse_loss()
        ssim_loss_val = self.ssim_loss()
        var_loss_val = self.var_loss()
        cov_loss_val = self.cov_loss()
        kl_loss_val = self.kl_loss()
        
        print(f"MSE loss: {mse_loss_val:.4f}")
        print(f"SSIM loss: {ssim_loss_val:.4f}")
        print(f"Var loss: {var_loss_val:.4f}")
        print(f"Cov loss: {cov_loss_val:.4f}")
        print(f"KL loss: {kl_loss_val:.4f}")
        
        totaloss = (
            mse_loss_val * mse_weight 
            + ssim_loss_val * ssim_weight 
            + var_loss_val * var_weight 
            + cov_loss_val * cov_weight 
            + kl_loss_val * kl_weight
            )
        
        print(f"Total loss: {totaloss:.4f}")
        return totaloss
    
    
if __name__ == "__main__":
    x = torch.randn(5, 3, 32, 32)
    xhat = torch.randn(5, 3, 32, 32)
    lx = torch.randn(5, 128)
    mu = torch.randn(5, 128)
    sig = torch.randn(5, 128)
    loss_fn = LossFns(x, xhat, lx, mu, sig)
    print(loss_fn.total_loss())
