import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFilter2D(nn.Module):
    def __init__(self, window_size=11, in_channels=1, sigma=1.5) -> None:
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.in_channels = in_channels
        self.padding = window_size // 2
        self.sigma = sigma
        self.register_buffer(name="gaussian_window2d", tensor=self._get_gaussian_window2d())

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x ** 2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, self.window_size, 1)

    def _get_gaussian_window2d(self):
        gaussian_window_1d = self._get_gaussian_window1d()
        w = torch.matmul(gaussian_window_1d, gaussian_window_1d.transpose(dim0=-1, dim1=-2))
        w.reshape(1, 1, self.window_size, self.window_size)
        return w.repeat(self.in_channels, 1, 1, 1)

    def forward(self, x):
        x = F.conv2d(input=x, weight=self.gaussian_window2d, padding=self.padding, groups=x.shape[1])
        return x


class SSIM(nn.Module):
    def __init__(
        self, window_size=11, in_channels=1, sigma=1.5, K1=0.01, K2=0.03, L=1, keep_batch_dim=False, return_log=False
    ):
        """Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
            K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
            L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
            return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.

        ```
            # setting 0: for 4d float tensors with the data range [0, 1] and 1 channel
            ssim_caller = SSIM().cuda()
            # setting 1: for 4d float tensors with the data range [0, 1] and 3 channel
            ssim_caller = SSIM(in_channels=3).cuda()
            # setting 2: for 4d float tensors with the data range [0, 255] and 3 channel
            ssim_caller = SSIM(L=255, in_channels=3).cuda()
            # setting 3: for 4d float tensors with the data range [0, 255] and 3 channel, and return the logarithmic form
            ssim_caller = SSIM(L=255, in_channels=3, return_log=True).cuda()
            # setting 4: for 4d float tensors with the data range [0, 1] and 1 channel,return the logarithmic form, and keep the batch dim
            ssim_caller = SSIM(return_log=True, keep_batch_dim=True).cuda()

            # two 4d tensors
            x = torch.randn(3, 1, 100, 100).cuda()
            y = torch.randn(3, 1, 100, 100).cuda()
            ssim_score_0 = ssim_caller(x, y)
            # or in the fp16 mode (we have fixed the computation progress into the float32 mode to avoid the unexpected result)
            with torch.cuda.amp.autocast(enabled=True):
                ssim_score_1 = ssim_caller(x, y)
            assert torch.isclose(ssim_score_0, ssim_score_1)
        ```
        """
        super().__init__()
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.keep_batch_dim = keep_batch_dim
        self.return_log = return_log

        self.gaussian_filer = GaussianFilter2D(window_size=window_size, in_channels=in_channels, sigma=sigma)

    def forward(self, x, y):
        return ssim(
            x,
            y,
            gaussian_filter=self.gaussian_filer,
            C1=self.C1,
            C2=self.C2,
            keep_batch_dim=self.keep_batch_dim,
            return_log=self.return_log,
        )


@torch.cuda.amp.autocast(enabled=False)
def ssim(x, y, gaussian_filter, C1, C2, keep_batch_dim=False, return_log=False):
    """Calculate the mean SSIM (MSSIM) between two 4d tensors.

    Args:
        x (Tensor): 4d tensor
        y (Tensor): 4d tensor
        gaussian_filter (GaussianFilter2D): the gaussian filter object
        C1 (float): the constant to avoid instability
        C2 (float): the constant to avoid instability
        keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
        return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.

    Returns:
        Tensor: MSSIM
    """
    assert x.shape == y.shape, f"x: {x.shape} != y: {y.shape}"
    assert x.ndim == y.ndim == 4, f"x: {x.ndim} != y: {y.ndim} != 4"
    assert x.type() == y.type(), f"x: {x.type()} != y: {y.type()}"

    mu_x = gaussian_filter(x)  # equ 14
    mu_y = gaussian_filter(y)  # equ 14
    sigma2_x = gaussian_filter(x * x) - mu_x * mu_x  # equ 15
    sigma2_y = gaussian_filter(y * y) - mu_y * mu_y  # equ 15
    sigma_xy = gaussian_filter(x * y) - mu_x * mu_y  # equ 16

    # equ 13 in ref1
    A1 = 2 * mu_x * mu_y + C1
    A2 = 2 * sigma_xy + C2
    B1 = mu_x * mu_x + mu_y * mu_y + C1
    B2 = sigma2_x + sigma2_y + C2
    S = (A1 * A2) / (B1 * B2)

    if return_log:
        S = S - S.min()
        S = S / S.max()
        S = -torch.log(S + 1e-8)

    if keep_batch_dim:
        return S.mean(dim=(1, 2, 3))
    else:
        return S.mean()
