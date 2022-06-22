import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFilter2D(nn.Module):
    def __init__(self, window_size=11, in_channels=1, sigma=1.5, padding=None, ensemble_kernel=True):
        """2D Gaussian Filer

        Args:
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
        """
        super().__init__()
        self.window_size = window_size
        if not (window_size % 2 == 1):
            raise ValueError("Window size must be odd.")
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma
        self.ensemble_kernel = ensemble_kernel

        kernel = self._get_gaussian_window1d()
        if ensemble_kernel:
            kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1))

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x ** 2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d):
        w = torch.matmul(gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d)
        return w

    def forward(self, x):
        if self.ensemble_kernel:
            # ensemble kernel: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L11-L15
            x = F.conv2d(input=x, weight=self.gaussian_window, stride=1, padding=self.padding, groups=x.shape[1])
        else:
            # splitted kernel: https://github.com/VainF/pytorch-msssim/blob/2398f4db0abf44bcd3301cfadc1bf6c94788d416/pytorch_msssim/ssim.py#L48
            for i, d in enumerate(x.shape[2:], start=2):
                if d >= self.window_size:
                    w = self.gaussian_window.transpose(dim0=-1, dim1=i)
                    x = F.conv2d(input=x, weight=w, stride=1, padding=self.padding, groups=x.shape[1])
                else:
                    warnings.warn(
                        f"Skipping Gaussian Smoothing at dimension {i} for x: {x.shape} and window size: {self.window_size}"
                    )
        return x


class SSIM(nn.Module):
    def __init__(
        self,
        window_size=11,
        in_channels=1,
        sigma=1.5,
        *,
        K1=0.01,
        K2=0.03,
        L=1,
        keep_batch_dim=False,
        return_log=False,
        return_msssim=False,
        padding=None,
        ensemble_kernel=True,
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
            return_msssim (bool, optional): Whether to return the MS-SSIM score. Defaults to False, which will return the original MSSIM score.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.

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
            # setting 5: for 4d float tensors with the data range [0, 1] and 1 channel, padding=0 and the splitted kernels.
            ssim_caller = SSIM(return_log=True, keep_batch_dim=True, padding=0, ensemble_kernel=False).cuda()

            # two 4d tensors
            x = torch.randn(3, 1, 100, 100).cuda()
            y = torch.randn(3, 1, 100, 100).cuda()
            ssim_score_0 = ssim_caller(x, y)
            # or in the fp16 mode (we have fixed the computation progress into the float32 mode to avoid the unexpected result)
            with torch.cuda.amp.autocast(enabled=True):
                ssim_score_1 = ssim_caller(x, y)
            assert torch.isclose(ssim_score_0, ssim_score_1)
        ```

        Reference:
        [1] SSIM: Wang, Zhou et al. “Image quality assessment: from error visibility to structural similarity.” IEEE Transactions on Image Processing 13 (2004): 600-612.
        [2] MS-SSIM: Wang, Zhou et al. “Multi-scale structural similarity for image quality assessment.” (2003).
        """
        super().__init__()
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.keep_batch_dim = keep_batch_dim
        self.return_log = return_log
        self.return_msssim = return_msssim

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
        )

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y):
        """Calculate the mean SSIM (MSSIM) between two 4d tensors.

        Args:
            x (Tensor): 4d tensor
            y (Tensor): 4d tensor

        Returns:
            Tensor: MSSIM or MS-SSIM
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"
        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        if self.return_msssim:
            return self.msssim(x, y)
        else:
            return self.ssim(x, y)

    def ssim(self, x, y):
        ssim, _ = self._ssim(x, y)
        if self.return_log:
            # https://github.com/xuebinqin/BASNet/blob/56393818e239fed5a81d06d2a1abfe02af33e461/pytorch_ssim/__init__.py#L81-L83
            ssim = ssim - ssim.min()
            ssim = ssim / ssim.max()
            ssim = -torch.log(ssim + 1e-8)

        if self.keep_batch_dim:
            return ssim.mean(dim=(1, 2, 3))
        else:
            return ssim.mean()

    def msssim(self, x, y):
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)

            if self.keep_batch_dim:
                ssim = ssim.mean(dim=(1, 2, 3))
                cs = cs.mean(dim=(1, 2, 3))
            else:
                ssim = ssim.mean()
                cs = cs.mean()

            if i == 4:
                ms_components.append(ssim ** w)
            else:
                ms_components.append(cs ** w)
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
        msssim = math.prod(ms_components)  # equ 7 in ref2
        return msssim

    def _ssim(self, x, y):
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x * mu_x + mu_y * mu_y + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l = A1 / B1
        cs = A2 / B2
        ssim = l * cs
        return ssim, cs


def ssim(
    x,
    y,
    *,
    window_size=11,
    in_channels=1,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    keep_batch_dim=False,
    return_log=False,
    return_msssim=False,
    padding=None,
    ensemble_kernel=True,
):
    """Calculate the mean SSIM (MSSIM) between two 4D tensors.

    Args:
        x (Tensor): 4d tensor
        y (Tensor): 4d tensor
        window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
        in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
        sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
        K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
        K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
        L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
        keep_batch_dim (bool, optional): Whether to keep the batch dim. Defaults to False.
        return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.
        return_msssim (bool, optional): Whether to return the MS-SSIM score. Defaults to False, which will return the original MSSIM score.
        padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
        ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.

    Returns:
        Tensor: MSSIM or MS-SSIM
    """
    ssim_obj = SSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        K1=K1,
        K2=K2,
        L=L,
        keep_batch_dim=keep_batch_dim,
        return_log=return_log,
        return_msssim=return_msssim,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
    ).to(device=x.device)
    return ssim_obj(x, y)
