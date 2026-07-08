import math
import warnings
from enum import Enum
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

FILTER = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}


class Reduction(str, Enum):
    MEAN = "mean"
    SUM = "sum"
    BATCH = "batch"
    NONE = "none"

    @classmethod
    def values(cls):
        return {None, *(reduction.value for reduction in cls)}

    @classmethod
    def coerce(cls, reduction):
        if reduction is None:
            return None
        try:
            return cls(reduction)
        except ValueError as exc:
            raise ValueError(f"reduction must be one of {cls.values()}, but got {reduction}.") from exc

    @classmethod
    def resolve(cls, reduction):
        reduction = cls.coerce(reduction)
        if reduction is None:
            return cls.MEAN
        return reduction


def _check_weights(weights, *, name="weights", eps=1e-12):
    if not isinstance(weights, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, but got {type(weights).__name__}.")
    if weights.ndim == 0:
        raise ValueError(f"{name} must not be a scalar tensor.")
    if weights.numel() == 0:
        raise ValueError(f"{name} must not be empty.")
    if not torch.isfinite(weights).all():
        raise ValueError(f"{name} must contain only finite values.")
    if torch.any(weights < 0):
        raise ValueError(f"{name} must be nonnegative.")
    if weights.sum() <= eps:
        raise ValueError(f"{name} must have a positive sum.")


def _reduce_similarity(value, reduction=None, stable_mean=False, weights=None, eps=1e-12):
    mode = Reduction.resolve(reduction)
    if mode is Reduction.NONE:
        return value
    if value.ndim == 0:
        if weights is not None:
            warnings.warn("weights do not affect scalar similarity scores.", UserWarning, stacklevel=2)
        return value

    if weights is None:
        if mode is Reduction.BATCH:
            if value.ndim == 1:
                return value
            return value.flatten(1).mean(-1)
        if mode is Reduction.SUM:
            return value.sum()
        if stable_mean:
            return value.double().mean().to(dtype=value.dtype)
        return value.mean()

    if weights.device != value.device:
        raise ValueError(f"weights and SSIM map must be on the same device, got {weights.device} and {value.device}.")
    weights = weights.to(dtype=value.dtype)
    _check_weights(weights, eps=eps)
    weighted = value * weights
    weight_map = torch.ones_like(value) * weights

    if mode is Reduction.BATCH:
        dims = tuple(range(1, value.ndim))
        weight_sum = weight_map.sum(dim=dims)
        if torch.any(weight_sum <= eps):
            raise ValueError("weights must have a positive sum for every batch item.")
        return weighted.sum(dim=dims) / weight_sum
    if mode is Reduction.SUM:
        weight_sum = weight_map.sum()
        if weight_sum <= eps:
            raise ValueError("weights must have a positive sum.")
        return weighted.sum()

    weight_sum = weight_map.sum()
    if weight_sum <= eps:
        raise ValueError("weights must have a positive sum.")
    return weighted.sum() / weight_sum


def _ssim_map(x, y, local_ops, *, C1, C2):
    mu_x = local_ops(x)  # equ 14
    mu_y = local_ops(y)  # equ 14
    sigma2_x = local_ops(x * x) - mu_x * mu_x  # equ 15
    sigma2_y = local_ops(y * y) - mu_y * mu_y  # equ 15
    sigma_xy = local_ops(x * y) - mu_x * mu_y  # equ 16

    A1 = 2 * mu_x * mu_y + C1
    A2 = 2 * sigma_xy + C2
    B1 = mu_x * mu_x + mu_y * mu_y + C1
    B2 = sigma2_x + sigma2_y + C2

    # equ 12, 13 in ref1
    l = A1 / B1
    cs = A2 / B2
    ssim = l * cs
    return ssim, cs


class IdentityPreprocess(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return x, y


class SSIMMapReducer(nn.Module):
    def __init__(self, reduction=None, stable_mean=False, eps=1e-12):
        super().__init__()
        self.reduction = Reduction.coerce(reduction)
        self.stable_mean = stable_mean
        self.eps = eps

    def forward(self, score_map, weights=None, ref=None):
        return _reduce_similarity(
            score_map, reduction=self.reduction, stable_mean=self.stable_mean, weights=weights, eps=self.eps
        )


class GaussianFilter(nn.Module):
    def __init__(self, data_dim, window_size, in_channels, sigma, padding=None, ensemble_kernel=True):
        """Gaussian Filer for 1D, 2D or 3D data (3D/4D/5D tensor)

        Args:
            data_dim (int, optional): The dimension of the data.
            window_size (int or Tuple[int], optional): The window size of the gaussian filter.
            in_channels (int, optional): The number of channels of the 4d tensor.
            sigma (float or Tuple[float], optional): The sigma of the gaussian filter.
            padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
        """
        super().__init__()
        if data_dim not in [1, 2, 3]:
            raise ValueError(f"data_dim must be 1, 2 or 3, but got {data_dim}.")
        self.data_dim = data_dim
        self.filter = FILTER[self.data_dim]
        self.in_channels = in_channels

        if isinstance(window_size, int):
            window_size = [window_size] * self.data_dim
        if not all([w % 2 == 1 for w in window_size]):
            raise ValueError(f"Window size must be odd, but got {window_size}.")
        self.window_size = window_size

        if padding is None:
            padding = [w // 2 for w in window_size]
        if isinstance(padding, int):
            padding = [padding] * self.data_dim
        self.padding = padding

        if isinstance(sigma, (float, int)):
            sigma = [sigma] * self.data_dim
        self.sigma2 = [s**2 for s in sigma]

        assert len(self.window_size) == len(self.padding) == len(self.sigma2) == self.data_dim
        kernels = [self._get_gaussian_window1d(w, s2) for w, s2 in zip(self.window_size, self.sigma2)]

        self.ensemble_kernel = ensemble_kernel
        self._set_gaussian_windows(kernels)

    def _set_gaussian_windows(self, kernels):
        if self.ensemble_kernel:
            kernels = self._get_gaussian_windowNd(kernels)
            kernels = kernels.reshape(1, 1, *self.window_size).repeat_interleave(repeats=self.in_channels, dim=0)
            self.register_buffer(name="gaussian_window", tensor=kernels)
        else:
            for dim_idx, kernel in enumerate(kernels, start=2):
                base_shape = [1, 1] + [1] * self.data_dim
                base_shape[dim_idx] = -1
                kernel = kernel.reshape(*base_shape).repeat_interleave(repeats=self.in_channels, dim=0)
                if dim_idx == 2:
                    name = "gaussian_window"
                else:
                    name = f"gaussian_window_{dim_idx}"
                self.register_buffer(name=name, tensor=kernel)

    def _apply(self, fn):
        super()._apply(fn)
        device = self.gaussian_window.device
        kernels = [
            self._get_gaussian_window1d(w, s2).to(device=device, dtype=torch.float32)
            for w, s2 in zip(self.window_size, self.sigma2)
        ]
        self._set_gaussian_windows(kernels)
        return self

    @staticmethod
    def _get_gaussian_window1d(window_size, sigma2):
        x = torch.arange(-(window_size // 2), window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w

    def _get_gaussian_windowNd(self, gaussian_windows_1d):
        for dim_idx, kernel in enumerate(gaussian_windows_1d, start=2):
            base_shape = [1, 1] + [1] * self.data_dim
            base_shape[dim_idx] = -1
            kernel = kernel.reshape(*base_shape)
            if dim_idx == 2:
                w = kernel
            else:
                w = w * kernel
        return w

    def __repr__(self):
        base_str = f"{self.__class__.__name__} with Kernel: {self.gaussian_window.shape}"
        if not self.ensemble_kernel:
            for dim_idx in range(3, self.data_dim + 2):
                kernel = self.get_buffer(f"gaussian_window_{dim_idx}")
                base_str += f", {kernel.shape}"
        return base_str

    def forward(self, x):
        if self.ensemble_kernel:
            # ensemble kernel: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L11-L15
            weight = self.gaussian_window.to(dtype=x.dtype)
            x = self.filter(input=x, weight=weight, stride=1, padding=self.padding, groups=x.shape[1])
        else:
            # splitted kernel: https://github.com/VainF/pytorch-msssim/blob/2398f4db0abf44bcd3301cfadc1bf6c94788d416/pytorch_msssim/ssim.py#L48
            for i, d in enumerate(x.shape[2:], start=2):
                if d >= self.window_size[i - 2]:
                    w = self.get_buffer(target="gaussian_window" if i == 2 else f"gaussian_window_{i}").to(
                        dtype=x.dtype
                    )
                    x = self.filter(input=x, weight=w, stride=1, padding=self.padding, groups=x.shape[1])
                else:
                    warnings.warn(
                        f"Skipping Gaussian Smoothing at dimension {i} for x: {x.shape} and window size: {self.window_size}"
                    )
        return x


class SSIMCore(nn.Module):
    def __init__(self, filter, *, K1=0.01, K2=0.03, L=1):
        super().__init__()
        self.filter = filter
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1

    def forward(self, x, y):
        return _ssim_map(x, y, self.filter, C1=self.C1, C2=self.C2)


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
        data_dim=2,
        return_log=False,
        reduction=None,
        padding=None,
        ensemble_kernel=True,
        nonnegative=False,
        eps=1e-12,
        preprocess=None,
        reducer=None,
    ):
        """Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            window_size (int or Tuple[int], optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float or Tuple[float], optional): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
            K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
            L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            data_dim (int, optional): The dimension of the data. Defaults to 2, which means a 2d image (4d tensor).
            return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.
            reduction (str, optional): Reduction mode, one of None, "mean", "sum", "batch", "none".
            padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
            nonnegative (bool, optional): Whether to clamp SSIM/CS maps to nonnegative values. Defaults to False.
            eps (float, optional): Positive-sum threshold for weighted pooling. Defaults to 1e-12.
            preprocess (nn.Module, optional): Pairwise preprocessing module applied before SSIM core computation.
            reducer (nn.Module, optional): SSIM map reduction module. Defaults to standard/weighted reduction.

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
            ssim_caller = SSIM(return_log=True, reduction="batch").cuda()
            # setting 5: for 4d float tensors with the data range [0, 1] and 1 channel, padding=0 and the splitted kernels.
            ssim_caller = SSIM(return_log=True, reduction="batch", padding=0, ensemble_kernel=False).cuda()

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
        self.data_dim = data_dim
        self.window_size = window_size
        self.return_log = return_log
        self.reduction = Reduction.coerce(reduction)
        self.nonnegative = nonnegative

        gaussian_filter = GaussianFilter(
            data_dim=self.data_dim,
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
        )
        self.core = SSIMCore(filter=gaussian_filter, K1=K1, K2=K2, L=L)
        self.preprocess = preprocess if preprocess is not None else IdentityPreprocess()
        self.reducer = (
            reducer
            if reducer is not None
            else SSIMMapReducer(
                self.reduction,
                stable_mean=self.data_dim == 3,
                eps=eps,
            )
        )

    def _check_input_pair(self, x, y):
        if x.shape != y.shape:
            raise ValueError(f"x: {x.shape} and y: {y.shape} must be the same")
        if x.ndim != self.data_dim + 2:
            raise ValueError(f"x: {x.ndim} and y: {y.ndim} must be {self.data_dim + 2}d tensors")
        if x.device != y.device:
            raise RuntimeError(f"x and y must be on the same device, got {x.device} and {y.device}.")

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, y, weights=None, return_msssim=False):
        """Calculate the mean SSIM (MSSIM) between two 3d/4d/5d tensors.

        Args:
            x (Tensor): 3d/4d/5d tensor
            y (Tensor): 3d/4d/5d tensor
            weights (Tensor, optional): Optional nonnegative pooling weights for SSIM maps.
            return_msssim (bool, optional): Whether to return the MS-SSIM score. Defaults to False, which will return the original MSSIM score.

        Returns:
            Tensor: MSSIM or MS-SSIM
        """
        if return_msssim and self.return_log:
            raise ValueError("return_log only support return_msssim=False")
        if return_msssim and self.data_dim < 2:
            raise ValueError("return_msssim only support data_dim>=2")
        with torch.autocast(device_type=x.device.type, enabled=False):
            self._check_input_pair(x, y)

            ref = self.core.filter.gaussian_window
            if x.device != ref.device or y.device != ref.device:
                raise RuntimeError(
                    "SSIM module buffers and input tensors must be on the same device. "
                    f"Got module device {ref.device}, x device {x.device}, and y device {y.device}. "
                    "Move the module explicitly with `.to(device)` before calling forward."
                )

            if x.dtype != torch.float32:
                x = x.float()
            if y.dtype != torch.float32:
                y = y.float()

            ref = x
            x, y = self.preprocess(x, y)
            self._check_input_pair(x, y)

            if return_msssim:
                if weights is not None:
                    raise ValueError("weights are only supported for SSIM, not MS-SSIM.")
                return self.msssim(x, y)
            else:
                return self.ssim(x, y, weights=weights, ref=ref)

    def ssim(self, x, y, weights=None, ref=None):
        ssim, _ = self.core(x, y)
        if self.nonnegative:
            ssim = torch.relu(ssim)
        if self.return_log:
            # https://github.com/xuebinqin/BASNet/blob/56393818e239fed5a81d06d2a1abfe02af33e461/pytorch_ssim/__init__.py#L81-L83
            ssim = ssim - ssim.min()
            # Avoid division by zero when the whole map is constant after shifting.
            ssim = ssim / ssim.max().clamp_min(1e-12)
            ssim = -torch.log(ssim + 1e-8)

        return self.reducer(ssim, weights=weights, ref=ref)

    def msssim(self, x, y):
        ms_components = []
        component_reduction = Reduction.BATCH if self.reduction is Reduction.NONE else self.reduction
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self.core(x, y)

            ssim = _reduce_similarity(ssim, reduction=component_reduction)
            cs = _reduce_similarity(cs, reduction=component_reduction)
            if self.nonnegative:
                ssim = torch.relu(ssim)
                cs = torch.relu(cs)

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                bs, *c, h, w = x.shape
                padding = [s % 2 for s in (h, w)]  # spatial padding
                if len(c) > 1:
                    # only pooling in the spatial domain
                    x = x.reshape(bs, -1, h, w)
                    y = y.reshape(bs, -1, h, w)
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
                if len(c) > 1:
                    x = x.reshape(bs, *c, h // 2, w // 2)
                    y = y.reshape(bs, *c, h // 2, w // 2)
        msssim = math.prod(ms_components)  # equ 7 in ref2
        return _reduce_similarity(msssim, reduction=self.reduction)


class MSSSIM(SSIM):
    """MS-SSIM module using the same initialization parameters as vanilla SSIM."""

    def forward(self, x, y):
        return super().forward(x, y, return_msssim=True)


class WeightedSSIM(SSIM):
    """Weighted/ROI SSIM module.

    This variant reuses the vanilla SSIM core and only changes the reduction stage.
    The forward method requires a nonnegative weight map that is broadcastable to the SSIM map.
    All initialization parameters are inherited from :class:`SSIM`, including ``return_log``, ``reduction`` and ``eps``.
    """

    def forward(self, x, y, weights):
        return super().forward(x, y, weights=weights)


class SobelFilter(nn.Module):
    """Depthwise Sobel magnitude filter for 2D gradient-based variants.

    The filter stores a single-channel Sobel kernel as buffers and repeats it per input channel at call time so every channel is processed independently.
    ``eps`` controls the minimum squared magnitude before taking the square root, which keeps the gradient magnitude finite around flat regions.
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel_x", sobel_x.reshape(1, 1, 3, 3))
        self.register_buffer("sobel_y", self.sobel_x.transpose(-1, -2).contiguous())

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"gradient_ssim only supports 4d tensors, but got {x.ndim}d tensor.")
        if self.sobel_x.device != x.device:
            raise RuntimeError(
                "SobelFilter module buffers and input tensors must be on the same device. "
                f"Got module device {self.sobel_x.device} and input device {x.device}."
            )

        channels = x.shape[1]
        sobel_x = self.sobel_x.to(dtype=x.dtype).repeat_interleave(repeats=channels, dim=0)
        sobel_y = self.sobel_y.to(dtype=x.dtype).repeat_interleave(repeats=channels, dim=0)
        grad_x = F.conv2d(x, sobel_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, sobel_y, padding=1, groups=channels)
        squared_magnitude = grad_x * grad_x + grad_y * grad_y

        if self.eps > 0:
            # Avoid sqrt(0) around flat regions while keeping the floor in squared units.
            squared_magnitude = squared_magnitude.clamp_min(self.eps * self.eps)
        return torch.sqrt(squared_magnitude)


class GradientSSIMPreprocess(nn.Module):
    """Pairwise preprocessing module for GradientSSIM.

    It converts each input image to Sobel gradient magnitude before the shared SSIM core computes the local structural similarity map.
    """

    def __init__(self, eps=1e-12):
        super().__init__()
        self.gradient_filter = SobelFilter(eps=eps)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.gradient_filter(x), self.gradient_filter(y)


class GradientSSIM(SSIM):
    """Gradient-based SSIM module.

    GradientSSIM applies a Sobel-magnitude preprocessing step and then reuses the same SSIM core and reducer as vanilla SSIM.
    It is limited to 2D tensors because the Sobel preprocessing implemented here is 2D-only.

    Args:
        window_size (int or Tuple[int], optional): Gaussian window size. Defaults to 11.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        sigma (float or Tuple[float], optional): Gaussian sigma. Defaults to 1.5.
        K1 (float, optional): K1 of SSIM. Defaults to 0.01.
        K2 (float, optional): K2 of SSIM. Defaults to 0.03.
        L (int, optional): Dynamic range of the input values. Defaults to 1.
        padding (int or Tuple[int], optional): Gaussian padding. Defaults to None.
        ensemble_kernel (bool, optional): Whether to fuse Gaussian kernels. Defaults to True.
        reduction (str, optional): Reduction mode, one of None, "mean", "sum", "batch", "none".
        nonnegative (bool, optional): Whether to clamp SSIM maps to nonnegative values. Defaults to False.
        eps (float, optional): Shared threshold for Sobel magnitude and weighted reduction. Defaults to 1e-12.
    """

    def __init__(
        self,
        window_size=11,
        in_channels=1,
        sigma=1.5,
        *,
        K1=0.01,
        K2=0.03,
        L=1,
        padding=None,
        ensemble_kernel=True,
        reduction=None,
        nonnegative=False,
        eps=1e-12,
    ):
        super().__init__(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            K1=K1,
            K2=K2,
            L=L,
            data_dim=2,
            return_log=False,
            reduction=reduction,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
            nonnegative=nonnegative,
            eps=eps,
            preprocess=GradientSSIMPreprocess(eps=eps),
        )


class ThreeSSIMReducer(nn.Module):
    """Reducer for Three-Component Structural Similarity.

    The reducer builds edge, texture and smooth masks from the reference image, computes one pooled SSIM score for each nonempty region, and combines the active regions with normalized region weights.
    This keeps 3-SSIM as a reducer over the shared SSIM map instead of duplicating SSIM computation.

    Args:
        region_weights (Sequence[float], optional): Weights for edge, texture and smooth regions. Defaults to (0.5, 0.25, 0.25).
        edge_sigma (float, optional): Gradient threshold multiplier for edge masks. Defaults to 1.0.
        texture_sigma (float, optional): Local variance threshold multiplier for texture masks. Defaults to 0.0.
        variance_window_size (int, optional): Odd local window for variance estimation. Defaults to 7.
        reduction (str, optional): Reduction mode, one of None, "mean", "sum", "batch", "none".
        eps (float, optional): Positive-sum threshold used during normalization. Defaults to 1e-12.
    """

    def __init__(
        self,
        region_weights=(0.5, 0.25, 0.25),
        edge_sigma=1.0,
        texture_sigma=0.0,
        variance_window_size=7,
        reduction=None,
        eps=1e-12,
    ):
        super().__init__()
        if region_weights is None:
            region_weights = (0.5, 0.25, 0.25)
        region_weights = torch.as_tensor(region_weights)
        _check_weights(region_weights, name="region_weights", eps=eps)
        if region_weights.numel() != 3:
            raise ValueError(
                f"region_weights must contain 3 values for edge/texture/smooth regions, got {region_weights.numel()}."
            )
        self.region_weights = region_weights
        self.gradient_filter = SobelFilter(eps=eps)

        self.edge_sigma = edge_sigma
        self.texture_sigma = texture_sigma
        if variance_window_size % 2 != 1:
            raise ValueError(f"variance_window_size must be odd, but got {variance_window_size}.")
        self.variance_window_size = variance_window_size

        self.reduction = Reduction.coerce(reduction)
        self.eps = eps

    @staticmethod
    def _center_crop_like(x, like):
        target_h, target_w = like.shape[-2:]
        if x.shape[-2:] == (target_h, target_w):
            return x
        if x.size(-2) < target_h or x.size(-1) < target_w:
            raise ValueError(f"Cannot crop tensor with shape {x.shape} to match {like.shape}.")

        start_h = (x.size(-2) - target_h) // 2
        start_w = (x.size(-1) - target_w) // 2
        return x[..., start_h : start_h + target_h, start_w : start_w + target_w]

    @torch.no_grad()
    def _masks(self, ref):
        # Region masks are structural labels inferred from the reference image; they should not introduce gradients into the metric.
        gray = ref.mean(dim=1, keepdim=True)
        grad = self.gradient_filter(gray)

        local_mean = F.avg_pool2d(
            gray,
            kernel_size=self.variance_window_size,
            stride=1,
            padding=self.variance_window_size // 2,
        )
        local_second = F.avg_pool2d(
            gray.square(),
            kernel_size=self.variance_window_size,
            stride=1,
            padding=self.variance_window_size // 2,
        )
        # Numerical variance can be slightly negative because of floating point cancellation.
        local_variance = (local_second - local_mean.square()).clamp_min(0)

        flat_grad = grad.flatten(1)
        flat_var = local_variance.flatten(1)

        grad_threshold = flat_grad.mean(dim=1, keepdim=True) + self.edge_sigma * flat_grad.std(dim=1, keepdim=True)
        var_threshold = flat_var.mean(dim=1, keepdim=True) + self.texture_sigma * flat_var.std(dim=1, keepdim=True)
        grad_threshold = grad_threshold.reshape(-1, 1, 1, 1)
        var_threshold = var_threshold.reshape(-1, 1, 1, 1)

        edge = grad >= grad_threshold
        texture = (grad < grad_threshold) & (local_variance >= var_threshold)
        smooth = ~(edge | texture)
        return edge.float(), texture.float(), smooth.float()

    def forward(self, score_map, weights=None, ref=None):
        if weights is not None:
            raise ValueError("ThreeSSIMReducer infers its own weights from edge/texture/smooth regions.")
        if ref is None:
            raise ValueError("ThreeSSIMReducer requires ref to infer edge/texture/smooth masks.")
        masks = self._masks(ref)

        region_weights = torch.as_tensor(self.region_weights, device=score_map.device, dtype=score_map.dtype)
        _check_weights(region_weights, name="region_weights", eps=self.eps)

        dims = tuple(range(1, score_map.ndim))
        weight_map = torch.zeros_like(score_map)
        active_weight_sum = score_map.new_zeros((score_map.shape[0],) + (1,) * (score_map.ndim - 1))
        for mask, region_weight in zip(masks, region_weights):
            mask = self._center_crop_like(mask, score_map).to(device=score_map.device, dtype=score_map.dtype)
            # Broadcast single-channel region masks across the SSIM map channels.
            mask = torch.broadcast_to(mask, score_map.shape)
            mask_sum = mask.sum(dim=dims, keepdim=True)
            active = mask_sum > self.eps
            active_weight = region_weight * active.to(dtype=score_map.dtype)
            # Empty regions are ignored.
            # Nonempty regions are normalized by area first, then weighted by their region weight.
            weight_map = weight_map + active_weight * mask / mask_sum.clamp_min(self.eps)
            active_weight_sum = active_weight_sum + active_weight
        # Re-normalize when one of edge/texture/smooth is absent for a batch item.
        weights = weight_map / active_weight_sum.clamp_min(self.eps)

        return _reduce_similarity(score_map, reduction=self.reduction, weights=weights, eps=self.eps)


class ThreeSSIM(SSIM):
    """Three-Component Structural Similarity module.

    ThreeSSIM keeps the vanilla SSIM preprocessing and core computation, then replaces the reducer with :class:`ThreeSSIMReducer`.
    The reducer infers edge, texture and smooth masks from the reference input and pools the SSIM map by these automatically generated regions.

    Args:
        window_size (int or Tuple[int], optional): Gaussian window size. Defaults to 11.
        in_channels (int, optional): Number of input channels. Defaults to 1.
        sigma (float or Tuple[float], optional): Gaussian sigma. Defaults to 1.5.
        region_weights (Sequence[float], optional): Weights for edge, texture and smooth regions.
        edge_sigma (float, optional): Gradient threshold multiplier for edge masks. Defaults to 1.0.
        texture_sigma (float, optional): Variance threshold multiplier for texture masks. Defaults to 0.0.
        variance_window_size (int, optional): Odd local window for variance estimation. Defaults to 7.
        K1 (float, optional): K1 of SSIM. Defaults to 0.01.
        K2 (float, optional): K2 of SSIM. Defaults to 0.03.
        L (int, optional): Dynamic range of the input values. Defaults to 1.
        data_dim (int, optional): Data dimension. Currently expected to be 2. Defaults to 2.
        padding (int or Tuple[int], optional): Gaussian padding. Defaults to None.
        ensemble_kernel (bool, optional): Whether to fuse Gaussian kernels. Defaults to True.
        reduction (str, optional): Reduction mode, one of None, "mean", "sum", "batch", "none".
        nonnegative (bool, optional): Whether to clamp SSIM maps to nonnegative values. Defaults to False.
        eps (float, optional): Positive-sum threshold used during region pooling. Defaults to 1e-12.
    """

    def __init__(
        self,
        window_size=11,
        in_channels=1,
        sigma=1.5,
        *,
        region_weights=None,
        edge_sigma=1.0,
        texture_sigma=0.0,
        variance_window_size=7,
        K1=0.01,
        K2=0.03,
        L=1,
        data_dim=2,
        padding=None,
        ensemble_kernel=True,
        reduction=None,
        nonnegative=False,
        eps=1e-12,
    ):
        super().__init__(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            K1=K1,
            K2=K2,
            L=L,
            data_dim=data_dim,
            return_log=False,
            reduction=reduction,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
            nonnegative=nonnegative,
            reducer=ThreeSSIMReducer(
                region_weights=region_weights,
                edge_sigma=edge_sigma,
                texture_sigma=texture_sigma,
                variance_window_size=variance_window_size,
                reduction=reduction,
                eps=eps,
            ),
        )


def ssim(
    x,
    y,
    weights=None,
    *,
    window_size=11,
    in_channels=1,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    data_dim=2,
    return_log=False,
    return_msssim=False,
    reduction=None,
    padding=None,
    ensemble_kernel=True,
    nonnegative=False,
    eps=1e-12,
):
    """Calculate the mean SSIM (MSSIM) between two 4D tensors.

    Args:
        x (Tensor): 4d tensor
        y (Tensor): 4d tensor
        weights (Tensor, optional): Optional nonnegative pooling weights broadcastable to the SSIM map.
        window_size (int or Tuple[int], optional): The window size of the gaussian filter. Defaults to 11.
        in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
        sigma (float or Tuple[float], optional): The sigma of the gaussian filter. Defaults to 1.5.
        K1 (float, optional): K1 of MSSIM. Defaults to 0.01.
        K2 (float, optional): K2 of MSSIM. Defaults to 0.03.
        L (int, optional): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
        data_dim (int, optional): The dimension of the data. Defaults to 2, which means a 2d image (4d tensor).
        return_log (bool, optional): Whether to return the logarithmic form. Defaults to False.
        return_msssim (bool, optional): Whether to return the MS-SSIM score. Defaults to False, which will return the original MSSIM score.
        reduction (str, optional): Reduction mode, one of None, "mean", "sum", "batch", "none".
        padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
        ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
        nonnegative (bool, optional): Whether to clamp SSIM/CS maps to nonnegative values. Defaults to False.
        eps (float, optional): Positive-sum threshold for weighted pooling. Defaults to 1e-12.

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
        data_dim=data_dim,
        return_log=return_log,
        reduction=reduction,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
        nonnegative=nonnegative,
        eps=eps,
    ).to(device=x.device)
    return ssim_obj(x, y, weights=weights, return_msssim=return_msssim)


def ms_ssim(
    x,
    y,
    *,
    window_size=11,
    in_channels=1,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    data_dim=2,
    reduction=None,
    padding=None,
    ensemble_kernel=True,
    nonnegative=False,
):
    """Calculate the Multi-Scale Structural Similarity (MS-SSIM) score."""
    metric = SSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        K1=K1,
        K2=K2,
        L=L,
        data_dim=data_dim,
        reduction=reduction,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
        nonnegative=nonnegative,
    ).to(device=x.device)
    return metric(x, y, return_msssim=True)


def weighted_ssim(
    x,
    y,
    weights,
    *,
    window_size=11,
    in_channels=1,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    data_dim=2,
    return_log=False,
    reduction=None,
    padding=None,
    ensemble_kernel=True,
    nonnegative=False,
    eps=1e-12,
):
    """Calculate a Weighted/ROI Structural Similarity score from the SSIM map."""
    metric = WeightedSSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        K1=K1,
        K2=K2,
        L=L,
        data_dim=data_dim,
        return_log=return_log,
        reduction=reduction,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
        nonnegative=nonnegative,
        eps=eps,
    ).to(device=x.device)
    return metric(x, y, weights)


def gradient_ssim(
    x,
    y,
    *,
    window_size=11,
    in_channels=None,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    reduction=None,
    padding=None,
    ensemble_kernel=True,
    nonnegative=False,
    eps=1e-12,
):
    """Calculate Gradient-based Structural Similarity on Sobel magnitude maps."""
    if in_channels is None:
        in_channels = x.shape[1]
    metric = GradientSSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        K1=K1,
        K2=K2,
        L=L,
        reduction=reduction,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
        nonnegative=nonnegative,
        eps=eps,
    ).to(device=x.device)
    return metric(x, y)


def three_ssim(
    x,
    y,
    *,
    region_weights=None,
    edge_sigma=1.0,
    texture_sigma=0.0,
    variance_window_size=7,
    window_size=11,
    in_channels=1,
    sigma=1.5,
    K1=0.01,
    K2=0.03,
    L=1,
    data_dim=2,
    reduction=None,
    padding=None,
    ensemble_kernel=True,
    nonnegative=False,
    eps=1e-12,
):
    """Calculate Three-Component Structural Similarity (3-SSIM)."""
    metric = ThreeSSIM(
        window_size=window_size,
        in_channels=in_channels,
        sigma=sigma,
        region_weights=region_weights,
        edge_sigma=edge_sigma,
        texture_sigma=texture_sigma,
        variance_window_size=variance_window_size,
        K1=K1,
        K2=K2,
        L=L,
        data_dim=data_dim,
        reduction=reduction,
        padding=padding,
        ensemble_kernel=ensemble_kernel,
        nonnegative=nonnegative,
        eps=eps,
    ).to(device=x.device)
    return metric(x, y)
