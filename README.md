# MSSIM.pytorch

A better pytorch-based implementation for the mean structural similarity (MSSIM).

Compared to this widely used implementation: <https://github.com/Po-Hsun-Su/pytorch-ssim>, I further optimized and refactored the code.

At the same time, in this implementation, I have dealt with the problem that the calculation with the fp16 mode cannot be consistent with the calculation with the fp32 mode. Typecasting is used here to ensure that the computation is done in fp32 mode. This might also avoid unexpected results when using it as a loss.

## Structural similarity index

> When comparing images, the mean squared error (MSE)–while simple to implement–is not highly indicative of perceived similarity. Structural similarity aims to address this shortcoming by taking texture into account. More details can be seen at https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html?highlight=structure+similarity

![results](https://user-images.githubusercontent.com/26847524/175031400-92426661-4536-43c7-8f6e-5c470fb9ccb5.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lartpang_ssim import SSIM
from po_hsun_su_ssim import SSIM as PoHsunSuSSIM
from vainf_ssim import MS_SSIM as VainFMSSSIM
from vainf_ssim import SSIM as VainFSSIM
from skimage import data, img_as_float

img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.3 * (img.max() - img.min())
rng = np.random.default_rng()
noise[rng.random(size=noise.shape) > 0.5] *= -1

img_noise = img + noise
img_const = np.zeros_like(img)

img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
img_noise_tensor = torch.from_numpy(img_noise).unsqueeze(0).unsqueeze(0).float()
img_const_tensor = torch.from_numpy(img_const).unsqueeze(0).unsqueeze(0).float()

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
ax = axes.ravel()

mse_none = F.mse_loss(img_tensor, img_tensor, reduction="mean")
mse_noise = F.mse_loss(img_tensor, img_noise_tensor, reduction="mean")
mse_const = F.mse_loss(img_tensor, img_const_tensor, reduction="mean")

# https://github.com/VainF/pytorch-msssim
vainf_ssim_none = VainFSSIM(channel=1, data_range=1)(img_tensor, img_tensor)
vainf_ssim_noise = VainFSSIM(channel=1, data_range=1)(img_tensor, img_noise_tensor)
vainf_ssim_const = VainFSSIM(channel=1, data_range=1)(img_tensor, img_const_tensor)
vainf_ms_ssim_none = VainFMSSSIM(channel=1, data_range=1)(img_tensor, img_tensor)
vainf_ms_ssim_noise = VainFMSSSIM(channel=1, data_range=1)(img_tensor, img_noise_tensor)
vainf_ms_ssim_const = VainFMSSSIM(channel=1, data_range=1)(img_tensor, img_const_tensor)

# use the settings of https://github.com/VainF/pytorch-msssim
ssim_none_0 = SSIM(return_msssim=False, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_tensor)
ssim_noise_0 = SSIM(return_msssim=False, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_noise_tensor)
ssim_const_0 = SSIM(return_msssim=False, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_const_tensor)
ms_ssim_none_0 = SSIM(return_msssim=True, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_tensor)
ms_ssim_noise_0 = SSIM(return_msssim=True, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_noise_tensor)
ms_ssim_const_0 = SSIM(return_msssim=True, L=1, padding=0, ensemble_kernel=False)(img_tensor, img_const_tensor)

# https://github.com/Po-Hsun-Su/pytorch-ssim
pohsunsu_ssim_none = PoHsunSuSSIM()(img_tensor, img_tensor)
pohsunsu_ssim_noise = PoHsunSuSSIM()(img_tensor, img_noise_tensor)
pohsunsu_ssim_const = PoHsunSuSSIM()(img_tensor, img_const_tensor)

# use the settings of https://github.com/Po-Hsun-Su/pytorch-ssim
ssim_none_1 = SSIM(return_msssim=False, L=1, padding=None, ensemble_kernel=True)(img_tensor, img_tensor)
ssim_noise_1 = SSIM(return_msssim=False, L=1, padding=None, ensemble_kernel=True)(img_tensor, img_noise_tensor)
ssim_const_1 = SSIM(return_msssim=False, L=1, padding=None, ensemble_kernel=True)(img_tensor, img_const_tensor)


ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(
    f"MSE: {mse_none:.6f}\n"
    f"SSIM {ssim_none_0:.6f}, MS-SSIM {ms_ssim_none_0:.6f}\n"
    f"(VainF) SSIM: {vainf_ssim_none:.6f}, MS-SSIM {vainf_ms_ssim_none:.6f}\n"
    f"SSIM {ssim_none_1:.6f}\n"
    f"(PoHsunSu) SSIM: {pohsunsu_ssim_none:.6f}\n"
)
ax[0].set_title("Original image")

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(
    f"MSE: {mse_noise:.6f}\n"
    f"SSIM {ssim_noise_0:.6f}, MS-SSIM {ms_ssim_noise_0:.6f}\n"
    f"(VainF) SSIM: {vainf_ssim_noise:.6f}, MS-SSIM {vainf_ms_ssim_noise:.6f}\n"
    f"SSIM {ssim_noise_1:.6f}\n"
    f"(PoHsunSu) SSIM: {pohsunsu_ssim_noise:.6f}\n"
)
ax[1].set_title("Image with noise")

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(
    f"MSE: {mse_const:.6f}\n"
    f"SSIM {ssim_const_0:.6f}, MS-SSIM {ms_ssim_const_0:.6f}\n"
    f"(VainF) SSIM: {vainf_ssim_const:.6f}, MS-SSIM {vainf_ms_ssim_const:.6f}\n"
    f"SSIM {ssim_const_1:.6f}\n"
    f"(PoHsunSu) SSIM: {pohsunsu_ssim_const:.6f}\n"
)
ax[2].set_title("Image plus constant")


[ax[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[]) for i in range(len(axes))]

plt.tight_layout()
plt.savefig("results.png")
```

## More Examples

```python
# setting 4: for 4d float tensors with the data range [0, 1] and 1 channel,return the logarithmic form, and keep the batch dim
ssim_caller = SSIM(return_log=True, keep_batch_dim=True).cuda()

# two 4d tensors
x = torch.randn(3, 1, 100, 100).cuda()
y = torch.randn(3, 1, 100, 100).cuda()
ssim_score_0 = ssim_caller(x, y)
# or in the fp16 mode (we have fixed the computation progress into the float32 mode to avoid the unexpected result)
with torch.cuda.amp.autocast(enabled=True):
    ssim_score_1 = ssim_caller(x, y)
assert torch.allclose(ssim_score_0, ssim_score_1)
print(ssim_score_0.shape, ssim_score_1.shape)
```

## As A Loss

As you can see from the respective thresholds of the two cases below, it is easier to optimize towards MSSIM=1 than MSSIM=-1.

### Optimize towards MSSIM=1

![prediction](https://user-images.githubusercontent.com/26847524/174930091-9d7f7505-1752-423a-b7c3-d4dbfeb8d336.png)

```python
import matplotlib.pyplot as plt
import torch
from pytorch_ssim import SSIM
from skimage import data
from torch import optim

original_image = data.moon() / 255
target_image = torch.from_numpy(original_image).unsqueeze(0).unsqueeze(0).float().cuda()
predicted_image = torch.zeros_like(
    target_image, device=target_image.device, dtype=target_image.dtype, requires_grad=True
)
initial_image = predicted_image.clone()

ssim = SSIM().cuda()
initial_ssim_value = ssim(predicted_image, target_image)

ssim_value = initial_ssim_value
optimizer = optim.Adam([predicted_image], lr=0.01)
loss_curves = []
while ssim_value < 0.999:
    ssim_out = 1 - ssim(predicted_image, target_image)
    loss_curves.append(ssim_out.item())
    ssim_value = 1 - ssim_out.item()
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()
    optimizer.zero_grad()

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original_image, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_title("Original Image")

ax[1].imshow(initial_image.squeeze().detach().cpu().numpy(), cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f"SSIM: {initial_ssim_value:.5f}")
ax[1].set_title("Initial Image")

ax[2].imshow(predicted_image.squeeze().detach().cpu().numpy(), cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f"SSIM: {ssim_value:.5f}")
ax[2].set_title("Predicted Image")

ax[3].plot(loss_curves)
ax[3].set_title("SSIM Loss Curve")

ax[4].set_title("Original Image")
ax[4].hist(original_image.ravel(), bins=256)
ax[4].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[4].set_xlabel("Pixel Intensity")

ax[5].set_title("Initial Image")
ax[5].hist(initial_image.squeeze().detach().cpu().numpy().ravel(), bins=256)
ax[5].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[5].set_xlabel("Pixel Intensity")

ax[6].set_title("Predicted Image")
ax[6].hist(predicted_image.squeeze().detach().cpu().numpy().ravel(), bins=256)
ax[6].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[6].set_xlabel("Pixel Intensity")

plt.tight_layout()
plt.savefig("prediction.png")
```

### Optimize towards MSSIM=-1

![prediction](https://user-images.githubusercontent.com/26847524/174929574-5332cab2-104f-4aab-a4e5-35e7635a793f.png)

```python
import matplotlib.pyplot as plt
import torch
from pytorch_ssim import SSIM
from skimage import data
from torch import optim

original_image = data.moon() / 255
target_image = torch.from_numpy(original_image).unsqueeze(0).unsqueeze(0).float().cuda()
predicted_image = torch.zeros_like(
    target_image, device=target_image.device, dtype=target_image.dtype, requires_grad=True
)
initial_image = predicted_image.clone()

ssim = SSIM(L=original_image.max() - original_image.min()).cuda()
initial_ssim_value = ssim(predicted_image, target_image)

ssim_value = initial_ssim_value
optimizer = optim.Adam([predicted_image], lr=0.01)
loss_curves = []
while ssim_value > -0.94:
    ssim_out = ssim(predicted_image, target_image)
    loss_curves.append(ssim_out.item())
    ssim_value = ssim_out.item()
    print(ssim_value)
    ssim_out.backward()
    optimizer.step()
    optimizer.zero_grad()

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original_image, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_title("Original Image")

ax[1].imshow(initial_image.squeeze().detach().cpu().numpy(), cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f"SSIM: {initial_ssim_value:.5f}")
ax[1].set_title("Initial Image")

ax[2].imshow(predicted_image.squeeze().detach().cpu().numpy(), cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f"SSIM: {ssim_value:.5f}")
ax[2].set_title("Predicted Image")

ax[3].plot(loss_curves)
ax[3].set_title("SSIM Loss Curve")

ax[4].set_title("Original Image")
ax[4].hist(original_image.ravel(), bins=256)
ax[4].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[4].set_xlabel("Pixel Intensity")

ax[5].set_title("Initial Image")
ax[5].hist(initial_image.squeeze().detach().cpu().numpy().ravel(), bins=256)
ax[5].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[5].set_xlabel("Pixel Intensity")

ax[6].set_title("Predicted Image")
ax[6].hist(predicted_image.squeeze().detach().cpu().numpy().ravel(), bins=256)
ax[6].ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
ax[6].set_xlabel("Pixel Intensity")

plt.tight_layout()
plt.savefig("prediction.png")
```

## Reference

- https://github.com/Po-Hsun-Su/pytorch-ssim
- https://github.com/VainF/pytorch-msssim
- https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html?highlight=structure+similarity
- Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
