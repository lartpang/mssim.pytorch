# MSSIM.pytorch

A better pytorch-based implementation for the mean structural similarity (MSSIM).

Compared to this widely used implementation: <https://github.com/Po-Hsun-Su/pytorch-ssim>, I further optimized and refactored the code.

At the same time, in this implementation, I have dealt with the problem that the calculation with the fp16 mode cannot be consistent with the calculation with the fp32 mode. Typecasting is used here to ensure that the computation is done in fp32 mode. This might also avoid unexpected results when using it as a loss.

## Structural similarity index

> When comparing images, the mean squared error (MSE)–while simple to implement–is not highly indicative of perceived similarity. Structural similarity aims to address this shortcoming by taking texture into account. More details can be seen at https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html?highlight=structure+similarity

![results](https://user-images.githubusercontent.com/26847524/174805728-81e8502b-2ecb-4b40-a2c4-b4f1e2361ea9.png)

```python
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_ssim import SSIM, ssim
from skimage import data, img_as_float

img = img_as_float(data.camera())
rows, cols = img.shape

noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
rng = np.random.default_rng()
noise[rng.random(size=noise.shape) > 0.5] *= -1

img_noise = img + noise
img_const = img + abs(noise)

img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
img_noise_tensor = torch.from_numpy(img_noise).unsqueeze(0).unsqueeze(0).float()
img_const_tensor = torch.from_numpy(img_const).unsqueeze(0).unsqueeze(0).float()

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8), sharex=True, sharey=True)
ax = axes.ravel()

mse_none = F.mse_loss(img_tensor, img_tensor, reduction="mean")
ssim_none = ssim(img_tensor, img_tensor, L=img_tensor.max() - img_tensor.min())

mse_noise = F.mse_loss(img_tensor, img_noise_tensor, reduction="mean")
ssim_noise = ssim(img_tensor, img_noise_tensor, L=img_noise_tensor.max() - img_noise_tensor.min())

mse_const = F.mse_loss(img_tensor, img_const_tensor, reduction="mean")
ssim_const = ssim(img_tensor, img_const_tensor, L=img_const_tensor.max() - img_const_tensor.min())

ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[0].set_xlabel(f"MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}")
ax[0].set_title("Original image")

ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[1].set_xlabel(f"MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}")
ax[1].set_title("Image with noise")

ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[2].set_xlabel(f"MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}")
ax[2].set_title("Image plus constant")

mse_none = F.mse_loss(img_tensor, img_tensor, reduction="mean")
ssim_none = SSIM(L=img_tensor.max() - img_tensor.min())(img_tensor, img_tensor)

mse_noise = F.mse_loss(img_tensor, img_noise_tensor, reduction="mean")
ssim_noise = SSIM(L=img_noise_tensor.max() - img_noise_tensor.min())(img_tensor, img_noise_tensor)

mse_const = F.mse_loss(img_tensor, img_const_tensor, reduction="mean")
ssim_const = SSIM(L=img_const_tensor.max() - img_const_tensor.min())(img_tensor, img_const_tensor)

ax[3].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[3].set_xlabel(f"MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}")
ax[3].set_title("Original image")

ax[4].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[4].set_xlabel(f"MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}")
ax[4].set_title("Image with noise")

ax[5].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
ax[5].set_xlabel(f"MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}")
ax[5].set_title("Image plus constant")

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

## Reference

- https://github.com/Po-Hsun-Su/pytorch-ssim
- https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html?highlight=structure+similarity
- Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli, “Image quality assessment: From error visibility to structural similarity,” IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, Apr. 2004.
