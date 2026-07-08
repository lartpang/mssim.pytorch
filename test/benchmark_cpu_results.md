# CPU benchmark across test_ssim scenarios

- baseline ref: `v1.0.0:ssim.py`
- input image size for generated 2D tensors: `128x128`
- input image size for generated 2D MS-SSIM tensors: `192x192`
- warmup: 5
- iterations: 20
- torch threads: 1
- values are mean milliseconds per forward call
- N/A means that implementation cannot express the same scenario/API

## Scenario Configurations

| case | metric | shape | data_dim | window_size | padding | ensemble_kernel | reduction | return_log | return_msssim |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| test_ssim1d | `ssim` | `(2, 1, 128)` | `1` | `11` | `None` | `True` | `mean` | `False` | `False` |
| test_ssim3d | `ssim` | `(2, 1, 5, 128, 128)` | `3` | `11` | `None` | `True` | `mean` | `False` | `False` |
| test_mssim3d_split | `ssim` | `(2, 1, 5, 128, 128)` | `3` | `(3, 11, 11)` | `None` | `False` | `mean` | `False` | `True` |
| test_mssim3d_ensemble | `ssim` | `(2, 1, 5, 128, 128)` | `3` | `(3, 11, 11)` | `None` | `True` | `mean` | `False` | `True` |
| oldversion_ssim2d_w11_mean | `ssim` | `(2, 1, 128, 128)` | `2` | `11` | `None` | `True` | `mean` | `False` | `False` |
| oldversion_ssim2d_w5_mean | `ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `mean` | `False` | `False` |
| oldversion_ssim2d_w5_batch | `ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `False` | `False` |
| oldversion_ssim2d_w5_batch_log | `ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `True` | `False` |
| oldversion_msssim2d_w5_batch_ensemble | `ssim` | `(2, 1, 192, 192)` | `2` | `5` | `None` | `True` | `batch` | `False` | `True` |
| oldversion_msssim2d_w5_batch_split | `ssim` | `(2, 1, 192, 192)` | `2` | `5` | `None` | `False` | `batch` | `False` | `True` |
| pohsunsu_ssim2d_w11 | `ssim` | `(2, 1, 128, 128)` | `2` | `11` | `None` | `True` | `mean` | `False` | `False` |
| pohsunsu_ssim2d_w5 | `ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `mean` | `False` | `False` |
| vainf_ssim2d_w11_split | `ssim` | `(2, 1, 128, 128)` | `2` | `11` | `0` | `False` | `mean` | `False` | `False` |
| vainf_ssim2d_w5_split | `ssim` | `(2, 1, 128, 128)` | `2` | `5` | `0` | `False` | `mean` | `False` | `False` |
| vainf_msssim2d_w11_split | `ssim` | `(2, 1, 192, 192)` | `2` | `11` | `0` | `False` | `mean` | `False` | `True` |
| vainf_msssim2d_w5_split | `ssim` | `(2, 1, 192, 192)` | `2` | `5` | `0` | `False` | `mean` | `False` | `True` |
| weighted_ssim2d_w5_batch | `weighted_ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `False` | `False` |
| gradient_ssim2d_w5_batch | `gradient_ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `False` | `False` |
| three_ssim2d_w5_batch | `three_ssim` | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `False` | `False` |

## Mean Runtime (ms)

| case | current | v1.0.0 | po_hsun_su | vainf |
| --- | ---: | ---: | ---: | ---: |
| test_ssim1d | 0.537 | N/A | N/A | N/A |
| test_ssim3d | 2752.957 | N/A | N/A | 10.341 |
| test_mssim3d_split | 92.958 | N/A | N/A | N/A |
| test_mssim3d_ensemble | 1591.262 | N/A | N/A | N/A |
| oldversion_ssim2d_w11_mean | 19.290 | 19.313 | 19.297 | 2.209 |
| oldversion_ssim2d_w5_mean | 1.848 | 1.795 | 1.700 | 1.812 |
| oldversion_ssim2d_w5_batch | 1.950 | 1.864 | 1.827 | 1.897 |
| oldversion_ssim2d_w5_batch_log | 2.206 | 2.138 | N/A | N/A |
| oldversion_msssim2d_w5_batch_ensemble | 8.060 | 7.799 | N/A | 9.045 |
| oldversion_msssim2d_w5_batch_split | 10.954 | 10.722 | N/A | 8.573 |
| pohsunsu_ssim2d_w11 | 20.859 | 19.579 | 19.617 | 2.282 |
| pohsunsu_ssim2d_w5 | 1.923 | 1.799 | 1.792 | 1.822 |
| vainf_ssim2d_w11_split | 2.221 | 2.113 | N/A | 2.215 |
| vainf_ssim2d_w5_split | 1.774 | 1.751 | N/A | 1.812 |
| vainf_msssim2d_w11_split | 9.870 | 11.874 | N/A | 9.955 |
| vainf_msssim2d_w5_split | 9.220 | 8.985 | N/A | 8.747 |
| weighted_ssim2d_w5_batch | 2.289 | N/A | N/A | N/A |
| gradient_ssim2d_w5_batch | 2.738 | N/A | N/A | N/A |
| three_ssim2d_w5_batch | 6.346 | N/A | N/A | N/A |

## Unsupported Details

- `test_ssim1d`: v1.0.0: NotImplementedError: v1.0.0 only supports 2D tensors; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation only supports 2D tensors; vainf: NotImplementedError: VainF implementation supports 2D/3D tensors
- `test_ssim3d`: v1.0.0: NotImplementedError: v1.0.0 only supports 2D tensors; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation only supports 2D tensors
- `test_mssim3d_split`: v1.0.0: NotImplementedError: v1.0.0 only supports 2D tensors; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation only supports 2D tensors; vainf: NotImplementedError: VainF implementation supports only integer window_size
- `test_mssim3d_ensemble`: v1.0.0: NotImplementedError: v1.0.0 only supports 2D tensors; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation only supports 2D tensors; vainf: NotImplementedError: VainF implementation supports only integer window_size
- `oldversion_ssim2d_w5_batch_log`: po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no return_log; vainf: NotImplementedError: VainF implementation has no return_log
- `oldversion_msssim2d_w5_batch_ensemble`: po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no MS-SSIM
- `oldversion_msssim2d_w5_batch_split`: po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no MS-SSIM
- `vainf_ssim2d_w11_split`: po_hsun_su: NotImplementedError: Po-Hsun-Su comparison only covers padded ensemble-kernel SSIM
- `vainf_ssim2d_w5_split`: po_hsun_su: NotImplementedError: Po-Hsun-Su comparison only covers padded ensemble-kernel SSIM
- `vainf_msssim2d_w11_split`: po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no MS-SSIM
- `vainf_msssim2d_w5_split`: po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no MS-SSIM
- `weighted_ssim2d_w5_batch`: v1.0.0: NotImplementedError: v1.0.0 has no variant SSIM APIs; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no variant SSIM APIs; vainf: NotImplementedError: VainF implementation has no variant SSIM APIs
- `gradient_ssim2d_w5_batch`: v1.0.0: NotImplementedError: v1.0.0 has no variant SSIM APIs; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no variant SSIM APIs; vainf: NotImplementedError: VainF implementation has no variant SSIM APIs
- `three_ssim2d_w5_batch`: v1.0.0: NotImplementedError: v1.0.0 has no variant SSIM APIs; po_hsun_su: NotImplementedError: Po-Hsun-Su implementation has no variant SSIM APIs; vainf: NotImplementedError: VainF implementation has no variant SSIM APIs
