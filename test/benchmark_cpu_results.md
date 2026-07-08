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

| case | shape | data_dim | window_size | padding | ensemble_kernel | reduction | return_log | return_msssim |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| test_ssim1d | `(2, 1, 128)` | `1` | `11` | `None` | `True` | `mean` | `False` | `False` |
| test_ssim3d | `(2, 1, 5, 128, 128)` | `3` | `11` | `None` | `True` | `mean` | `False` | `False` |
| test_mssim3d_split | `(2, 1, 5, 128, 128)` | `3` | `(3, 11, 11)` | `None` | `False` | `mean` | `False` | `True` |
| test_mssim3d_ensemble | `(2, 1, 5, 128, 128)` | `3` | `(3, 11, 11)` | `None` | `True` | `mean` | `False` | `True` |
| oldversion_ssim2d_w11_mean | `(2, 1, 128, 128)` | `2` | `11` | `None` | `True` | `mean` | `False` | `False` |
| oldversion_ssim2d_w5_mean | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `mean` | `False` | `False` |
| oldversion_ssim2d_w5_batch | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `False` | `False` |
| oldversion_ssim2d_w5_batch_log | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `batch` | `True` | `False` |
| oldversion_msssim2d_w5_batch_ensemble | `(2, 1, 192, 192)` | `2` | `5` | `None` | `True` | `batch` | `False` | `True` |
| oldversion_msssim2d_w5_batch_split | `(2, 1, 192, 192)` | `2` | `5` | `None` | `False` | `batch` | `False` | `True` |
| pohsunsu_ssim2d_w11 | `(2, 1, 128, 128)` | `2` | `11` | `None` | `True` | `mean` | `False` | `False` |
| pohsunsu_ssim2d_w5 | `(2, 1, 128, 128)` | `2` | `5` | `None` | `True` | `mean` | `False` | `False` |
| vainf_ssim2d_w11_split | `(2, 1, 128, 128)` | `2` | `11` | `0` | `False` | `mean` | `False` | `False` |
| vainf_ssim2d_w5_split | `(2, 1, 128, 128)` | `2` | `5` | `0` | `False` | `mean` | `False` | `False` |
| vainf_msssim2d_w11_split | `(2, 1, 192, 192)` | `2` | `11` | `0` | `False` | `mean` | `False` | `True` |
| vainf_msssim2d_w5_split | `(2, 1, 192, 192)` | `2` | `5` | `0` | `False` | `mean` | `False` | `True` |

## Mean Runtime (ms)

| case | current | v1.0.0 | po_hsun_su | vainf |
| --- | ---: | ---: | ---: | ---: |
| test_ssim1d | 0.520 | N/A | N/A | N/A |
| test_ssim3d | 2711.446 | N/A | N/A | 10.368 |
| test_mssim3d_split | 93.600 | N/A | N/A | N/A |
| test_mssim3d_ensemble | 1597.726 | N/A | N/A | N/A |
| oldversion_ssim2d_w11_mean | 19.795 | 19.535 | 19.265 | 2.204 |
| oldversion_ssim2d_w5_mean | 1.930 | 2.021 | 1.715 | 1.781 |
| oldversion_ssim2d_w5_batch | 1.909 | 1.850 | 1.842 | 1.814 |
| oldversion_ssim2d_w5_batch_log | 2.208 | 2.137 | N/A | N/A |
| oldversion_msssim2d_w5_batch_ensemble | 7.635 | 7.433 | N/A | 9.278 |
| oldversion_msssim2d_w5_batch_split | 10.628 | 10.924 | N/A | 8.877 |
| pohsunsu_ssim2d_w11 | 20.133 | 19.883 | 19.458 | 2.143 |
| pohsunsu_ssim2d_w5 | 1.854 | 1.884 | 1.798 | 1.795 |
| vainf_ssim2d_w11_split | 2.155 | 2.153 | N/A | 2.157 |
| vainf_ssim2d_w5_split | 1.765 | 1.763 | N/A | 1.804 |
| vainf_msssim2d_w11_split | 9.517 | 9.513 | N/A | 9.637 |
| vainf_msssim2d_w5_split | 9.967 | 8.955 | N/A | 9.050 |

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
