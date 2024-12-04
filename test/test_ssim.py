import sys
import unittest

import numpy as np
import old_version
import po_hsun_su_ssim
import torch
import vainf_ssim
from skimage import data, img_as_float

sys.path.append("..")
import ssim


class CheckSSIMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        img = img_as_float(data.camera())
        noise = np.ones_like(img) * 0.3 * (img.max() - img.min())
        rng = np.random.default_rng(seed=20241204)
        noise[rng.random(size=noise.shape) > 0.5] *= -1
        img_noise = img + noise

        cls.x_2d = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=0).float()  # 2,1,H,W
        cls.y_2d = torch.from_numpy(img_noise).unsqueeze(0).unsqueeze(0).repeat_interleave(2, dim=0).float()

        cls.x_1d = cls.x_2d.mean(dim=-1)  # 2,1,H
        cls.y_1d = cls.y_2d.mean(dim=-1)  # 2,1,H

        cls.x_3d = cls.x_2d.unsqueeze(dim=2).repeat_interleave(5, dim=2)  # 2,1,5,H,W
        cls.y_3d = cls.y_2d.unsqueeze(dim=2).repeat_interleave(5, dim=2)  # 2,1,5,H,W

    def test_ssim1d(self):
        our_ssim_score = ssim.ssim(
            self.x_1d, self.y_1d, return_msssim=False, L=1, padding=None, ensemble_kernel=True, data_dim=1
        ).item()
        self.assertEqual(our_ssim_score, 0.8740299940109253)

    def test_mssim1d(self):
        with self.assertRaises(ValueError):
            ssim.ssim(self.x_1d, self.y_1d, return_msssim=True, L=1, padding=None, ensemble_kernel=True, data_dim=1)

    def test_ssim3d(self):
        our_ssim_score = ssim.ssim(
            self.x_3d, self.y_3d, return_msssim=False, L=1, padding=None, ensemble_kernel=True, data_dim=3
        ).item()
        self.assertEqual(our_ssim_score, 0.4981585144996643)

    def test_mssim3d(self):
        our_ssim_score = ssim.ssim(
            self.x_3d,
            self.y_3d,
            return_msssim=True,
            L=1,
            ensemble_kernel=False,
            data_dim=3,
            window_size=(3, 11, 11),
        ).item()
        self.assertEqual(our_ssim_score, 0.8404632806777954)

        our_ssim_score = ssim.ssim(
            self.x_3d,
            self.y_3d,
            return_msssim=True,
            L=1,
            ensemble_kernel=True,
            data_dim=3,
            window_size=(3, 11, 11),
        ).item()
        self.assertEqual(our_ssim_score, 0.6245790719985962)

    def test_ssim2d_with_oldversion(self):
        kwargs = dict(
            window_size=11, L=1, keep_batch_dim=False, return_log=False, return_msssim=False, ensemble_kernel=True
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d).item()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).item()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

        kwargs = dict(
            window_size=5, L=1, keep_batch_dim=False, return_log=False, return_msssim=False, ensemble_kernel=True
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d, **kwargs).item()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).item()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

        kwargs = dict(
            window_size=5, L=1, keep_batch_dim=True, return_log=False, return_msssim=False, ensemble_kernel=True
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d, **kwargs).tolist()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).tolist()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

        kwargs = dict(
            window_size=5, L=1, keep_batch_dim=True, return_log=True, return_msssim=False, ensemble_kernel=True
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d, **kwargs).tolist()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).tolist()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

        kwargs = dict(
            window_size=5, L=1, keep_batch_dim=True, return_log=False, return_msssim=True, ensemble_kernel=True
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d, **kwargs).tolist()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).tolist()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

        kwargs = dict(
            window_size=5, L=1, keep_batch_dim=True, return_log=False, return_msssim=True, ensemble_kernel=False
        )
        old_version_ssim_score = old_version.ssim(self.x_2d, self.y_2d, **kwargs).tolist()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, data_dim=2, **kwargs).tolist()
        self.assertEqual(our_ssim_score, old_version_ssim_score)

    def test_ssim2d_with_pohsunsu_method(self):
        # https://github.com/Po-Hsun-Su/pytorch-ssim
        po_hsun_su_ssim_score = po_hsun_su_ssim.ssim(self.x_2d, self.y_2d, window_size=11).item()
        # use the settings of https://github.com/Po-Hsun-Su/pytorch-ssim
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, L=1, ensemble_kernel=True, data_dim=2, window_size=11).item()
        # 由于计算顺序的差异，导致存在一定的误差
        self.assertAlmostEqual(our_ssim_score, po_hsun_su_ssim_score)

        # https://github.com/Po-Hsun-Su/pytorch-ssim
        po_hsun_su_ssim_score = po_hsun_su_ssim.ssim(self.x_2d, self.y_2d, window_size=5).item()
        # use the settings of https://github.com/Po-Hsun-Su/pytorch-ssim
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, L=1, ensemble_kernel=True, data_dim=2, window_size=5).item()
        # 由于计算顺序的差异，导致存在一定的误差
        self.assertAlmostEqual(our_ssim_score, po_hsun_su_ssim_score)

    def test_ssim2d_with_vainf_method(self):
        """https://github.com/VainF/pytorch-msssim

        VainF的方法中，最后先计算了空间上的均值，之后再平均其他维度，这与我们方法中直接整体平均的计算结果存在差异
        """
        vainf_ssim_score = vainf_ssim.ssim(self.x_2d, self.y_2d, data_range=1, win_size=11).item()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, L=1, ensemble_kernel=False, padding=0, window_size=11).item()
        self.assertAlmostEqual(our_ssim_score, vainf_ssim_score)

        vainf_ssim_score = vainf_ssim.ssim(self.x_2d, self.y_2d, data_range=1, win_size=5).item()
        our_ssim_score = ssim.ssim(self.x_2d, self.y_2d, L=1, ensemble_kernel=False, padding=0, window_size=5).item()
        self.assertAlmostEqual(our_ssim_score, vainf_ssim_score)

    def test_msssim2d_with_vainf_method(self):
        """https://github.com/VainF/pytorch-msssim

        VainF的方法中，最后先计算了空间上的均值，之后再平均其他维度，这与我们方法中直接整体平均的计算结果存在差异
        """
        vainf_ssim_score = vainf_ssim.ms_ssim(self.x_2d, self.y_2d, data_range=1, win_size=11).item()
        our_ssim_score = ssim.ssim(
            self.x_2d, self.y_2d, return_msssim=True, L=1, ensemble_kernel=False, window_size=11, padding=0
        ).item()
        self.assertAlmostEqual(our_ssim_score, vainf_ssim_score)

        vainf_ssim_score = vainf_ssim.ms_ssim(self.x_2d, self.y_2d, data_range=1, win_size=5).item()
        our_ssim_score = ssim.ssim(
            self.x_2d, self.y_2d, return_msssim=True, L=1, ensemble_kernel=False, window_size=5, padding=0
        ).item()
        self.assertAlmostEqual(our_ssim_score, vainf_ssim_score)


if __name__ == "__main__":
    unittest.main()
