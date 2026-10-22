# Copyright 2026 The sen-bridge authors.
# SPDX-License-Identifier: Apache-2.0
"""FID, LPIPS, PSNR and SSIM with the settings SEN-Bridge specifies.

Import the framework you use; this package imports neither:

    from sen_bridge import torch as sbt  # [B, 3, H, W] from sen_bridge import jax
    as sbj  # [H, W, 3] per image, [B, H, W, 3] for FID.update
"""

__version__ = "0.1.0"
