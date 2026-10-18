"""Monocular relative-depth estimation used by the screen based environment.

MiDaS produces relative inverse depth: brighter pixels are generally nearer.
It does not recover metric distances in meters.
"""

import cv2
import numpy as np
import torch


class RelativeDepthEstimator:
    def __init__(self, device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # First use downloads the official MiDaS_small weights through torch.hub.
        try:
            self.model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
        except ModuleNotFoundError as error:
            if error.name == "timm":
                raise RuntimeError(
                    "MiDaS requires the 'timm' package. Install project depth dependencies "
                    "with: .\\.venv\\Scripts\\python.exe -m pip install -r requirements-depth.txt"
                ) from error
            raise
        self.model.to(self.device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform

    @torch.inference_mode()
    def __call__(self, rgb):
        image = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
        batch = self.transform(image).to(self.device)
        prediction = self.model(batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False
        ).squeeze().float().cpu().numpy()
        # Per-frame robust normalization preserves near/far ordering and avoids outliers.
        low, high = np.percentile(depth, (2, 98))
        if high <= low:
            return np.zeros(rgb.shape[:2], dtype=np.uint8)
        depth = np.clip((depth - low) / (high - low), 0.0, 1.0)
        return np.round(depth * 255).astype(np.uint8)
