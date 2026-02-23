"""
pipeline/preprocess.py
Image Preprocessing Pipeline.

Handles resizing, normalization, noise reduction, and adaptive contrast
enhancement (CLAHE) for robust pothole detection under varying conditions.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Configuration for the image preprocessing pipeline."""
    target_size: Optional[Tuple[int, int]] = None  # (width, height); None = keep original
    normalize: bool = False                         # Normalize pixel values to [0, 1]
    denoise: bool = True                            # Apply bilateral denoising
    denoise_d: int = 9                              # Bilateral filter diameter
    denoise_sigma_color: float = 75.0
    denoise_sigma_space: float = 75.0
    clahe: bool = True                              # Adaptive contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    gamma_correction: bool = False                  # Manual gamma correction
    gamma_value: float = 1.0                        # < 1 brightens, > 1 darkens


class ImagePreprocessor:
    """
    Configurable image preprocessing pipeline for pothole detection.

    Applies a sequence of operations:
        1. Resize (optional)
        2. Bilateral denoising
        3. CLAHE contrast enhancement (per-channel in LAB space)
        4. Gamma correction (optional)
        5. Normalization (optional, for model input)

    Args:
        config: PreprocessConfig with all parameters.
    """

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_grid_size,
        )

    # ── Core Processing Steps ────────────────────────────────────

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target dimensions if configured."""
        if self.config.target_size is not None:
            return cv2.resize(
                image, self.config.target_size, interpolation=cv2.INTER_LINEAR
            )
        return image

    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for noise reduction.
        Preserves edges while smoothing noise — ideal for road surfaces.
        """
        return cv2.bilateralFilter(
            image,
            d=self.config.denoise_d,
            sigmaColor=self.config.denoise_sigma_color,
            sigmaSpace=self.config.denoise_sigma_space,
        )

    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Operates in LAB color space on the L (lightness) channel to
        enhance contrast adaptively — critical for low-light road scenes.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_enhanced = self._clahe.apply(l_channel)
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    def apply_gamma(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction for brightness adjustment."""
        inv_gamma = 1.0 / self.config.gamma_value
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
        ).astype(np.uint8)
        return cv2.LUT(image, table)

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize pixel values to [0, 1] float32."""
        return image.astype(np.float32) / 255.0

    # ── Full Pipeline ────────────────────────────────────────────

    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Run the full preprocessing pipeline on an image.

        Args:
            image: Input BGR image (uint8).

        Returns:
            Preprocessed image (uint8 or float32 if normalized).
        """
        result = image.copy()

        # 1. Resize
        result = self.resize(result)

        # 2. Denoise
        if self.config.denoise:
            result = self.denoise(result)

        # 3. Adaptive contrast enhancement
        if self.config.clahe:
            result = self.apply_clahe(result)

        # 4. Gamma correction
        if self.config.gamma_correction:
            result = self.apply_gamma(result)

        # 5. Normalize
        if self.config.normalize:
            result = self.normalize_image(result)

        return result

    # ── Convenience Factory ──────────────────────────────────────

    @classmethod
    def default(cls) -> "ImagePreprocessor":
        """Create preprocessor with sensible defaults for pothole detection."""
        return cls(PreprocessConfig(
            denoise=True,
            clahe=True,
            normalize=False,
        ))

    @classmethod
    def low_light(cls) -> "ImagePreprocessor":
        """Create preprocessor optimized for low-light conditions."""
        return cls(PreprocessConfig(
            denoise=True,
            clahe=True,
            clahe_clip_limit=3.5,
            gamma_correction=True,
            gamma_value=0.6,  # Brighten
        ))

    def __repr__(self) -> str:
        return f"ImagePreprocessor(config={self.config})"
