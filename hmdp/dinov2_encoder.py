#!/usr/bin/env python3
"""
Unified DINOv2 Proxy Encoder Module.

Provides both full-image and per-region encoding capabilities for the
H-MDP blind-but-smart pipeline. All encoders use frozen DINOv2-ViT-L/14
and run client-side, ensuring raw pixels never leave the device.

Classes:
    DINOv2Encoder:        Full-image encoding → (1, 1024)
    DINOv2RegionEncoder:  Per-region grid encoding → (M, 1024)

Usage:
    from hmdp.dinov2_encoder import DINOv2RegionEncoder

    encoder = DINOv2RegionEncoder(device="cuda", grid=5)
    encoder.load()
    features = encoder.extract_regions(image)  # (25, 1024)
    encoder.unload()
"""

import gc
import time
from typing import List, Optional, Tuple

import torch
from PIL import Image

from hmdp.constants import GridConfig, ModelDims


# ═══════════════════════════════════════════════════════════════════════
#  Base Class: Common DINOv2 Functionality
# ═══════════════════════════════════════════════════════════════════════

class DINOv2Base:
    """Base class for DINOv2 encoders with common load/unload logic."""

    MODEL_ID: str = "facebook/dinov2-large"
    OUTPUT_DIM: int = ModelDims.DINOV2_OUTPUT  # 1024 for ViT-L/14

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model: Optional[torch.nn.Module] = None
        self.processor: Optional[object] = None
        self._load_time: float = 0.0

    def load(self, verbose: bool = True) -> None:
        """Load frozen DINOv2 model and processor."""
        from transformers import AutoImageProcessor, AutoModel

        if verbose:
            print(f"  Loading DINOv2: {self.MODEL_ID}...")
        t0 = time.time()

        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = AutoModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.model.eval()

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        self._load_time = time.time() - t0
        if verbose:
            print(f"  DINOv2 loaded in {self._load_time:.1f}s")

    def unload(self) -> None:
        """Free GPU memory and clear references."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()
        gc.collect()

    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self.model is not None and self.processor is not None


# ═══════════════════════════════════════════════════════════════════════
#  Full-Image Encoder
# ═══════════════════════════════════════════════════════════════════════

class DINOv2Encoder(DINOv2Base):
    """
    Full-image DINOv2 encoder.

    Encodes entire screenshots into a single 1024-dim CLS feature.
    Used for baseline analysis and non-grid feature extraction.

    Example:
        >>> encoder = DINOv2Encoder(device="cuda")
        >>> encoder.load()
        >>> features = encoder.extract(image)  # (1, 1024)
        >>> encoder.unload()
    """

    @torch.no_grad()
    def extract(self, image: Image.Image) -> torch.Tensor:
        """
        Extract CLS token features from full image.

        Args:
            image: PIL Image (any size)

        Returns:
            Tensor of shape (1, 1024) — DINOv2 CLS token
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token: (1, 1024)


# ═══════════════════════════════════════════════════════════════════════
#  Region (Grid) Encoder
# ═══════════════════════════════════════════════════════════════════════

class DINOv2RegionEncoder(DINOv2Base):
    """
    Per-region DINOv2 encoder with grid partitioning.

    Partitions screenshots into M=grid×grid regions and encodes each
    independently. This is the core encoder for H-MDP's LDP pipeline,
    producing per-region latent features φ(R_t^(i)) for Eq. 3-4.

    Args:
        device: Compute device ("cuda" or "cpu")
        grid: Grid size (default 5 for 5×5=25 regions)

    Example:
        >>> encoder = DINOv2RegionEncoder(device="cuda", grid=5)
        >>> encoder.load()
        >>> regions = encoder.extract_regions(image)  # (25, 1024)
        >>> encoder.unload()
    """

    def __init__(self, device: str = "cuda", grid: int = GridConfig.SIZE):
        super().__init__(device)
        self.grid = grid
        self.num_regions = GridConfig.get_num_regions(grid)

    @staticmethod
    def partition(image: Image.Image, grid: int = GridConfig.SIZE) -> List[Image.Image]:
        """
        Partition image into uniform grid×grid regions (row-major order).

        Args:
            image: PIL Image
            grid: Grid size (e.g., 5 for 5×5)

        Returns:
            List of M=grid² cropped PIL Images
        """
        w, h = image.size
        gw, gh = w / grid, h / grid
        regions = []
        for row in range(grid):
            for col in range(grid):
                box = (
                    int(col * gw),
                    int(row * gh),
                    int((col + 1) * gw),
                    int((row + 1) * gh),
                )
                regions.append(image.crop(box))
        return regions

    @torch.no_grad()
    def extract_regions(self, image: Image.Image) -> torch.Tensor:
        """
        Extract per-region DINOv2 CLS features.

        Args:
            image: PIL Image (any size)

        Returns:
            Tensor of shape (M, 1024) where M = grid×grid
            Each row is the CLS token for region i (row-major order)

        Note:
            This implements Eq. 3's E(R_t^(i)) — the per-region encoding
            before projection to latent space φ.
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        regions = self.partition(image, self.grid)
        inputs = self.processor(images=regions, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # Return CLS tokens for all M regions: (M, 1024)
        return outputs.last_hidden_state[:, 0, :]

    def get_region_bounds(self, image_size: Tuple[int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Get pixel bounds for each region without cropping.

        Useful for visualization and coordinate mapping.

        Args:
            image_size: (width, height) of original image

        Returns:
            List of M bounding boxes (left, top, right, bottom)
        """
        w, h = image_size
        gw, gh = w / self.grid, h / self.grid
        bounds = []
        for row in range(self.grid):
            for col in range(self.grid):
                box = (
                    int(col * gw),
                    int(row * gh),
                    int((col + 1) * gw),
                    int((row + 1) * gh),
                )
                bounds.append(box)
        return bounds


# ═══════════════════════════════════════════════════════════════════════
#  Unified Encoder (Convenience Wrapper)
# ═══════════════════════════════════════════════════════════════════════

class DINOv2UnifiedEncoder:
    """
    Convenience wrapper managing both full-image and per-region encoders.

    This is the recommended entry point for HMDPAgent and similar classes
    that need both encoding modes.

    Example:
        >>> encoder = DINOv2UnifiedEncoder(device="cuda", grid=5)
        >>> encoder.load()
        >>> full_feat = encoder.extract(image)           # (1, 1024)
        >>> region_feats = encoder.extract_regions(image)  # (25, 1024)
        >>> encoder.unload()
    """

    def __init__(self, device: str = "cuda", grid: int = GridConfig.SIZE):
        self.full_encoder = DINOv2Encoder(device)
        self.region_encoder = DINOv2RegionEncoder(device, grid)

    def load(self, verbose: bool = True) -> None:
        """Load both encoders."""
        self.full_encoder.load(verbose=verbose)
        self.region_encoder.load(verbose=False)  # Suppress duplicate message

    def unload(self) -> None:
        """Unload both encoders."""
        self.full_encoder.unload()
        self.region_encoder.unload()

    def extract(self, image: Image.Image) -> torch.Tensor:
        """Full-image encoding."""
        return self.full_encoder.extract(image)

    def extract_regions(self, image: Image.Image) -> torch.Tensor:
        """Per-region encoding."""
        return self.region_encoder.extract_regions(image)

    @property
    def grid(self) -> int:
        return self.region_encoder.grid

    @property
    def num_regions(self) -> int:
        return self.region_encoder.num_regions
