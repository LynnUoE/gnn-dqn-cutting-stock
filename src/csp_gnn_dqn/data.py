"""KTH-TIPS preprocessing and graph construction (paper Sec. 2.1).

Each image becomes one graph: the 128x128 grayscale image is tiled into a
regular grid of patches, every patch is a node, and spatially adjacent patches
are connected. A node carries the LBP histogram of its patch; the flattened
grayscale pixels of the patch can be appended with `features="lbp+gray"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch_geometric.data import Data

IMAGE_SIZE = 128
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE  # 8x8 = 64 nodes per image
LBP_POINTS = 8
LBP_RADIUS = 1.0
LBP_BINS = LBP_POINTS + 2  # rotation-invariant uniform patterns
SCALE_PATTERN = re.compile(r"scale_(\d+)")


def load_gray(path: Path) -> np.ndarray:
    """Load an image as grayscale, clip it to a square and resize to 128x128."""
    img = Image.open(path).convert("L")
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def lbp_uniform(gray: np.ndarray, points: int = LBP_POINTS, radius: float = LBP_RADIUS) -> np.ndarray:
    """Rotation-invariant uniform LBP (Ojala et al., 2002); codes in [0, points + 1]."""
    pad = int(np.ceil(radius)) + 1
    padded = np.pad(gray, pad, mode="edge")
    h, w = gray.shape
    bits = np.empty((points, h, w), dtype=np.int8)
    for p in range(points):
        angle = 2.0 * np.pi * p / points
        dy = round(-radius * np.sin(angle), 9)
        dx = round(radius * np.cos(angle), 9)
        y0, x0 = int(np.floor(dy)), int(np.floor(dx))
        fy, fx = dy - y0, dx - x0
        neighbour = np.zeros_like(gray)
        # bilinear interpolation of the circular neighbour
        for oy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            for ox, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                if wy * wx > 0:
                    neighbour += wy * wx * padded[pad + oy : pad + oy + h, pad + ox : pad + ox + w]
        bits[p] = neighbour >= gray - 1e-6
    transitions = np.abs(bits - np.roll(bits, 1, axis=0)).sum(axis=0)
    return np.where(transitions <= 2, bits.sum(axis=0), points + 1)


def _to_patches(arr: np.ndarray) -> np.ndarray:
    """(128, 128) -> (64, 256): one row of flattened pixels per patch, row-major."""
    return (
        arr.reshape(GRID, PATCH_SIZE, GRID, PATCH_SIZE)
        .transpose(0, 2, 1, 3)
        .reshape(GRID * GRID, PATCH_SIZE * PATCH_SIZE)
    )


def node_features(gray: np.ndarray, mode: str = "lbp") -> np.ndarray:
    feats = []
    if "lbp" in mode:
        codes = _to_patches(lbp_uniform(gray))
        hist = np.stack([(codes == b).mean(axis=1) for b in range(LBP_BINS)], axis=1)
        feats.append(hist.astype(np.float32))
    if "gray" in mode:
        feats.append(_to_patches(gray))
    if not feats:
        raise ValueError(f"unknown node feature mode: {mode!r}")
    return np.concatenate(feats, axis=1)


def grid_edge_index(grid: int = GRID) -> torch.Tensor:
    """4-neighbour grid connectivity, both directions (self-loops are added by GCNConv)."""
    idx = np.arange(grid * grid).reshape(grid, grid)
    right = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()])
    down = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()])
    edges = np.concatenate([right, down], axis=1)
    edges = np.concatenate([edges, edges[::-1]], axis=1)
    return torch.from_numpy(edges).long()


@dataclass
class TextureGraphs:
    graphs: list[Data]
    classes: list[str]
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def num_features(self) -> int:
        return self.graphs[0].num_node_features

    def labels(self) -> np.ndarray:
        return np.array([int(g.y) for g in self.graphs])


def load_kth_tips(
    data_dir: str | Path,
    features: str = "lbp",
    test_scales: tuple[int, ...] = (3, 7),
) -> TextureGraphs:
    """Build one graph per image; the images taken at `test_scales` form the test set.

    Each KTH-TIPS class is a single physical sample photographed at 9 scales
    under 3 poses x 3 illuminations, so a random split would put near-identical
    photos into both sets. Holding out whole scales measures generalisation to
    scales that were not seen in training.
    """
    data_dir = Path(data_dir)
    classes = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if not classes:
        raise FileNotFoundError(f"no class folders found in {data_dir}")
    edge_index = grid_edge_index()
    graphs: list[Data] = []
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label, name in enumerate(classes):
        for path in sorted((data_dir / name).glob("*.png")):
            match = SCALE_PATTERN.search(path.name)
            if match is None:
                raise ValueError(f"cannot read the scale from {path.name}")
            (test_idx if int(match.group(1)) in test_scales else train_idx).append(len(graphs))
            x = torch.from_numpy(node_features(load_gray(path), features))
            graphs.append(Data(x=x, edge_index=edge_index, y=torch.tensor([label])))
    return TextureGraphs(graphs, classes, np.array(train_idx), np.array(test_idx))
