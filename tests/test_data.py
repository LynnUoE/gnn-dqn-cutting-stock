import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from csp_gnn_dqn.data import GRID, LBP_BINS, grid_edge_index, lbp_uniform, load_kth_tips, node_features


class LBPTest(unittest.TestCase):
    def test_flat_image_is_one_uniform_pattern(self):
        codes = lbp_uniform(np.full((16, 16), 0.5, dtype=np.float32))
        self.assertTrue((codes == 8).all())

    def test_codes_are_in_range(self):
        rng = np.random.default_rng(0)
        codes = lbp_uniform(rng.random((64, 64), dtype=np.float32))
        self.assertGreaterEqual(codes.min(), 0)
        self.assertLessEqual(codes.max(), LBP_BINS - 1)

    def test_isolated_bright_pixel_is_pattern_zero(self):
        img = np.zeros((9, 9), dtype=np.float32)
        img[4, 4] = 1.0
        self.assertEqual(lbp_uniform(img)[4, 4], 0)

    def test_node_histograms_sum_to_one(self):
        rng = np.random.default_rng(0)
        feats = node_features(rng.random((128, 128), dtype=np.float32), "lbp")
        self.assertEqual(feats.shape, (GRID * GRID, LBP_BINS))
        np.testing.assert_allclose(feats.sum(axis=1), 1.0, rtol=1e-6)


class GraphTest(unittest.TestCase):
    def test_grid_edges(self):
        edges = grid_edge_index().numpy()
        self.assertEqual(edges.shape, (2, 2 * 2 * GRID * (GRID - 1)))
        pairs = set(map(tuple, edges.T))
        self.assertTrue(all((b, a) in pairs for a, b in pairs))
        self.assertTrue(all(abs(a - b) in (1, GRID) for a, b in pairs))

    def test_split_holds_out_whole_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("cotton", "sponge"):
                (Path(tmp) / name).mkdir()
                for scale in range(1, 10):
                    for im in (1, 2):
                        Image.new("RGB", (40, 40), (scale * 20, 0, 0)).save(
                            Path(tmp) / name / f"0-scale_{scale}_im_{im}_col.png")
            dataset = load_kth_tips(tmp, test_scales=(3, 7))
        self.assertEqual(dataset.classes, ["cotton", "sponge"])
        self.assertEqual(len(dataset.test_idx), 2 * 2 * 2)
        self.assertEqual(len(dataset.train_idx), 2 * 7 * 2)
        self.assertFalse(set(dataset.train_idx) & set(dataset.test_idx))


if __name__ == "__main__":
    unittest.main()
