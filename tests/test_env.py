import unittest

import numpy as np

from csp_gnn_dqn.env import ORIENTATIONS, CuttingStockEnv


def brute_force_mask(env: CuttingStockEnv) -> np.ndarray:
    """Feasibility of every action, checked cell by cell."""
    w, h = env.pieces[env.cursor]
    mask = np.zeros(env.n_actions, dtype=bool)
    for action in range(env.n_actions):
        x, y, theta = env.decode(action)
        pw, ph = (w, h) if theta == 0 else (h, w)
        if theta == 90 and env.grained:
            continue
        if x + pw <= env.width and y + ph <= env.height:
            mask[action] = not env.occupied[y : y + ph, x : x + pw].any()
    return mask


class CuttingStockEnvTest(unittest.TestCase):
    def rollout(self, material: str, seed: int, policy):
        env = CuttingStockEnv(seed=seed)
        env.reset(material)
        rng = np.random.default_rng(seed)
        done = env.cursor >= len(env.pieces)
        while not done:
            yield env
            _, _, done, _ = env.step(policy(env, rng))

    def test_orders_tile_the_sheet(self):
        env = CuttingStockEnv(seed=0)
        for material in ["sponge", "linen"] * 50:
            env.reset(material)
            self.assertEqual(sum(w * h for w, h in env.pieces), env.area)
            self.assertTrue(all(max(w, h) <= env.max_piece_side for w, h in env.pieces))

    def test_valid_actions_match_brute_force(self):
        def policy(env, rng):
            return env.first_fit_action() if rng.random() < 0.5 else int(rng.integers(env.n_actions))

        for material in ("sponge", "corduroy"):
            for seed in range(30):
                for env in self.rollout(material, seed, policy):
                    np.testing.assert_array_equal(env.valid_actions(), brute_force_mask(env))

    def test_current_piece_always_fits_somewhere(self):
        for seed in range(30):
            for env in self.rollout("sponge", seed, lambda env, rng: int(rng.integers(env.n_actions))):
                self.assertTrue(env.valid_actions().any())

    def test_step_agrees_with_mask(self):
        for seed in range(30):
            env = CuttingStockEnv(seed=seed)
            env.reset("cotton")
            rng = np.random.default_rng(seed)
            done = False
            while not done:
                action = int(rng.integers(env.n_actions))
                expected = bool(env.valid_actions()[action])
                _, reward, done, info = env.step(action)
                self.assertEqual(info["valid"], expected)
                self.assertEqual(reward < 0, not expected)

    def test_grained_material_cannot_rotate(self):
        env = CuttingStockEnv(seed=0)
        env.reset("linen")
        mask = env.valid_actions().reshape(len(ORIENTATIONS), env.height, env.width)
        self.assertFalse(mask[1].any())

    def test_first_fit_is_always_valid(self):
        for seed in range(30):
            for env in self.rollout("sponge", seed, lambda env, rng: env.first_fit_action()):
                self.assertTrue(env.valid_actions()[env.first_fit_action()])

    def test_episode_return_is_utilisation_when_all_cuts_are_valid(self):
        env = CuttingStockEnv(seed=3)
        env.reset("sponge")
        total, done = 0.0, False
        while not done:
            _, reward, done, info = env.step(env.first_fit_action())
            total += reward
        self.assertAlmostEqual(total, env.utilization)

    def test_features_stay_in_unit_range(self):
        for seed in range(30):
            for env in self.rollout("sponge", seed, lambda env, rng: env.first_fit_action()):
                f = env.features()
                self.assertTrue(((f >= 0) & (f <= 1)).all())
            self.assertTrue(env.valid_actions().sum() == 0)


if __name__ == "__main__":
    unittest.main()
