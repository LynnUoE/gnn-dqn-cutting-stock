"""Stage 2: train the GNN and the DQN together (paper Fig. 5 and Fig. 6).

Every epoch consists of
  1. one pass over the training graphs minimising the classification loss of
     the GNN, after which the embedding v_G of every material is refreshed;
  2. `episodes_per_epoch` cutting episodes with an epsilon-greedy policy on
     states s = [v_G, f_C]; after each environment step the Q-network takes one
     gradient step on the TD loss of Eq. (6) over a replay mini-batch.
The GNN output enters the DQN as data: the TD loss does not back-propagate
into the GNN. The combined loss of an epoch is the sum of the two mean losses.

`--mask-invalid` and `--double-dqn` are extensions that are not in the paper.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from . import plots
from .data import TextureGraphs, load_kth_tips
from .dqn import QNetwork, ReplayBuffer, td_loss
from .env import CuttingStockEnv
from .gnn import TextureGCN
from .train_gnn import evaluate, make_loaders, train_epoch


@torch.no_grad()
def embed_all(gnn: TextureGCN, dataset: TextureGraphs) -> torch.Tensor:
    """v_G for every image of the dataset, shape (n_graphs, embed_dim)."""
    gnn.eval()
    loader = DataLoader(dataset.graphs, batch_size=256)
    return torch.cat([gnn.encode(b.x, b.edge_index, b.batch) for b in loader])


def evaluate_policy(policy: str, dataset: TextureGraphs, graph_ids: np.ndarray, episodes: int, seed: int,
                    q_net: QNetwork | None = None, v_g: torch.Tensor | None = None, mask_invalid: bool = False,
                    **env_kwargs) -> dict:
    """Roll out a policy on a fixed set of orders.

    Policies: 'dqn' (greedy), 'random', 'random_valid' (uniform over feasible
    cuts) and 'first_fit'.
    """
    env = CuttingStockEnv(seed=seed, **env_kwargs)
    rng = np.random.default_rng(seed)
    labels = dataset.labels()
    utilization, returns, invalid, steps = [], [], 0, 0
    for k in range(episodes):
        graph_id = int(graph_ids[k % len(graph_ids)])
        f_c = env.reset(dataset.classes[labels[graph_id]])
        done, total = False, 0.0
        while not done:
            if policy == "dqn":
                action = q_net.greedy_action(v_g[graph_id], f_c, env.valid_actions() if mask_invalid else None)
            elif policy == "first_fit":
                action = env.first_fit_action()
            elif policy == "random_valid":
                action = int(rng.choice(np.flatnonzero(env.valid_actions())))
            else:
                action = int(rng.integers(env.n_actions))
            f_c, reward, done, info = env.step(action)
            total += reward
            invalid += not info["valid"]
            steps += 1
        utilization.append(info["utilization"])
        returns.append(total)
    return {"utilization": float(np.mean(utilization)), "return": float(np.mean(returns)),
            "invalid_rate": invalid / steps}


def run(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 20,
    episodes_per_epoch: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    features: str = "lbp",
    gamma: float = 0.99,
    buffer_size: int = 10_000,
    target_sync: int = 1_000,
    eps_start: float = 1.0,
    eps_decay: float = 0.995,
    eps_min: float = 0.01,
    mask_invalid: bool = False,
    double_dqn: bool = False,
    grain_constraint: bool = True,
    eval_episodes: int = 200,
    seed: int = 0,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dataset = load_kth_tips(data_dir, features=features, seed=seed)
    labels = dataset.labels()
    train_loader, test_loader = make_loaders(dataset, batch_size, seed)
    env_kwargs = dict(grain_constraint=grain_constraint)
    env = CuttingStockEnv(seed=seed, **env_kwargs)

    gnn = TextureGCN(dataset.num_features, hidden_dim, hidden_dim, len(dataset.classes))
    q_net = QNetwork(gnn.embed_dim, env.feature_dim, env.n_actions)
    target_net = copy.deepcopy(q_net)
    gnn_optimizer = torch.optim.Adam(gnn.parameters(), lr=lr)
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    buffer = ReplayBuffer(buffer_size, env.feature_dim, env.n_actions, seed=seed)

    def evaluate_baseline(policy: str) -> dict:
        return evaluate_policy(policy, dataset, dataset.test_idx, eval_episodes, seed + 1, **env_kwargs)

    def evaluate_greedy() -> dict:
        return evaluate_policy("dqn", dataset, dataset.test_idx, eval_episodes, seed + 1, q_net, v_g,
                               mask_invalid, **env_kwargs)

    history = {"combined_loss": [], "ce_loss": [], "td_loss": [], "test_loss": [], "test_acc": [],
               "mean_max_q": [], "greedy_utilization": [], "episode_reward": [], "episode_utilization": [],
               "episode_epsilon": []}
    epsilon, env_steps = eps_start, 0
    for epoch in range(1, epochs + 1):
        # 1. GNN: texture classification, then refresh the material embeddings
        ce = train_epoch(gnn, train_loader, gnn_optimizer)
        test_loss, test_acc = evaluate(gnn, test_loader)
        v_g = embed_all(gnn, dataset)

        # 2. DQN: interact with the cutting environment
        td_losses, q_values = [], []
        for _ in range(episodes_per_epoch):
            graph_id = int(rng.choice(dataset.train_idx))
            f_c = env.reset(dataset.classes[labels[graph_id]])
            mask = env.valid_actions() if mask_invalid else None
            done, total = False, 0.0
            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(env.n_actions) if mask is None else rng.choice(np.flatnonzero(mask)))
                else:
                    action = q_net.greedy_action(v_g[graph_id], f_c, mask)
                next_f_c, reward, done, info = env.step(action)
                mask = env.valid_actions() if mask_invalid else None
                buffer.push(graph_id, f_c, action, reward, next_f_c, done, mask)
                f_c, total = next_f_c, total + reward
                env_steps += 1
                if len(buffer) >= batch_size:
                    batch = buffer.sample(batch_size)
                    loss = td_loss(q_net, target_net, v_g, batch, gamma, double_dqn)
                    q_optimizer.zero_grad()
                    loss.backward()
                    q_optimizer.step()
                    td_losses.append(loss.item())
                    if len(td_losses) % 50 == 1:
                        with torch.no_grad():
                            q_values.append(q_net(v_g[batch.graph_id], batch.f_c).max(dim=1).values.mean().item())
                if env_steps % target_sync == 0:
                    target_net.load_state_dict(q_net.state_dict())
            history["episode_reward"].append(total)
            history["episode_utilization"].append(info["utilization"])
            history["episode_epsilon"].append(epsilon)
            epsilon = max(eps_min, epsilon * eps_decay)

        td = float(np.mean(td_losses)) if td_losses else 0.0
        greedy = evaluate_greedy()
        history["ce_loss"].append(ce)
        history["td_loss"].append(td)
        history["combined_loss"].append(ce + td)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["mean_max_q"].append(float(np.mean(q_values)) if q_values else 0.0)
        history["greedy_utilization"].append(greedy["utilization"])
        recent = history["episode_reward"][-episodes_per_epoch:]
        print(f"[combined] epoch {epoch:2d}  loss {ce + td:.3f} (ce {ce:.3f} + td {td:.4f})  "
              f"reward {np.mean(recent):+.3f}  eps {epsilon:.3f}  max-Q {history['mean_max_q'][-1]:+.2f}  "
              f"greedy util {greedy['utilization']:.3f}")

    final = {"dqn_greedy": evaluate_greedy(), "first_fit": evaluate_baseline("first_fit"),
             "random_valid": evaluate_baseline("random_valid"), "random": evaluate_baseline("random")}
    history["final_eval"] = final
    history["baselines"] = {"First-fit heuristic": final["first_fit"]["utilization"],
                            "Random feasible cut": final["random_valid"]["utilization"],
                            "Random cut": final["random"]["utilization"]}
    history["config"] = dict(epochs=epochs, episodes_per_epoch=episodes_per_epoch, batch_size=batch_size, lr=lr,
                             hidden_dim=hidden_dim, features=features, gamma=gamma, buffer_size=buffer_size,
                             target_sync=target_sync, eps_start=eps_start, eps_decay=eps_decay, eps_min=eps_min,
                             mask_invalid=mask_invalid, double_dqn=double_dqn, grain_constraint=grain_constraint,
                             eval_episodes=eval_episodes, seed=seed, env_steps=env_steps)
    (out_dir / "combined_metrics.json").write_text(json.dumps(history))
    torch.save({"gnn": gnn.state_dict(), "q_net": q_net.state_dict()}, out_dir / "gnn_dqn.pt")

    plots.loss_curve(history["combined_loss"], "Training Loss Over Epochs (combine)",
                     out_dir / "fig5_combined_loss.png", "fig5_combined_loss")
    plots.reward_and_epsilon(history["episode_reward"], history["episode_epsilon"],
                             out_dir / "fig6_reward_epsilon.png")
    plots.utilization_curve({"GNN-DQN": history["greedy_utilization"]}, history["baselines"], episodes_per_epoch,
                            out_dir / "utilization_curve.png")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/kth_tips/KTH_TIPS")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--features", default="lbp", choices=["lbp+gray", "lbp", "gray"])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer-size", type=int, default=10_000)
    parser.add_argument("--target-sync", type=int, default=1_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-decay", type=float, default=0.995)
    parser.add_argument("--eps-min", type=float, default=0.01)
    parser.add_argument("--mask-invalid", action="store_true", help="restrict actions to feasible cuts (extension)")
    parser.add_argument("--double-dqn", action="store_true", help="Double DQN target (extension)")
    parser.add_argument("--no-grain-constraint", dest="grain_constraint", action="store_false")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
