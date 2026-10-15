"""Stage 1: train the GNN on KTH-TIPS texture classification (paper Fig. 3 and Fig. 4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from . import plots
from .data import TextureGraphs, load_kth_tips
from .gnn import TextureGCN


def make_loaders(dataset: TextureGraphs, batch_size: int, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Shuffled training batches, plus the training and test sets for evaluation."""
    generator = torch.Generator().manual_seed(seed)
    train_graphs = [dataset.graphs[i] for i in dataset.train_idx]
    train = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, generator=generator)
    train_eval = DataLoader(train_graphs, batch_size=256)
    test = DataLoader([dataset.graphs[i] for i in dataset.test_idx], batch_size=256)
    return train, train_eval, test


@torch.no_grad()
def evaluate(model: TextureGCN, loader: DataLoader) -> tuple[float, float]:
    """Mean cross-entropy and accuracy over a loader."""
    model.eval()
    loss = correct = count = 0
    for batch in loader:
        logits = model(batch)
        loss += F.cross_entropy(logits, batch.y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == batch.y).sum().item()
        count += batch.num_graphs
    return loss / count, correct / count


def train_epoch(model: TextureGCN, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """One pass over the training batches; returns the running mean of the batch losses."""
    model.train()
    loss_sum = count = 0
    for batch in loader:
        loss = F.cross_entropy(model(batch), batch.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * batch.num_graphs
        count += batch.num_graphs
    return loss_sum / count


def run(
    data_dir: str | Path,
    out_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    features: str = "lbp",
    seed: int = 0,
    dataset: TextureGraphs | None = None,
) -> dict:
    """Train and evaluate the GNN; `dataset` skips reloading KTH-TIPS (it must use `features`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    if dataset is None:
        dataset = load_kth_tips(data_dir, features=features)
    train_loader, train_eval_loader, test_loader = make_loaders(dataset, batch_size, seed)
    model = TextureGCN(dataset.num_features, hidden_dim, hidden_dim, len(dataset.classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train and test loss are both measured with the model at the end of the epoch
    history = {"train_loss": [], "train_loss_running": [], "test_loss": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        running = train_epoch(model, train_loader, optimizer)
        train_loss, _ = evaluate(model, train_eval_loader)
        test_loss, test_acc = evaluate(model, test_loader)
        history["train_loss"].append(train_loss)
        history["train_loss_running"].append(running)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"[gnn] epoch {epoch:2d}  train loss {train_loss:.3f}  test loss {test_loss:.3f}  "
              f"test acc {test_acc:.3f}")

    history["config"] = dict(epochs=epochs, batch_size=batch_size, lr=lr, hidden_dim=hidden_dim,
                             features=features, seed=seed, n_train=len(dataset.train_idx),
                             n_test=len(dataset.test_idx))
    (out_dir / "gnn_metrics.json").write_text(json.dumps(history, indent=2))
    torch.save(model.state_dict(), out_dir / "gnn.pt")
    plots.loss_curve(history["train_loss"], "Training Loss Over Epochs", out_dir / "fig3_gnn_train_loss.png",
                     "fig3_train_loss")
    plots.loss_curve(history["test_loss"], "Test Loss Over Epochs", out_dir / "fig4_gnn_test_loss.png",
                     "fig4_test_loss")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/kth_tips/KTH_TIPS")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--features", default="lbp", choices=["lbp+gray", "lbp", "gray"])
    parser.add_argument("--seed", type=int, default=0)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
