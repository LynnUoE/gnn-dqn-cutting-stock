"""Figures 3-6 of the paper, plus the utilisation learning curve."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
REFERENCE = "#8a8983"

# Curves of the paper, read off its figures by eye (approximate, for comparison only).
PAPER = {
    "fig3_train_loss": [2.60, 2.42, 2.30, 2.18, 2.02, 1.85, 1.70, 1.65, 1.55, 1.51,
                        1.48, 1.40, 1.35, 1.30, 1.25, 1.30, 1.18, 1.20, 1.10, 1.17],
    "fig4_test_loss": [2.42, 2.30, 2.25, 2.13, 1.98, 1.87, 1.77, 1.75, 1.82, 1.71,
                       1.65, 1.71, 1.58, 1.57, 1.74, 1.52, 1.54, 1.56, 1.58, 1.41],
    "fig5_combined_loss": [2.45, 2.28, 2.21, 2.12, 1.92, 1.87, 1.77, 1.75, 1.67, 1.59,
                           1.55, 1.48, 1.45, 1.42, 1.42, 1.32, 1.48, 1.39, 1.22, 1.29],
    "fig6_epsilon": [1.00, 0.95, 0.89, 0.85, 0.80, 0.77, 0.73, 0.70, 0.68, 0.63],
}


def _axes(ax, title: str, xlabel: str, ylabel: str, title_pad: float = 10):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=title_pad)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _line(ax, x, y, color, label=None, **kwargs):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, solid_capstyle="round", solid_joinstyle="round", label=label, **kwargs)


def _legend(ax, **kwargs):
    ax.legend(frameon=False, labelcolor=INK_MUTED, **kwargs)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def loss_curve(values: Sequence[float], title: str, path: Path, paper_key: str | None = None):
    """Loss per epoch; optionally overlays the corresponding curve of the paper."""
    epochs = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    _axes(ax, title, "Epoch", "Loss")
    reference = PAPER.get(paper_key or "")
    if reference and len(reference) == len(values):
        ax.plot(epochs, reference, color=REFERENCE, linewidth=1.5, label="Paper (read off the figure)")
        _line(ax, epochs, values, BLUE, label="Reimplementation")
        _legend(ax, loc="upper right")
    else:
        _line(ax, epochs, values, BLUE)
    ax.set_xticks(epochs)
    _save(fig, path)


def reward_and_epsilon(rewards: Sequence[float], epsilons: Sequence[float], path: Path):
    """Fig. 6. Reward and epsilon share the episode axis but get one panel each.

    With the paper's 100 episodes every 10th episode is marked, as in the
    paper; other run lengths are summarised by the mean of each tenth of the run.
    """
    rewards, epsilons = np.asarray(rewards), np.asarray(epsilons)
    n = len(rewards)
    episodes = np.arange(n)
    paper_protocol = n == 100  # the run length of the paper
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6), sharex=True, facecolor=SURFACE,
                                      gridspec_kw={"height_ratios": [3, 2], "hspace": 0.28})
    _axes(top, "Total reward per episode", "", "Total reward", title_pad=30)
    top.plot(episodes, rewards, color=BLUE, linewidth=1, alpha=0.35, label="Every episode")
    if paper_protocol:
        marks = episodes[::10]
        _line(top, marks, rewards[marks], BLUE, label="Every 10th episode (as plotted in the paper)")
    elif n >= 10:
        edges = np.linspace(0, n, 11).astype(int)
        means = [rewards[a:b].mean() for a, b in zip(edges[:-1], edges[1:])]
        _line(top, (edges[:-1] + edges[1:] - 1) / 2, means, BLUE, label="Mean of each tenth of the run")
    _legend(top, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2, borderaxespad=0.2, handlelength=1.6)

    _axes(bottom, "Epsilon", "Episode", "Epsilon")
    if paper_protocol:
        bottom.plot(marks, PAPER["fig6_epsilon"], color=REFERENCE, linewidth=1.5, label="Paper (read off the figure)")
        _line(bottom, marks, epsilons[marks], ORANGE, label="Reimplementation")
        _legend(bottom, loc="upper right")
    else:
        bottom.plot(episodes, epsilons, color=ORANGE, linewidth=2)
    _save(fig, path)


def utilization_curve(series: dict[str, Sequence[float]], baselines: dict[str, float], episodes_per_epoch: int,
                      path: Path):
    """Greedy-policy utilisation after each epoch against the non-learning baselines."""
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    _axes(ax, "Utilisation of the greedy policy on held-out materials", "Training episodes", "Utilisation (%)",
          title_pad=14 + 16 * len(series))
    for name, value in baselines.items():
        ax.axhline(100 * value, color=REFERENCE, linewidth=1.5)
        ax.annotate(f"{name} {100 * value:.0f}%", (1.015, 100 * value), xycoords=("axes fraction", "data"),
                    va="center", color=INK_MUTED, fontsize=9)
    for color, (name, values) in zip((BLUE, ORANGE, AQUA), series.items()):
        x = np.arange(1, len(values) + 1) * episodes_per_epoch
        _line(ax, x, 100 * np.asarray(values), color, label=f"{name}: {100 * values[-1]:.0f}% after {x[-1]:,} episodes")
    _legend(ax, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2, handlelength=1.6)
    ax.set_ylim(0, 100)
    _save(fig, path)
