"""Run all experiments of the paper.

Fig. 3-6 are produced with the protocol of the paper (20 epochs, 100 episodes).
`--extended` goes beyond the paper with two 40 000-episode runs: the DQN exactly
as specified, and the same DQN with invalid-action masking and a Double-DQN
target.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import plots, train_combined, train_gnn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/kth_tips/KTH_TIPS")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extended", action="store_true", help="also run the two 40 000-episode experiments (about 10 minutes)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    train_gnn.run(args.data_dir, out_dir, seed=args.seed)
    train_combined.run(args.data_dir, out_dir, seed=args.seed)
    if args.extended:
        long_run = dict(episodes_per_epoch=2000, eps_min=0.05, seed=args.seed)
        paper = train_combined.run(args.data_dir, out_dir / "extended_paper_dqn", **long_run)
        masked = train_combined.run(args.data_dir, out_dir / "extended_masked_double_dqn", mask_invalid=True,
                                    double_dqn=True, **long_run)
        plots.utilization_curve(
            {"DQN as in the paper": paper["greedy_utilization"],
             "DQN + invalid-action masking + Double DQN": masked["greedy_utilization"]},
            masked["baselines"], long_run["episodes_per_epoch"], out_dir / "extended_utilization.png")


if __name__ == "__main__":
    main()
