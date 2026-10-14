# Sustainable Material Cutting Optimization Using Deep Q-Networks — Code

English | [简体中文](README.zh-CN.md)

Paper: Linxuan Chen, *Sustainable Material Cutting Optimization Using Deep Q-Networks: A Reinforcement Learning Approach for Resource Efficiency*, ITM Web of Conferences **73**, 01002 (2025), https://doi.org/10.1051/itmconf/20257301002 (PDF: `itmconf_iwadi2024_01002.pdf`).

The original code of the paper was lost; this repository re-implements it from the paper. Details the paper does not specify (graph construction, the cutting environment, the reward function and some hyperparameters) were decided anew for this rewrite, see [Details not specified in the paper](#details-not-specified-in-the-paper), so the numbers do not match the paper's figures exactly.

## Comparison with the paper

| Figure | Paper (read off the figure) | This code (seed 0) | 5 seeds | Comparison |
|---|---|---|---|---|
| Fig. 3 GNN training loss | 2.60 → 1.17 | 2.30 → 1.34 | final 1.39 ± 0.06 | same trend, slightly higher final value |
| Fig. 4 GNN test loss | 2.42 → 1.41 | 2.28 → 1.30 | final 1.40 ± 0.08 | consistent |
| Fig. 5 combined loss | 2.45 → 1.29 | 2.31 → 1.34 | final 1.39 ± 0.06 | consistent |
| Fig. 6 ε | 1.00 → 0.63 (episode 90) | 1.00 → 0.637 | independent of the seed | consistent (×0.995 per episode) |
| Fig. 6 total reward | 397–415, no clear trend | −0.16–0.42, no clear trend | mean 0.14 over the first 50 episodes, 0.12 over the last 50 | qualitatively consistent; values not comparable (different reward definition) |

The rewrite reproduces the four figures of the paper qualitatively. Two further points:

1. **With the paper's training budget (100 episodes, ε never below 0.61), the DQN has not yet learned a cutting policy.** After training, the greedy policy reaches 10% material utilisation (13 ± 3% over 5 seeds), below the 31% of cutting at random. The reward curve in the paper's Fig. 6 shows no upward trend either: at this budget the policy has not converged.
2. **Training longer does not help without changing the algorithm.** Trained for 40,000 episodes, the same DQN overestimates its Q-values badly (the mean maximum Q-value climbs to 18 while returns are bounded by 1) and ends at 3% utilisation. Only with invalid-action masking and a Double DQN target, neither of which is in the paper, does it reach 79%: above the 73% of random feasible cuts, but still below the 84% of a first-fit heuristic.

Both points depend on the cutting environment designed for this rewrite (the paper does not define one).

## Running

```bash
uv venv --python 3.12 .venv
```

```bash
uv pip install --python .venv/bin/python -e .
```

Dataset (KTH-TIPS, colour, 200×200, about 68 MB):

```bash
mkdir -p data/kth_tips && curl -L -o data/kth_tips/kth_tips_col_200x200.tar https://www.csc.kth.se/cvap/databases/kth-tips/kth_tips_col_200x200.tar && tar -xf data/kth_tips/kth_tips_col_200x200.tar -C data/kth_tips
```

The paper's protocol (Fig. 3–6, about 20 seconds on a CPU):

```bash
.venv/bin/python -m csp_gnn_dqn.reproduce
```

Including the two 40,000-episode experiments (about 9 more minutes):

```bash
.venv/bin/python -m csp_gnn_dqn.reproduce --extended
```

For another seed add `--seed 1 --out-dir results/seed_1`. The two stages can also be run on their own with `python -m csp_gnn_dqn.train_gnn` and `python -m csp_gnn_dqn.train_combined` (`--help` lists all hyperparameters). Results are deterministic for a given seed.

## Code layout

| File | Contents | Paper |
|---|---|---|
| `src/csp_gnn_dqn/data.py` | KTH-TIPS loading, grayscale conversion, crop and resize to 128×128, LBP histograms, graph construction | Sec. 2.1 |
| `src/csp_gnn_dqn/gnn.py` | 3 GCN layers + Jumping Knowledge + global mean pooling + fully connected classifier | Sec. 2.2, Eq. (1)(2) |
| `src/csp_gnn_dqn/env.py` | 2-D cutting stock environment, action space, first-fit baseline | Sec. 2.3, Eq. (3)(4) |
| `src/csp_gnn_dqn/dqn.py` | Q-network, experience replay, TD loss | Sec. 2.3, Eq. (5)(6) |
| `src/csp_gnn_dqn/train_gnn.py` | Stage 1: GNN texture classification | Fig. 3, Fig. 4 |
| `src/csp_gnn_dqn/train_combined.py` | Stage 2: joint GNN + DQN training and policy evaluation | Fig. 5, Fig. 6 |
| `src/csp_gnn_dqn/plots.py` | Figures (grey lines are the paper's curves read off its figures, for comparison only) | — |
| `src/csp_gnn_dqn/reproduce.py` | Runs everything | — |

## As specified in the paper

- Data: KTH-TIPS, 10 classes × 81 images = 810 images; converted to grayscale, cropped to a square and resized to 128×128; LBP histograms as input features.
- GNN: 3 GCN layers (ReLU, symmetrically normalised adjacency with self-loops as in Eq. (1)); Jumping Knowledge concatenates the representations of layers 0–3, followed by a linear map (Eq. (2)); global mean pooling gives the graph vector v_G; a fully connected layer classifies.
- DQN: state s = [v_G, f_C]; |A| = 100 actions (x, y, θ) with θ ∈ {0°, 90°}; MLP with hidden layers of 128 and 64 units (ReLU); replay memory of 10,000; batch size 32; γ = 0.99; target network synchronised every 1,000 steps; squared TD error (Eq. (6)).
- Budget: GNN 20 epochs; joint training 20 epochs and 100 episodes.

## Details not specified in the paper

The paper does not specify the following, and the original code is lost. This rewrite makes these choices:

| Item | Choice | Rationale |
|---|---|---|
| Graph nodes and edges | The image is split into 8×8 patches of 16×16 pixels, one node per patch (64 nodes), with 4-neighbour edges | The paper represents the image height and width as graph nodes; a histogram needs a region |
| Node features | 10-bin histogram of uniform LBP (P=8, R=1) within the patch | The paper uses LBP histograms as input features. Appending the flattened grayscale pixels (`--features lbp+gray`) is clearly worse, see the table below |
| Train/test split | Stratified 80/20 per class (650/160) | Not specified |
| Hidden size, optimiser | 64; Adam, lr 1e-3, batch 32 | Not specified; common defaults, not tuned to match the paper's curves |
| ε schedule | Start at 1.0, ×0.995 per episode | Inferred from the ε curve in Fig. 6, which it matches |
| Stock and orders | A 10×5-cell sheet (2:1), so 10×5×2 = 100 actions; the sheet is split into rectangles by random guillotine cuts, which arrive one at a time in random order, so 100% utilisation is achievable in hindsight | The paper does not define the environment |
| Reward | The area of a placed piece divided by the sheet area, so the return of an episode is its utilisation; an infeasible cut costs 0.02 (one cell) and the piece is discarded | The paper does not define the reward, so the reward axis of Fig. 6 cannot be matched |
| f_C | Remaining area, number of cuts made, width and height of the current piece, plus a 50-cell occupancy map | The first three are listed in the paper; without the occupancy map the agent cannot see where the sheet is free |
| How the material affects cutting | Corduroy, cotton and linen have a grain, so their pieces may not be rotated (θ = 90° is invalid) | The paper says decisions depend on material characteristics but not how; disable with `--no-grain-constraint` |
| "Combined loss" | Each epoch: one pass of the GNN over the training set (cross-entropy) → refresh v_G for all materials → 5 episodes with one DQN update per step; combined loss = cross-entropy + TD loss; the TD gradient does not flow into the GNN | The paper says the DRL "utilizes the data generated by the GNN"; Fig. 5 starts at 2.45, i.e. training starts from scratch |

## Results

### Stage 1: GNN (Fig. 3, Fig. 4)

![Fig. 3](results/fig3_gnn_train_loss.png)
![Fig. 4](results/fig4_gnn_test_loss.png)

Training loss 2.30 → 1.34, test loss 2.28 → 1.30, test accuracy 56% (54 ± 6% over 5 seeds; the paper does not report accuracy). The starting point is ln 10 ≈ 2.30, uniform guessing over 10 classes; the paper starts higher at 2.6, presumably because of a different initialisation. After 20 epochs the model is far from converged.

Sensitivity to the node features (lr 1e-3, seeds 0 and 1, loss at epoch 20):

| Node features | Training loss | Test loss |
|---|---|---|
| LBP histogram (default) | 1.33–1.34 | 1.30–1.49 |
| LBP histogram + flattened grayscale pixels | 1.66–1.74 | 1.69–1.70 |
| Flattened grayscale pixels only | 1.71–1.81 | 1.67–1.76 |

The default was chosen after seeing this table: of the three readings of the paper, it is the closest both to the paper's text and to its curves.

### Stage 2: GNN + DQN (Fig. 5, Fig. 6)

![Fig. 5](results/fig5_combined_loss.png)
![Fig. 6](results/fig6_reward_epsilon.png)

- Combined loss 2.31 → 1.34. The TD loss is only about 0.002, so the curve is essentially the GNN's cross-entropy; for the same seed, the GNN training is identical to that of Fig. 3.
- The ε curve coincides with the paper's. To avoid a dual y-axis, reward and ε are drawn in two stacked panels.
- The reward fluctuates between −0.16 and 0.42 with no trend, as in the paper's Fig. 6.

After training, 200 episodes per policy on the test-set materials:

| Policy | Utilisation | Infeasible cuts |
|---|---|---|
| Greedy GNN-DQN (after 100 training episodes) | 10% | 87% |
| Random cut | 31% | 49% |
| Random feasible cut (uniform over positions where the piece fits) | 73% | 9% |
| First-fit heuristic | 84% | 6% |

![Utilisation](results/utilization_curve.png)

### Extended experiment (beyond the paper): 40,000 episodes

![Extended experiment](results/extended_utilization.png)

| Setting | Final utilisation | Observation |
|---|---|---|
| DQN as in the paper | 3% | The mean maximum Q-value reaches 18 by epoch 4 (returns are bounded by 1) and the TD loss 34; both fall back later, but by then the policy has collapsed to almost only infeasible cuts (94%) |
| + invalid-action masking + Double DQN | 79% | Q-values stay around 0.35; utilisation rises from 75% to 79%, fluctuating between 73% and 80% across epochs |

While investigating, I also tried a few variants (throwaway scripts, not part of the code; state f_C only, without v_G): Double DQN alone stops the Q-values from diverging, but utilisation stays below 25%; ending the episode on an infeasible cut, or raising the penalty to 0.2, also stays at or below 30%. None of them learns where pieces fit. My reading is that a 128-64 MLP has to learn, separately for each of the 100 actions, whether the piece fits given a 50-cell occupancy map, which is very sample-inefficient.

## Not verified

- The paper states that the structural information from the GNN improves the DRL's decisions, but it contains no ablation, and none is done here (a control without v_G). In this rewrite's environment the material affects cutting only through whether rotation is allowed, and with action masking that constraint is handled by the mask, so v_G matters little.
- The graph construction, environment and reward of the original code can no longer be checked, so the absolute loss values and the reward scale can only be compared with the paper qualitatively.
- The paper column in the tables is read off the paper's figures by eye, with an error of about ±0.03.

## License

The code is released under the [MIT License](LICENSE). The paper PDF is not covered by it; it is published under the publisher's CC BY 4.0 licence.
