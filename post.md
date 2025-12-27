## Project Title
The Spiral Untangler: NumPy Neural Network on Intertwined Spirals

## Author line
Imad Eddine - Computer Science / ML Engineer

## Hero image
![Successful untangling with a wide hidden layer](https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/width100_cost_boundary.png)

## Short Description
This project builds a 2-layer neural network from scratch (NumPy only) to classify intertwined spirals, a nonlinear toy problem where linear separators fail. It pairs the core ML loop (Xavier init, forward/backprop, gradient descent) with visual diagnostics: cost curves, decision boundaries, and confidence maps. Two experiments show the capacity gap between a narrow model that underfits and a wider model that can "untangle" the space.

## Full Description
The notebook (`main.ipynb`) generates a synthetic 2-class spiral dataset and trains a minimalist MLP (2 -> H -> 1) with tanh hidden activations and a sigmoid output. The training loop is fully vectorized and implements binary cross-entropy with a small epsilon for numerical stability. To make model behavior inspectable, it renders decision boundaries over a dense 2D mesh, tracks training cost, and visualizes uncertainty as |A2 - 0.5|.

## Pipeline or Architecture
1. Generate intertwined spirals with configurable difficulty/noise (`generate_intertwined_spirals`).
2. Reshape to matrix form: `X` as (2, m) and `Y` as (1, m).
3. Initialize parameters with Xavier scaling and a fixed seed (`init_params`, seed=42).
4. Forward pass: `tanh` hidden layer -> `sigmoid` output (`forward_prop`).
5. Compute binary cross-entropy loss with epsilon (`compute_cost`).
6. Backpropagate analytic gradients (`back_prop`).
7. Update weights with gradient descent (`update_params`).
8. Plot cost curves and decision boundaries; render a boundary-evolution animation; generate a confidence heatmap.

## Dataset/Training Snapshot
| Item | Value |
|---|---|
| Dataset | Intertwined spirals (2 classes) |
| Samples | 1000 (500 per class) |
| Features | 2 (x, y) |
| Difficulty params | hard, rotations=3.5, noise=0.2 |
| Model | 2 -> H -> 1 (tanh, sigmoid) |
| Loss | Binary cross-entropy (epsilon=1e-15) |
| Experiment A (bottleneck) | H=4, 5,000 iterations, learning_rate=0.5 |
| Experiment B (wide) | H=100, 10,000 iterations, learning_rate=0.8 |
| Init seed | 42 (weights) |

## Evaluation/Tracking Snapshot
| Epoch (H=100) | Cost (BCE) |
|---:|---:|
| 0 | 0.8374 |
| 1000 | 0.9168 |
| 2000 | 3.6613 |
| 3000 | 3.7177 |
| 4000 | 3.3317 |
| 5000 | 2.8076 |
| 6000 | 3.8639 |
| 7000 | 3.9763 |
| 8000 | 1.2682 |
| 9000 | 1.7117 |

| Setting (from README) | Reported accuracy |
|---|---:|
| Easy spirals, shallow net | ~58% |
| Hard spirals, shallow net | ~50% (chance) |

## Visual Evidence and Artifacts
![Hard spiral dataset ("Topological Nightmare")](https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/dataset_spirals.png)

![Width=4 underfitting (cost curve + boundary)](https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/width4_cost_boundary.png)

![Width=100 boundary after training (cost curve + boundary)](https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/width100_cost_boundary.png)

![Confidence map (where predictions are unsure)](https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/confidence_map.png)

## Engineering Highlights
- NumPy-only MLP with explicit forward/backprop and vectorized matrix math.
- Xavier initialization and numerically stable binary cross-entropy (epsilon guard).
- Parameterized synthetic data generator to scale difficulty (rotations, noise, sample count).
- Experiment design that isolates model capacity (H=4 vs H=100) with comparable training loops.
- Inspection-first tooling: decision boundaries on mesh grids, cost tracking, and an uncertainty heatmap.
- Animation artifacts that make training dynamics debuggable frame-by-frame.

## Tech Stack
Python, NumPy, Matplotlib, Jupyter Notebook, IPython

## Demo Video
<video controls src="https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/training_boundary_evolution.mp4" poster="https://raw.githubusercontent.com/imaddde867/Spiral-Untagler-ANN/main/results/showcase/width100_cost_boundary.png"></video>

## Try It
1. `python -m venv .venv && source .venv/bin/activate`
2. `python -m pip install numpy matplotlib jupyter`
3. `jupyter notebook main.ipynb` (then run all cells)
