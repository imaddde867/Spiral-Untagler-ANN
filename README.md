# The Spiral Untangler

This project shows how a neural network can **untangle intertwined spirals** (a classic nonlinear classification problem) using a from-scratch NumPy MLP. The notebook focuses on clarity: every step of forward/backprop is explicit, training is stable via normalization + Adam, and results are visualized with decision boundaries, confidence maps, and an animation.

---

## Highlights
- **Spiral data generation** with configurable difficulty, noise, and fixed seeds.
- **Input normalization** to [-1, 1] for stable training.
- **MLP from scratch (NumPy only)** with tanh + sigmoid, Xavier init, and Adam.
- **Diagnostics**: cost curves, boundary plots, uncertainty heatmap, training animation.
- **Reproducibility**: fixed seeds for data (7) and weights (42).

---

## Results (Hard spirals, seed=7)
| Setting | Steps | Optimizer | Final Cost | Training Accuracy |
|---|---:|---|---:|---:|
| Width=4 (underfit) | 5,000 | Adam (lr=0.01) | ~0.59 | 59.40% |
| Width=100 (untangle) | 15,000 | Adam (lr=0.01) | 0.0418 | 99.90% |

Dataset details:
- Samples: 1000 (500 per class)
- Rotations: 3.5 (hard)
- Noise: 0.2
- Normalization: divide by max abs of raw X (scale=47.66)

---

## Artifacts (Latest)
- Dataset: `results/showcase/dataset_spirals.png`
- Underfit baseline: `results/showcase/width4_cost_boundary.png`
- Untangled model: `results/showcase/width100_cost_boundary.png`
- Confidence map: `results/showcase/confidence_map.png`
- Training animation: `results/showcase/training_boundary_evolution.mp4`
- Preview GIF (1600x900): `results/showcase/width100_cost_boundary.gif`

---

## Quickstart
```
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy matplotlib jupyter
jupyter notebook main.ipynb
```
Run all cells to regenerate the figures and animation.

---

## References
Olah, C. (2014, March). Neural networks, manifolds, and topology. Christopher Olah's Blog. https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
