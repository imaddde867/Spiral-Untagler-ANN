# The Spiral Untangler

This project explores how neural networks can learn to **untangle intertwined spirals** — a classic toy problem in nonlinear classification.  

The goal is to build a neural network **from scratch (NumPy only)** and visualize how each layer geometrically transforms the data space in real time. Along the way, we’ll investigate the limits of simple architectures and experiment with progressively harder datasets.

---

## Features
- **Spiral Data Generation**: Easy, medium, and hard intertwined spirals.
- **Visualization Tools**: Scatter plots, decision boundaries, and cost curves.
- **Neural Network Core**:
  - Xavier initialization
  - Forward & backward propagation
  - Gradient descent parameter updates
- **Training Loop**: Cost tracking and accuracy evaluation.
- **Decision Boundary Plotting**: See how the network partitions space.
- **Challenge Modes**: Test performance on progressively harder spirals.

---

## Example Results
- Easy spirals → ~58% accuracy with 2-layer net
- Hard spirals → ~50% accuracy (chance level)

*(Clearly, shallow nets struggle with topologically tricky datasets!)*

---

## Next Steps
- The **"Impossibility Dataset"** for more extreme tests  
- Layer-by-layer **transformation visualization**  
- **Animated training process**  
- Gradient flow analysis (where they vanish/explode)  
- Experiment with **wider & deeper architectures**  
- Design **new topologically challenging datasets**

---

## 🛠️ Getting Started
Clone the repo and run the notebooks/scripts:

```bash
git clone https://github.com/your-username/spiral-untangler.git
cd spiral-untangler
python main.py
````

Requirements:

* Python 3.x
* NumPy
* Matplotlib

---

## References

Olah, C. (2014, March). Neural networks, manifolds, and topology. Christopher Olah's Blog. Retrieved from https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/
