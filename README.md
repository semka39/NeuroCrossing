# Neural Network Embedding Visualization Project

## 📌 Overview
This project explores how individual neurons in a neural network "remember" an image by analyzing their activation patterns. The key steps include:
1. Training a neural network to memorize an input image (RGB prediction from pixel coordinates).
2. Extracting neuron embeddings based on their influence on the output.
3. Visualizing neuron specialization patterns using dimensionality reduction and clustering.

![Example Visualization](https://i.imgur.com/z6tUppe.png)
![Example Visualization](https://i.imgur.com/u4aP7SL.png)

## 🔍 Methodology
### 1. Image Memorization
- **Input**: Pixel coordinates `(x, y)`
- **Output**: RGB color values
- The network successfully reconstructs the original image after training.

### 2. Neuron Embedding
- Computed gradient correlations between each neuron's activations and final RGB outputs
- Created high-dimensional embeddings representing each neuron's "role"

### 3. Dimensionality Reduction & Clustering
- Applied **t-SNE** to project embeddings to 2D/3D
- Used **DBSCAN** to identify functional clusters
- Special findings:
  - Some neurons form **continuous lines** (showing smooth functional transitions)
  - Others appear as **isolated clusters** (indicating specialized roles)

### 4. Cluster Ablation Study
- Systematically removed neurons from each cluster
- Reconstructed images without these neurons


Requires a [library](https://github.com/semka39/Simple-NN) to work with multilayer perceptrons.