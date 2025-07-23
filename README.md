# Perceptron Algorithm Implementation

A comprehensive implementation of the Perceptron algorithm for binary classification with detailed comments and visualizations.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Jupyter](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)

## 📖 Overview

This repository contains a detailed implementation of the **Perceptron algorithm**, one of the fundamental algorithms in machine learning for binary classification. The implementation includes:

- **Synthetic dataset generation** using scikit-learn
- **Step-by-step Perceptron training** with visualization
- **Real-time decision boundary updates** during training
- **Comprehensive comments** explaining each line of code
- **Mathematical explanations** of the algorithm

## 🧠 What is the Perceptron Algorithm?

The Perceptron is a linear binary classifier that finds a hyperplane to separate two classes of data. It works by:

1. **Initialization**: Start with random weights and bias
2. **Prediction**: Calculate the signed distance from each point to the current hyperplane
3. **Update**: Adjust weights and bias when a point is misclassified
4. **Convergence**: Repeat until all points are correctly classified

### Mathematical Foundation

The decision boundary is defined by: **w₁x₁ + w₂x₂ + w₀ = 0**

- **w₁, w₂**: Weights for features
- **w₀**: Bias term
- **x₁, x₂**: Input features

## 🚀 Features

- **Interactive Visualization**: See how the decision boundary evolves during training
- **Real-time Metrics**: Track accuracy and weight updates at each iteration
- **Educational Code**: Extensively commented for learning purposes
- **Synthetic Data**: Generates linearly separable clusters for demonstration
- **Convergence Detection**: Automatically stops when optimal solution is found

## 📁 Repository Structure

```
perceptron-algorithm/
│
├── 04_Linear_Algebra_4_Notebook.ipynb    # Main implementation notebook
├── README.md                             # This file
└── .gitignore                           # Git ignore rules
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Clone the Repository

```bash
git clone https://github.com/PAVANKUMARELETI/perceptron-algorithm.git
cd perceptron-algorithm
```

## 🔬 Usage

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**: `04_Linear_Algebra_4_Notebook.ipynb`

3. **Run all cells** to see the Perceptron algorithm in action!

### Quick Start Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Generate synthetic data
x, y = datasets.make_blobs(n_samples=200, n_features=2, 
                          centers=[[2,2],[10,10]], cluster_std=3)

# Convert labels to -1 and +1
y[y==0] = -1

# Initialize weights
w = np.random.normal(size=(2,))
w0 = np.random.normal()

# Train the Perceptron
w, w0 = perceptron_training(x, y, w, w0)
```

## 📊 Algorithm Workflow

1. **Data Generation**
   - Creates 200 synthetic data points
   - Two clusters centered at (2,2) and (10,10)
   - Labels converted from {0,1} to {-1,+1}

2. **Visualization Setup**
   - Plots data points colored by class
   - Shows decision boundary as a line
   - Updates visualization after each iteration

3. **Training Process**
   - Iterates through each data point
   - Calculates signed distance to hyperplane
   - Updates weights when misclassification occurs
   - Tracks accuracy and convergence

4. **Convergence**
   - Stops when all points are correctly classified
   - Returns final learned weights and bias

## 🎯 Key Functions

### `show(w, w0, x, y)`
Visualizes the dataset and current decision boundary.

### `dist_from_hyperplane(w, w0, x)`
Calculates the signed distance from a point to the hyperplane.

### `perceptron_training(x, y, w, w0, num_iter=100)`
Main training function that implements the Perceptron learning algorithm.

## 📈 Expected Output

The algorithm will display:
- **Iteration number** for each training step
- **Current accuracy** as percentage
- **Updated weights** (W1, W2, W0)
- **Visual plot** showing decision boundary evolution

Example output:
```
Iteration Number: 1
Current Accuracy: 85.0
Weights are W1: 1.234, W2: -0.567, W0: 2.345
[Plot showing decision boundary]
```

## 🎓 Educational Value

This implementation is perfect for:
- **Machine Learning students** learning linear classifiers
- **Data Science enthusiasts** understanding neural network foundations
- **Algorithm visualization** and understanding convergence
- **Linear algebra applications** in machine learning

## 🔍 Algorithm Analysis

### Time Complexity
- **Training**: O(n × k) where n = number of samples, k = number of iterations
- **Space**: O(d) where d = number of features

### Assumptions
- Data must be **linearly separable**
- Works only for **binary classification**
- Requires **numerical features**

## 🤝 Contributing

Contributions are welcome! Here are some ways you can help:

1. **Bug Reports**: Open an issue if you find any bugs
2. **Feature Requests**: Suggest new features or improvements
3. **Documentation**: Help improve the documentation
4. **Code**: Submit pull requests with enhancements

### How to Contribute

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Pavan Kumar Eleti**
- GitHub: [@PAVANKUMARELETI](https://github.com/PAVANKUMARELETI)

## 🙏 Acknowledgments

- **Frank Rosenblatt** for inventing the Perceptron algorithm (1957)
- **Scikit-learn** for providing excellent synthetic data generation tools
- **Matplotlib** for powerful visualization capabilities
- **NumPy** for efficient numerical computations

## 📚 References

1. Rosenblatt, F. (1958). The perceptron: a probabilistic model for information storage and organization in the brain.
2. Bishop, C. M. (2006). Pattern Recognition and Machine Learning.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning.

## 🏷️ Keywords

`machine-learning` `perceptron` `binary-classification` `linear-classifier` `neural-networks` `supervised-learning` `python` `jupyter-notebook` `data-science` `algorithm-implementation` `educational` `linear-algebra` `visualization`

---

⭐ **Star this repository** if you found it helpful!

🐛 **Found a bug?** [Open an issue](https://github.com/PAVANKUMARELETI/perceptron-algorithm/issues)

💡 **Have a suggestion?** [Start a discussion](https://github.com/PAVANKUMARELETI/perceptron-algorithm/discussions)
