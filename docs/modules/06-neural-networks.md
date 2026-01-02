# Module 6: Neural Networks Fundamentals

## Introduction

Today we cross a threshold—we're entering deep learning.

Everything we've covered so far—regression, classification, ensemble methods, unsupervised learning—those are "classical" machine learning. Powerful, interpretable, widely used. But deep learning has transformed what's possible with images, text, audio, and complex patterns.

Here's the key insight: **neural networks are not magic.** They're built on the same principles we've been learning. Remember gradient descent from Module 2? You'll see it again. Remember the bias-variance tradeoff from Module 1? It applies here too.

What makes neural networks special is their ability to learn hierarchical representations—layer by layer, from simple patterns to complex concepts.

**Hierarchical learning is automatic**: We design the architecture and loss function; the specific representations are discovered, not designed. Through backpropagation, weights organize themselves to extract useful features. Researchers visualizing trained networks find edges in layer 1, textures in layer 2, object parts in later layers—this emerges from optimization as the most efficient solution.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the historical development and architecture of neural networks
2. **Describe** the components of a neural network (weights, biases, activations)
3. **Understand** backpropagation and gradient-based optimization
4. **Implement** a simple neural network in PyTorch
5. **Train** and evaluate networks on classification tasks
6. **Apply** regularization techniques to prevent overfitting

---

## 6.1 Introduction to Neural Networks

### Three Components: Neural Networks

The same framework applies here:

| Component | Neural Network |
|-----------|----------------|
| **Decision Model** | Stacked layers with non-linear activations |
| **Quality Measure** | Cross-entropy (classification) or MSE (regression) |
| **Update Method** | Backpropagation + gradient descent (SGD, Adam) |

In Module 2, you implemented gradient descent for two parameters ($\beta_0$, $\beta_1$). Neural networks apply the same idea to millions of parameters. The algorithm is the same; the scale is different.

### Historical Context

**1957**: Frank Rosenblatt invents the **Perceptron**—a single layer of weights that could learn simple patterns. The New York Times predicted thinking machines within a decade.

**1969**: Minsky and Papert publish "Perceptrons," proving single-layer networks can't learn XOR. Funding dries up. First "AI Winter."

**1986**: Rumelhart, Hinton, and Williams popularize **backpropagation**—making deep network training practical.

**2012**: **AlexNet** wins ImageNet by a massive margin, demonstrating that deep networks trained on GPUs could dramatically outperform traditional methods.

**Today**: Transformers, GPT, and large language models.

**The lesson**: Neural networks have existed for 70 years. What changed is data, compute, and better training techniques.

**Why deep learning works now**: Three factors combined: (1) **Data**—ImageNet provided 14M labeled images; the internet generated billions of documents. (2) **GPUs**—parallel operations for matrix multiplication, turning weeks into hours. (3) **Better techniques**—ReLU solved vanishing gradients, dropout provided regularization, batch norm stabilized training, Adam made optimization robust. AlexNet (2012) combined all three and won ImageNet decisively.

### The XOR Problem

The XOR function outputs 1 if exactly one input is 1:

| x₁ | x₂ | XOR |
|----|----|----|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

A single-layer perceptron can only learn linearly separable patterns. XOR isn't linearly separable—you can't draw a single straight line to separate the 1s from the 0s.

**The solution**: Add a hidden layer. The hidden layer "transforms" the space to make the problem linearly separable.

**How the hidden layer transforms space**: Each neuron computes a weighted sum (defining a hyperplane) plus activation (bending space around it). For XOR, one neuron might learn "x₁ + x₂ > 0.5" and another "x₁ + x₂ < 1.5"—together creating a representation where (0,1) and (1,0) map similarly while (0,0) and (1,1) map differently. The output layer can now draw a line in this transformed space.

> **Numerical Example: XOR with a Hidden Layer**
>
> ```python
> import torch
> import torch.nn as nn
> import torch.optim as optim
>
> torch.manual_seed(42)
>
> # XOR data
> X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
> y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
>
> # Network: 2 inputs -> 4 hidden (tanh) -> 1 output (sigmoid)
> class XORNet(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.hidden = nn.Linear(2, 4)
>         self.output = nn.Linear(4, 1)
>
>     def forward(self, x):
>         h = torch.tanh(self.hidden(x))
>         return torch.sigmoid(self.output(h))
>
> model = XORNet()
> optimizer = optim.Adam(model.parameters(), lr=0.5)
> criterion = nn.BCELoss()
>
> for epoch in range(2000):
>     optimizer.zero_grad()
>     loss = criterion(model(X), y)
>     loss.backward()
>     optimizer.step()
>
> # Test
> model.eval()
> with torch.no_grad():
>     for i in range(4):
>         pred = model(X[i:i+1]).item()
>         print(f"({X[i,0]:.0f}, {X[i,1]:.0f}) -> {pred:.3f} -> {1 if pred > 0.5 else 0}")
> ```
>
> **Output:**
> ```
> (0, 0) -> 0.000 -> 0
> (0, 1) -> 1.000 -> 1
> (1, 0) -> 1.000 -> 1
> (1, 1) -> 0.000 -> 0
> ```
>
> **Interpretation:** The network learns XOR perfectly. The hidden layer transforms the 2D input space so that (0,0) and (1,1) map to one region while (0,1) and (1,0) map to another—making the problem linearly separable for the output layer.
>
> *Source: `slide_computations/module6_examples.py` - `demo_xor_hidden_layer()`*

### Multi-Layer Perceptron (MLP) Architecture

![MLP Architecture](../assets/module6/mlp_architecture.png)

**Reading the diagram**: This network has three input neurons (x1, x2, x3) shown in blue on the left, two hidden layers with four neurons each shown in purple in the middle, and a single output neuron (ŷ) shown in gray on the right. Every neuron in one layer connects to every neuron in the next layer—these gray lines represent the weights that the network learns during training. Information flows left to right: inputs enter, get transformed through hidden layers, and produce a prediction. The "depth" of this network is 2 (two hidden layers), and the "width" of each hidden layer is 4. Notice that the input layer is not counted when describing network depth—it's just the raw data entry point.

**Terminology:**
- **Input layer**: Raw features (not counted in "layers")
- **Hidden layers**: Intermediate representations
- **Output layer**: Final predictions
- **Depth**: Number of hidden layers
- **Width**: Neurons per layer

| Network Type | Hidden Layers | Typical Use |
|-------------|---------------|-------------|
| Shallow | 1-2 | Simple patterns |
| Deep | 3+ | Complex patterns |
| Very Deep | 50+ | State-of-the-art |

### Why Depth Matters

Each layer learns more abstract features:
- **Layer 1**: Edges, simple patterns
- **Layer 2**: Textures, shapes
- **Layer 3**: Object parts
- **Layer N**: Complete concepts

Deep networks learn hierarchical representations that match how complex patterns are actually structured.

### Universal Approximation Theorem

A feedforward network with a single hidden layer can approximate any continuous function, given enough neurons.

**What it means**: With enough neurons, any reasonable function can be approximated.

**What it doesn't mean**: It doesn't tell you how many neurons you need, how to find the weights, or that one layer is optimal.

In practice, deep networks represent the same functions more efficiently than wide shallow ones.

**Why depth over width?** A function that a 10-layer network represents with 1,000 neurons might require millions in a single layer. Complex patterns are compositional (faces = eyes + nose + mouth; eyes = curves + colors)—deep networks represent this hierarchy naturally. Shallow networks must learn all combinations directly, which explodes exponentially. Deeper architectures outperform shallow ones with the same parameter count on complex benchmarks.

### Network Components

**1. Weights (W)**: Learnable parameters connecting neurons
**2. Biases (b)**: Learnable offset per neuron
**3. Activation functions**: Non-linear transformations

The computation at each neuron:

$$output = activation(Wx + b)$$

### Activation Functions

**ReLU (Rectified Linear Unit)** — most common:

$$\text{ReLU}(x) = \max(0, x)$$

- Simple: negative → 0, positive → pass through
- Default choice for hidden layers
- Helps with vanishing gradients

**Sigmoid**:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- Output between 0 and 1
- Good for binary output layer
- Suffers from vanishing gradients in deep networks

**Softmax** (for multi-class):

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

- Outputs sum to 1 (probabilities)
- Used in final layer for classification

### Why Non-linear Activations?

Without non-linearity:

$$Layer_2(Layer_1(x)) = W_2(W_1 x) = (W_2 W_1)x = Wx$$

**Multiple linear layers = one linear layer!**

No matter how many linear layers you stack, the result is still linear. Non-linear activations allow each layer to transform representations in ways linear functions can't.

**Why ReLU works**: (1) **Vanishing gradient solution**—sigmoid's gradient approaches zero for large inputs; ReLU has gradient 1 for positives, letting gradients pass through unchanged. (2) **Computational efficiency**—just max(0,x), orders of magnitude faster than sigmoid. (3) **Sparse activation**—50% of neurons may be "dead" for any input, improving efficiency. Despite being piecewise linear, stacking many ReLUs can approximate any continuous function.

**Understanding "dead" ReLU neurons**: When a neuron's input is negative, ReLU outputs 0 and its gradient is also 0. This means negative-input neurons don't contribute to predictions or learning for that example. While this sounds problematic, it's actually beneficial: (1) it creates **sparsity**—only a subset of neurons activate for any given input, making computation efficient, and (2) different inputs activate different neuron subsets, so the network implicitly learns specialized sub-networks for different patterns. However, if a neuron's weights drift so that it *always* receives negative inputs (for all training examples), it becomes permanently "dead" and stops learning. This is the "dying ReLU" problem, which techniques like Leaky ReLU address.

> **Numerical Example: ReLU vs Sigmoid Gradients**
>
> ```python
> import numpy as np
>
> def sigmoid(x):
>     return 1 / (1 + np.exp(-x))
>
> def sigmoid_gradient(x):
>     s = sigmoid(x)
>     return s * (1 - s)
>
> def relu_gradient(x):
>     return 1.0 if x > 0 else 0.0
>
> # Simulate gradient flowing backward through 10 layers
> # (assuming all neurons in saturated sigmoid region, z=2)
> print("Gradient flowing backward through 10 layers:")
> print(f"{'Layer':>6} {'Sigmoid grad':>15} {'ReLU grad':>15}")
>
> sigmoid_grad = 1.0
> relu_grad = 1.0
> for layer in range(10, 0, -1):
>     print(f"{layer:>6} {sigmoid_grad:>15.6f} {relu_grad:>15.1f}")
>     sigmoid_grad *= sigmoid_gradient(2.0)  # Saturated region
>     relu_grad *= relu_gradient(2.0)        # Positive region
>
> print(f"\nAfter 10 layers: sigmoid={sigmoid_grad:.2e}, ReLU={relu_grad:.1f}")
> ```
>
> **Output:**
> ```
> Gradient flowing backward through 10 layers:
>  Layer    Sigmoid grad        ReLU grad
>     10        1.000000             1.0
>      9        0.104994             1.0
>      8        0.011024             1.0
>      7        0.001157             1.0
>      6        0.000122             1.0
>      5        0.000013             1.0
>      4        0.000001             1.0
>      3        0.000000             1.0
>      2        0.000000             1.0
>      1        0.000000             1.0
>
> After 10 layers: sigmoid=1.63e-10, ReLU=1.0
> ```
>
> **Interpretation:** With sigmoid activations, the gradient shrinks by ~10x at each layer. After 10 layers, it's essentially zero (1.63×10⁻¹⁰)—early layers receive no learning signal. ReLU maintains gradient magnitude, enabling training of very deep networks. This is the **vanishing gradient problem** that plagued early deep learning.
>
> *Source: `slide_computations/module6_examples.py` - `demo_relu_vs_sigmoid_gradients()`*

### Parameter Counting

**For a fully connected layer:**

$$Parameters = (input \times output) + output = weights + biases$$

**Example**: Network with layers [784, 256, 128, 10]
- Layer 1: 784×256 + 256 = 200,960
- Layer 2: 256×128 + 128 = 32,896
- Layer 3: 128×10 + 10 = 1,290
- **Total: 235,146 parameters**

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**Is 10 million parameters a lot?** It depends on your data. If you have 1,000 examples and 10 million parameters, you'll overfit. If you have 10 million examples, it's reasonable. The ratio matters.

> **Numerical Example: Parameter Counting Walkthrough**
>
> ```python
> def count_params(architecture):
>     """Count parameters for a fully connected network."""
>     total = 0
>     for i in range(len(architecture) - 1):
>         weights = architecture[i] * architecture[i + 1]
>         biases = architecture[i + 1]
>         total += weights + biases
>         print(f"Layer {i+1}: {architecture[i]}x{architecture[i+1]} "
>               f"= {weights:,} weights + {biases} biases = {weights + biases:,}")
>     return total
>
> # Three different architectures for MNIST (784 inputs, 10 outputs)
> print("Architecture 1: [784, 256, 128, 10]")
> total1 = count_params([784, 256, 128, 10])
> print(f"Total: {total1:,}\n")
>
> print("Architecture 2 (deeper): [784, 128, 64, 32, 16, 10]")
> total2 = count_params([784, 128, 64, 32, 16, 10])
> print(f"Total: {total2:,}\n")
>
> print("Architecture 3 (wider): [784, 512, 10]")
> total3 = count_params([784, 512, 10])
> print(f"Total: {total3:,}")
> ```
>
> **Output:**
> ```
> Architecture 1: [784, 256, 128, 10]
> Layer 1: 784x256 = 200,704 weights + 256 biases = 200,960
> Layer 2: 256x128 = 32,768 weights + 128 biases = 32,896
> Layer 3: 128x10 = 1,280 weights + 10 biases = 1,290
> Total: 235,146
>
> Architecture 2 (deeper): [784, 128, 64, 32, 16, 10]
> Layer 1: 784x128 = 100,352 weights + 128 biases = 100,480
> Layer 2: 128x64 = 8,192 weights + 64 biases = 8,256
> Layer 3: 64x32 = 2,048 weights + 32 biases = 2,080
> Layer 4: 32x16 = 512 weights + 16 biases = 528
> Layer 5: 16x10 = 160 weights + 10 biases = 170
> Total: 111,514
>
> Architecture 3 (wider): [784, 512, 10]
> Layer 1: 784x512 = 401,408 weights + 512 biases = 401,920
> Layer 2: 512x10 = 5,120 weights + 10 biases = 5,130
> Total: 407,050
> ```
>
> **Interpretation:** The first layer (connecting to high-dimensional input) dominates the parameter count. Deeper networks can actually have *fewer* parameters than wide shallow ones while achieving better representational power. Architecture 2 has 5 layers but only 111K parameters, while architecture 3 has just 2 layers but 407K parameters.
>
> *Source: `slide_computations/module6_examples.py` - `demo_parameter_counting()`*

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Deep learning is different from ML" | Deep learning IS machine learning. Same principles apply. |
| "More layers always better" | Deeper = harder to train, can overfit. Match depth to complexity. |
| "Neural networks are black boxes" | Many interpretability tools exist. The criticism is overstated. |
| "Need millions of data points" | Transfer learning enables NNs with small datasets. |

---

## 6.2 Training Neural Networks

### Loss Functions

**Regression — Mean Squared Error (MSE):**

$$L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$$

**Binary Classification — Binary Cross-Entropy:**

$$L = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

**Multi-class — Cross-Entropy:**

$$L = -\frac{1}{n}\sum_{i}\sum_{c} y_{ic}\log(\hat{y}_{ic})$$

**Why cross-entropy?** The log function severely penalizes confident wrong predictions:
- $\log(1) = 0$ — no penalty for correct confidence
- $\log(0.5) \approx -0.69$ — moderate penalty
- $\log(0.01) \approx -4.6$ — severe penalty

**The information-theoretic intuition**: Cross-entropy measures "surprise." If you're 99% confident an email is spam and it turns out to be legitimate, that's very surprising—and the network should pay a heavy penalty for that confident mistake. MSE treats all errors linearly, but cross-entropy's log penalty means confident-wrong is exponentially worse than uncertain-wrong. This provides much stronger gradients to fix the most problematic predictions.

> **Numerical Example: Cross-Entropy vs MSE for Classification**
>
> ```python
> import numpy as np
>
> def mse_gradient(y_true, y_pred):
>     return -2 * (y_true - y_pred)
>
> def cross_entropy_gradient(y_true, y_pred):
>     eps = 1e-10
>     return -(y_true / (y_pred + eps)) + (1 - y_true) / (1 - y_pred + eps)
>
> # True label is 1 (positive class), compare gradients at different predictions
> y_true = 1.0
> predictions = [0.99, 0.9, 0.5, 0.1, 0.01]
>
> print(f"True label: {y_true} (positive class)")
> print(f"{'Prediction':>12} {'MSE Grad':>12} {'CE Grad':>12}")
> for p in predictions:
>     print(f"{p:>12.2f} {mse_gradient(y_true, p):>12.2f} "
>           f"{cross_entropy_gradient(y_true, p):>12.2f}")
> ```
>
> **Output:**
> ```
> True label: 1.0 (positive class)
>   Prediction     MSE Grad      CE Grad
>         0.99        -0.02        -1.01
>         0.90        -0.20        -1.11
>         0.50        -1.00        -2.00
>         0.10        -1.80       -10.00
>         0.01        -1.98      -100.00
> ```
>
> **Interpretation:** When the model predicts 0.01 for a true positive (confidently wrong), cross-entropy provides a gradient of -100 while MSE gives only -1.98. This 50x stronger signal means cross-entropy can fix catastrophic mistakes much faster. MSE's gradients plateau near the extremes, making it sluggish at correcting confident errors.
>
> *Source: `slide_computations/module6_examples.py` - `demo_cross_entropy_vs_mse()`*

### Backpropagation

**The algorithm that makes deep learning possible.**

1. **Forward pass**: Compute predictions
2. **Compute loss**: How wrong are we?
3. **Backward pass**: Compute gradients using chain rule
4. **Update**: Adjust weights

The chain rule lets us compute how each weight contributed to error, layer by layer, from output back to input.

**Why gradient computation is fast**: Backpropagation reuses computations—when computing gradients for layer 5, you reuse gradient info from layers 6-10. Total cost is ~2× the forward pass, O(n) in weights. GPUs parallelize matrix multiplications across thousands of cores. Processing 64 examples in parallel takes almost the same time as 1. A network with 100M parameters takes seconds per batch on modern GPUs.

**Key point**: PyTorch does this automatically!

```python
loss.backward()   # Computes all gradients
optimizer.step()  # Updates all parameters
```

One line computes gradients. One line updates weights.

> **Numerical Example: Backpropagation by Hand**
>
> Let's trace a single training step through a minimal network: 1 input → 1 hidden (ReLU) → 1 output (sigmoid).
>
> ```
> Initial: w1=0.5, b1=0.1, w2=0.8, b2=-0.2
> Input: x=0.5, True label: y=1
>
> --- FORWARD PASS ---
> z1 = w1*x + b1 = 0.5*0.5 + 0.1 = 0.35
> h1 = ReLU(z1) = max(0, 0.35) = 0.35
> z2 = w2*h1 + b2 = 0.8*0.35 + (-0.2) = 0.08
> y_pred = sigmoid(z2) = 0.5200
>
> Loss = -[y*log(y_pred)] = -log(0.52) = 0.6539
>
> --- BACKWARD PASS ---
> dL/dy_pred = -1/y_pred = -1.9231
> dy_pred/dz2 = y_pred*(1-y_pred) = 0.2496
> dL/dz2 = -1.9231 * 0.2496 = -0.4800
>
> dL/dw2 = dL/dz2 * h1 = -0.4800 * 0.35 = -0.1680
> dL/db2 = dL/dz2 = -0.4800
>
> dL/dh1 = dL/dz2 * w2 = -0.4800 * 0.8 = -0.3840
> dh1/dz1 = ReLU'(0.35) = 1.0  (since z1 > 0)
> dL/dz1 = -0.3840 * 1.0 = -0.3840
>
> dL/dw1 = dL/dz1 * x = -0.3840 * 0.5 = -0.1920
> dL/db1 = dL/dz1 = -0.3840
>
> --- UPDATE (lr=0.1) ---
> w1_new = 0.5 - 0.1*(-0.1920) = 0.5192
> w2_new = 0.8 - 0.1*(-0.1680) = 0.8168
> ```
>
> **Interpretation:** The chain rule propagates error backward through each operation. Negative gradients mean we should *increase* the weights (moving opposite to the gradient decreases loss). After this single step, all weights increased slightly, which will push y_pred higher toward the true label of 1. PyTorch's `loss.backward()` computes all these gradients automatically.
>
> *Source: `slide_computations/module6_examples.py` - `demo_backprop_by_hand()`*

### Optimization Algorithms

**SGD (Stochastic Gradient Descent)**:

$$W \leftarrow W - \alpha \cdot \nabla L$$

Same as Module 2. Simple but can be slow.

**SGD + Momentum**:

$$v \leftarrow \beta v + \nabla L$$

$$W \leftarrow W - \alpha \cdot v$$

Accumulates velocity in consistent directions. Like a ball rolling downhill.

**Why momentum helps**: Imagine a loss surface shaped like a long, narrow valley. Plain SGD oscillates back and forth across the narrow dimension while making slow progress along the valley floor. Momentum accumulates velocity in the consistent direction (along the valley) while canceling out oscillations (across the valley). It also helps escape shallow local minima and saddle points—the accumulated momentum carries optimization past small bumps that would trap vanilla SGD.

**Adam (Adaptive Moment Estimation)** — most popular:
- Combines momentum with adaptive learning rates
- Per-parameter learning rates
- Usually works well with defaults

**How Adam works (simplified):**
- Track moving average of gradients (momentum)
- Track moving average of squared gradients (adapt rates)
- Parameters with large gradients get smaller learning rates

**The intuition behind adaptive rates**: Not all parameters need the same learning rate. A weight connected to a frequently-activated feature gets gradients on every batch—it should take smaller steps to avoid overshooting. A weight connected to a rare feature (like an uncommon word in NLP) gets gradients infrequently—when it does get a signal, it should take a larger step to make progress. Adam automatically scales learning rates: divide by the root-mean-square of recent gradients, so high-gradient parameters get smaller effective rates and vice versa.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Practical advice**: Start with Adam. Try SGD with momentum if you have time to tune.

### Learning Rate

**The most important hyperparameter.**

| Too High | Just Right | Too Low |
|----------|------------|---------|
| Loss explodes | Steady decrease | Very slow |
| Diverges | Converges | Gets stuck |

**Tips:**
- Start with 0.001 for Adam, 0.01 for SGD
- If loss explodes: divide by 10
- If loss barely moves: multiply by 3-10
- Use schedulers to reduce rate during training

> **Numerical Example: Learning Rate Effects on Neural Networks**
>
> ```python
> import torch
> import torch.nn as nn
> import torch.optim as optim
>
> torch.manual_seed(42)
>
> # Simple regression network
> X = torch.linspace(-2, 2, 100).reshape(-1, 1)
> y = torch.sin(X * 3.14) + torch.randn_like(X) * 0.1
>
> class Net(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.fc1 = nn.Linear(1, 32)
>         self.fc2 = nn.Linear(32, 1)
>     def forward(self, x):
>         return self.fc2(torch.relu(self.fc1(x)))
>
> for lr, label in [(0.0001, "Too small"), (0.01, "Good"), (1.0, "Too large")]:
>     torch.manual_seed(42)
>     model = Net()
>     optimizer = optim.SGD(model.parameters(), lr=lr)
>     for _ in range(50):
>         loss = nn.MSELoss()(model(X), y)
>         optimizer.zero_grad()
>         loss.backward()
>         optimizer.step()
>     print(f"lr={lr}: Final loss = {loss.item():.4f} ({label})")
> ```
>
> **Output:**
> ```
> lr=0.0001: Final loss = 0.4968 (Too small)
> lr=0.01: Final loss = 0.3692 (Good)
> lr=1.0: Final loss = inf (Too large)
> ```
>
> **Interpretation:** With lr=0.0001, the network barely learns in 50 epochs. With lr=0.01, it converges to a reasonable solution. With lr=1.0, the loss explodes to infinity—the optimizer overshoots so badly that weights become NaN. The fix is simple: if loss explodes, reduce learning rate by 10x.
>
> *Source: `slide_computations/module6_examples.py` - `demo_learning_rate_effects_nn()`*

### Batch Size

| Variant | Batch Size | Trade-off |
|---------|------------|-----------|
| Batch GD | All data | Stable but slow |
| SGD | 1 sample | Fast but noisy |
| Mini-batch | 32-256 | Best of both |

**Standard practice**: 32, 64, 128, or 256

**Trade-offs:**
- Larger: More stable, more memory, may generalize worse
- Smaller: Noisier (regularizing), faster per epoch

### Regularization: Dropout

**Randomly zero neurons during training.**

```python
self.dropout = nn.Dropout(0.5)  # 50% dropout
```

- Forces network to not rely on any single neuron
- Like training an ensemble of sub-networks
- Only active during training, not inference

**Connection to ensembles**: Dropout trains many different sub-networks (different neurons dropped each time) and averages at test time. It's bagging for neural networks.

**How dropout learning works**: Each training example sees a different random subset of neurons. Features that depend on one specific neuron won't work consistently (it might be dropped), forcing distributed, robust representations. At test time, ALL neurons are used but scaled by the dropout rate. The ensemble interpretation: training exponentially many sub-networks simultaneously, averaging at test time.

> **Numerical Example: Dropout Effect on Overfitting**
>
> ```python
> import torch
> import torch.nn as nn
>
> torch.manual_seed(42)
>
> # Small dataset (easy to overfit): 50 train, 200 test, 20 features
> n_train, n_test, n_features = 50, 200, 20
> X_train = torch.randn(n_train, n_features)
> true_w = torch.randn(n_features, 1)
> y_train = X_train @ true_w + torch.randn(n_train, 1) * 0.5
> X_test = torch.randn(n_test, n_features)
> y_test = X_test @ true_w + torch.randn(n_test, 1) * 0.5
>
> class Net(nn.Module):
>     def __init__(self, dropout_rate):
>         super().__init__()
>         self.fc1, self.fc2, self.fc3 = nn.Linear(20, 64), nn.Linear(64, 32), nn.Linear(32, 1)
>         self.dropout = nn.Dropout(dropout_rate)
>     def forward(self, x):
>         x = self.dropout(torch.relu(self.fc1(x)))
>         x = self.dropout(torch.relu(self.fc2(x)))
>         return self.fc3(x)
>
> for dropout in [0.0, 0.3, 0.5]:
>     torch.manual_seed(42)
>     model = Net(dropout)
>     opt = torch.optim.Adam(model.parameters(), lr=0.01)
>     for _ in range(200):
>         opt.zero_grad()
>         nn.MSELoss()(model(X_train), y_train).backward()
>         opt.step()
>     model.eval()
>     with torch.no_grad():
>         train_mse = nn.MSELoss()(model(X_train), y_train).item()
>         test_mse = nn.MSELoss()(model(X_test), y_test).item()
>     print(f"Dropout={dropout}: Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}, Gap={test_mse-train_mse:.4f}")
> ```
>
> **Output:**
> ```
> Dropout=0.0: Train MSE=0.0000, Test MSE=3.7633, Gap=3.7633
> Dropout=0.3: Train MSE=0.1016, Test MSE=2.6384, Gap=2.5368
> Dropout=0.5: Train MSE=0.1672, Test MSE=2.7283, Gap=2.5611
> ```
>
> **Interpretation:** Without dropout, the network achieves near-zero training error but terrible test error (gap of 3.76)—classic overfitting. With dropout=0.3, training error increases slightly but test error drops substantially. The train/test gap shrinks from 3.76 to 2.54, indicating better generalization. Dropout forces the network to learn robust features that don't depend on any single neuron.
>
> *Source: `slide_computations/module6_examples.py` - `demo_dropout_effect()`*

### Regularization: Batch Normalization

Normalize activations within each mini-batch.

```python
self.bn1 = nn.BatchNorm1d(256)
```

- Stabilizes training
- Allows higher learning rates
- Add after linear layer, before activation

**What "stabilizes" means**: As a network trains, the distribution of inputs to each layer keeps shifting because the previous layer's weights changed. Layer 5 has to constantly adapt to a moving target. This "internal covariate shift" makes training unstable and requires tiny learning rates. Batch normalization fixes this by normalizing each layer's inputs to zero mean and unit variance, then learning optimal scale and shift parameters. The layer always sees similarly-distributed inputs, regardless of what earlier layers are doing. This allows much higher learning rates and faster convergence.

### Regularization: Early Stopping

**Stop when validation loss stops improving.**

```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    save_model()
else:
    patience_counter += 1
    if patience_counter >= patience:
        stop_training()
```

Simple and effective.

> **Numerical Example: Early Stopping in Action**
>
> ```python
> import torch
> import torch.nn as nn
> import numpy as np
>
> torch.manual_seed(42)
>
> # Data that's easy to overfit
> n_train, n_val, n_features = 100, 100, 10
> X_train = torch.randn(n_train, n_features)
> true_w = torch.randn(n_features, 1)
> y_train = X_train @ true_w + torch.randn(n_train, 1) * 0.3
> X_val = torch.randn(n_val, n_features)
> y_val = X_val @ true_w + torch.randn(n_val, 1) * 0.3
>
> class Net(nn.Module):
>     def __init__(self):
>         super().__init__()
>         self.fc1 = nn.Linear(10, 128)
>         self.fc2 = nn.Linear(128, 64)
>         self.fc3 = nn.Linear(64, 1)
>     def forward(self, x):
>         return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
>
> model = Net()
> opt = torch.optim.Adam(model.parameters(), lr=0.01)
> train_losses, val_losses = [], []
>
> for epoch in range(150):
>     model.train()
>     opt.zero_grad()
>     loss = nn.MSELoss()(model(X_train), y_train)
>     loss.backward()
>     opt.step()
>     train_losses.append(loss.item())
>     model.eval()
>     with torch.no_grad():
>         val_losses.append(nn.MSELoss()(model(X_val), y_val).item())
>
> best_epoch = np.argmin(val_losses)
> print(f"Best epoch: {best_epoch}, Val loss: {val_losses[best_epoch]:.4f}")
> print(f"Final epoch: 149, Val loss: {val_losses[-1]:.4f}")
> ```
>
> **Output:**
> ```
> Epoch    Train Loss    Val Loss
>     0       21.9408     16.1734
>    25        0.4885      0.5259
>    50        0.0437      0.3270
>    78        0.0098      0.2645  <-- best
>   100        0.0038      0.2733
>   149        0.0006      0.2816
>
> Best epoch: 78, Val loss: 0.2645
> Final epoch: 149, Val loss: 0.2816
> ```
>
> **Interpretation:** Training loss keeps decreasing to near-zero, but validation loss hits a minimum at epoch 78 then starts rising—the classic overfitting pattern. Early stopping saves the model at epoch 78, preventing 71 epochs of wasted computation and a worse final model. The gap between train (0.0098) and val (0.2645) at the stopping point is already notable; by epoch 149, train is near-perfect (0.0006) but val is worse (0.2816).
>
> *Source: `slide_computations/module6_examples.py` - `demo_early_stopping()`*

### Diagnosing Overfitting

**Signs:**
- Training loss decreasing
- Validation loss increasing
- Large gap between train/val accuracy

**Solutions:**
- More data
- Dropout
- Early stopping
- Simpler architecture
- Data augmentation

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Lower training loss is always better" | If validation loss increases, you're overfitting. |
| "Dropout makes the network weaker" | Only during training. At test, all neurons active. |
| "Just use Adam defaults" | Tuning learning rate still helps. |
| "Train until loss is zero" | Zero training loss usually means severe overfitting. |

---

## 6.3 PyTorch Overview

### Why PyTorch?

- Dynamic computation graphs (easier debugging)
- Pythonic and intuitive
- Strong research community
- Seamless GPU support
- Great documentation

### Tensors and Autograd

**Tensors**: Like NumPy arrays but with GPU support and automatic differentiation.

```python
import torch

# Create tensors
x = torch.randn(3, 4)  # Random normal

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

**Autograd**: Automatic differentiation

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y.backward()
print(x.grad)  # dy/dx = 2x = 4 at x=2
```

### Building Models with nn.Module

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

Define layers in `__init__`, define forward pass in `forward`.

### The Training Loop

**This is the heart of neural network training. Learn this pattern:**

```python
model = MLP(784, 256, 10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()       # 1. Clear gradients
        output = model(data)        # 2. Forward pass
        loss = criterion(output, target)
        loss.backward()             # 3. Backward pass
        optimizer.step()            # 4. Update weights
```

**The pattern:**
1. `optimizer.zero_grad()` — Clear old gradients
2. `output = model(data)` — Forward pass
3. `loss.backward()` — Compute gradients
4. `optimizer.step()` — Update weights

### Evaluation Mode

```python
model.eval()  # Disables dropout

with torch.no_grad():  # No gradient tracking
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
```

**Key points:**
- `model.eval()` disables dropout (uses all neurons)
- `torch.no_grad()` saves memory

Always switch to eval mode for validation and testing!

### Complete MNIST Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def test():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return 100. * correct / len(test_loader.dataset)

# Run
for epoch in range(10):
    train(epoch)
    print(f'Epoch {epoch}: Test Accuracy: {test():.2f}%')
```

---

## Reflection Questions

1. Why couldn't the original perceptron learn XOR? Draw the XOR data and explain.

2. If a neural network can approximate any function with one hidden layer (Universal Approximation), why do we need deep networks?

3. Why do we need non-linear activation functions? What would happen with only linear activations?

4. A model has 10 million parameters. Is that a lot? What determines if this is appropriate?

5. Your training loss is decreasing but validation loss is increasing. What's happening and how do you fix it?

6. Why might Adam work better than vanilla SGD without tuning?

7. How is dropout similar to ensemble methods like Random Forest?

---

## Practice Problems

1. Calculate parameters for a [784, 512, 256, 128, 10] network

2. Identify overfitting from training curves (given a plot description)

3. Choose appropriate activation for: (a) hidden layers, (b) binary output, (c) multi-class output

4. Debug: "My training loss keeps increasing." Most likely cause?

5. Write the PyTorch training loop pattern from memory

---

## Chapter Summary

**Six key takeaways from Module 6:**

1. **Neural networks** = stacked layers + non-linear activations

2. **Depth** enables learning hierarchical features

3. **Backpropagation** computes gradients via chain rule

4. **Adam** is a good default optimizer; learning rate is the key hyperparameter

5. **Dropout + early stopping** prevent overfitting

6. **PyTorch pattern**: zero_grad → forward → backward → step

---

## What's Next

In Module 7, we tackle **Computer Vision & CNNs**:
- Convolutional layers for images
- Pooling and feature maps
- Famous architectures (LeNet, VGG, ResNet)
- Transfer learning

Same training principles, but specialized for images. Instead of fully connected layers, we'll use convolutional layers that exploit spatial structure.
