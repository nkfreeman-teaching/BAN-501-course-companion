# Module 7: Computer Vision & CNNs

## Introduction

Last module we learned neural network fundamentals—layers, activations, backpropagation, PyTorch. Today we specialize those concepts for images.

Images are everywhere in business: quality control in manufacturing, inventory management in retail, medical imaging in healthcare, document processing in finance. Computer vision has transformed all of these industries.

But images present unique challenges. A single photo is millions of numbers. Fully connected networks can't scale. And we need spatial awareness—a cat in the corner is still a cat, but its pixels are in completely different positions.

Convolutional Neural Networks solve these problems. By the end of today, you'll understand how CNNs work, and critically, you'll know how to leverage **transfer learning** so you don't have to train from scratch.

**Transfer learning works broadly**: Early CNN layers learn universal visual primitives (edges, textures) that transfer to any domain. Studies show ImageNet transfer helps on X-rays, satellite images, even art classification. Train from scratch only with massive domain data AND truly different image statistics—even then, ImageNet weights as initialization usually help.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** how images are represented as data (matrices, channels)
2. **Describe** why fully connected networks are inefficient for images
3. **Explain** the mechanics of convolutional layers and pooling
4. **Implement** a CNN in PyTorch for image classification
5. **Apply** transfer learning using pre-trained models
6. **Understand** modern CV applications (detection, segmentation, ViT)

---

## 7.1 Working with Images

### How Images Are Represented

Digital images are matrices of numbers.

**Grayscale**: 2D matrix (Height × Width). Each pixel is an intensity from 0 (black) to 255 (white).

**Color (RGB)**: 3D tensor (Height × Width × 3). Three channels—Red, Green, Blue—each with its own intensity matrix.

**Example**: A 224×224 color image
- Shape: (224, 224, 3)
- Total values: 224 × 224 × 3 = **150,528 numbers**

**Think of RGB as a layer cake**: Imagine three transparent sheets stacked on top of each other—one tinted red, one green, one blue. Each sheet has the same dimensions (Height × Width), and each position has an intensity value. When you look through all three layers at once, the colors combine to produce the full-color image. A pixel isn't just "one number"—it's a stack of three numbers, one from each color channel. This stacking concept extends to CNNs: as you go deeper, instead of 3 channels (RGB), you might have 64, 128, or 512 "feature channels"—each representing a different learned feature like edges, textures, or shapes.

**PyTorch convention**: (Batch, Channels, Height, Width)—NCHW format.

```python
from PIL import Image
import numpy as np

img = Image.open('photo.jpg')
img_array = np.array(img)
print(f"Shape: {img_array.shape}")  # (Height, Width, Channels)
```

### ImageNet: The Benchmark That Changed Everything

| Year | Winner | Top-5 Error | Significance |
|------|--------|-------------|--------------|
| 2010 | Traditional | 28.2% | Pre-deep learning |
| 2012 | AlexNet | 16.4% | CNN breakthrough |
| 2015 | ResNet | 3.6% | Beat humans (~5%) |

In 2012, AlexNet—a convolutional neural network—crushed the competition. Error dropped from 28% to 16%. That's not incremental improvement; that's a paradigm shift.

By 2015, ResNet beat human performance on ImageNet classification.

### Why Fully Connected Networks Fail

**Problem 1: Too many parameters**
- 224×224×3 input with 1000 hidden neurons
- = 150 million parameters in first layer alone!
- Impossible to train, will overfit immediately

**Problem 2: No spatial understanding**
- Fully connected layers treat each pixel independently
- A cat in the corner has completely different pixel positions than a cat in the center
- The network can't generalize

**The solution**: Convolutional Neural Networks

> **Numerical Example: Parameter Explosion in Fully Connected Networks**
>
> ```python
> # Calculate first FC layer parameters for different image sizes
> image_configs = [
>     ("MNIST", 28, 28, 1),      # Grayscale
>     ("CIFAR-10", 32, 32, 3),   # Color
>     ("ImageNet", 224, 224, 3), # Standard photo
> ]
> hidden_neurons = 1000
>
> for name, h, w, c in image_configs:
>     input_features = h * w * c
>     parameters = input_features * hidden_neurons + hidden_neurons
>     print(f"{name}: {input_features:,} inputs → {parameters:,} parameters")
> ```
>
> **Output:**
> ```
> MNIST: 784 inputs → 785,000 parameters
> CIFAR-10: 3,072 inputs → 3,073,000 parameters
> ImageNet: 150,528 inputs → 150,529,000 parameters
> ```
>
> **Interpretation:** A single FC layer on a 224×224 image requires 150 million parameters—just to connect inputs to the first hidden layer! This is why FC networks are impractical for images. CNNs achieve the same task with ~100x fewer parameters through local connectivity and weight sharing.
>
> *Source: `slide_computations/module7_examples.py` - `demo_fc_parameter_explosion()`*

**Why position matters**: A fully connected network treats each pixel independently—"pixel 1,000 is orange" vs. "pixel 50,000 is orange" are completely different inputs. To recognize cats anywhere, it would need examples at every possible position (billions of configurations). CNNs solve this with weight sharing: the same filter scans all positions, so learning to detect a cat's eye at one position automatically applies everywhere.

---

## 7.2 Convolutional Neural Networks

### The Convolution Operation

Instead of connecting every input to every output, we slide a small filter across the image.

**The operation:**
1. Take a small filter (e.g., 3×3)
2. Slide it across the image
3. At each position, compute dot product of filter and patch
4. Output is a "feature map"

**Key parameters:**
- **Filter size**: 3×3 or 5×5 typical
- **Stride**: How many pixels to move (1 or 2)
- **Padding**: Zeros around edges to control output size
- **Number of filters**: Each learns a different feature

**The sliding window intuition**: Imagine holding a magnifying glass (the filter) over a photograph (the input image). You look at a small 3×3 patch, write down a summary number, then slide the magnifying glass one position to the right and repeat. When you reach the edge, you move down one row and start from the left again. The "summary number" is the dot product: multiply each pixel by the corresponding filter weight and sum them all. After scanning the entire image, you've produced a new, smaller image called a "feature map"—where each position tells you "how strongly does this local region match what this filter is looking for?"

> **Numerical Example: Convolution by Hand**
>
> ```python
> import numpy as np
>
> # 5×5 image with bright center
> image = np.array([
>     [10, 10, 10, 10, 10],
>     [10, 50, 50, 50, 10],
>     [10, 50, 100, 50, 10],
>     [10, 50, 50, 50, 10],
>     [10, 10, 10, 10, 10],
> ])
>
> # Horizontal edge detector
> filter_h = np.array([[-1, -2, -1],
>                      [ 0,  0,  0],
>                      [ 1,  2,  1]])
>
> # Convolve center position (1,1)
> patch = image[1:4, 1:4]  # Extract 3×3 patch
> result = np.sum(patch * filter_h)
> print(f"Patch:\n{patch}")
> print(f"Element-wise product sum: {result}")
> ```
>
> **Output:**
> ```
> Patch:
> [[50 50 50]
>  [50 100 50]
>  [50 50 50]]
> Element-wise product sum: 0
> ```
>
> **Interpretation:** The center patch is symmetric top-to-bottom, so the horizontal edge detector outputs 0 (no horizontal edge). At the top of the image where intensity changes from 10→50, the filter outputs +170, detecting the edge. The filter automatically responds to edges wherever they occur.
>
> *Source: `slide_computations/module7_examples.py` - `demo_convolution_by_hand()`*

### Multi-Channel Convolution

**Key insight: A "3×3 filter" on an RGB image is actually a 3×3×3 tensor.**

When we say "3×3 filter," we're describing the spatial dimensions. But the filter must match the depth of the input.

For an RGB image with 3 channels:
- Filter shape: 3 × 3 × 3 = **27 weights** (plus 1 bias)
- Each channel (R, G, B) has its own 3×3 slice

**How the computation works:**

```
At each spatial position:
1. Extract the 3×3×3 patch from the input
2. Multiply element-wise with the 3×3×3 filter (27 multiplications)
3. Sum ALL 27 products + bias → ONE output value
```

**Multiple filters → Multiple output channels:**

If we want 64 output channels, we need 64 separate filters, each with shape 3×3×3. Total parameters: 64 × (27 + 1) = **1,792**.

**The "deep handshake" intuition**: A filter doesn't just look at one color—it reaches through all input channels simultaneously, like a hand reaching through stacked sheets to grab information from every layer at once. If the input has 3 channels (RGB), the filter has 3 slices. If the input has 64 feature channels from a previous layer, the filter has 64 slices. Each slice learns what to look for in that specific input channel, and the results are summed into a single output value. This is why deeper layers can detect complex combinations: a filter might learn "look for vertical edges in channel 12 AND horizontal edges in channel 37" by having strong weights in those specific filter slices.

```python
conv = nn.Conv2d(
    in_channels=3,      # RGB input
    out_channels=64,    # Number of filters
    kernel_size=3,      # 3×3 filter
    stride=1,
    padding=1
)
```

> **Numerical Example: Output Size Formula in Action**
>
> ```python
> # Formula: output = (W - K + 2P) / S + 1
> # W=input, K=kernel, P=padding, S=stride
>
> # Trace 32×32 image through 3 conv+pool blocks
> size = 32
> print(f"Input: {size}×{size}")
>
> for i in range(3):
>     # Conv with padding=1 preserves size
>     size = (size - 3 + 2*1) // 1 + 1
>     print(f"After Conv{i+1}: {size}×{size}")
>     # MaxPool 2×2 halves dimensions
>     size = size // 2
>     print(f"After Pool{i+1}: {size}×{size}")
> ```
>
> **Output:**
> ```
> Input: 32×32
> After Conv1: 32×32
> After Pool1: 16×16
> After Conv2: 16×16
> After Pool2: 8×8
> After Conv3: 8×8
> After Pool3: 4×4
> ```
>
> **Interpretation:** With padding=1 on 3×3 convolutions, spatial dimensions are preserved. Each 2×2 max pool halves the dimensions. A 32×32 image becomes 4×4 after three pool layers—a 64x reduction in spatial positions, concentrating information into fewer, more meaningful locations.
>
> *Source: `slide_computations/module7_examples.py` - `demo_output_size_formula()`*

### What Filters Learn

Filters automatically learn features through training:

- **Early layers**: Edges, colors, simple textures
- **Middle layers**: Textures, patterns, shapes
- **Deep layers**: Object parts, semantic concepts

The first layer might learn vertical edges, horizontal edges, color gradients. The second combines those into textures. The third combines textures into shapes. This is **hierarchical feature learning**.

**What an edge detector actually looks like**: A horizontal edge detector might have weights like:
```
[-1, -1, -1]
[ 0,  0,  0]
[ 1,  1,  1]
```
This filter responds strongly when it sees dark pixels above and bright pixels below (a horizontal edge). The negative weights say "penalize brightness here," the positive weights say "reward brightness here," and zeros mean "don't care." When this filter slides over a horizontal edge in the image, the dark-above-light-below pattern produces a large positive output. Over uniform regions, positives and negatives cancel out. The network learns these patterns automatically through backpropagation—we don't hand-design them.

**Hierarchy emerges automatically**: You don't design what each layer learns. Early layers only see raw pixels (can only learn edges); deep layers receive processed representations (can combine into complex features). When researchers visualize trained networks, they find edges in layer 1, textures in layers 2-3, object parts in mid-layers—discovered, not programmed.

### Pooling Layers

After convolution, we reduce spatial dimensions with pooling.

**Max Pooling**: Take maximum value in each patch
- Reduces spatial dimensions (224 → 112 → 56...)
- Adds translation invariance—slight shifts don't change output
- Keeps strongest activations

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 224×224 → 112×112
```

A 2×2 max pool with stride 2 halves each dimension.

**Why max pooling dominates over average pooling**: Imagine a feature map where most values are near zero (no edge detected) but one position has a strong response (edge found!). Max pooling preserves that strong signal—"there's definitely an edge somewhere in this 2×2 region." Average pooling would dilute it with the zeros: "there's maybe a weak edge here." For detecting features, we care that a feature is *present*, not its average strength. Exception: Global Average Pooling at the very end of a network (averaging across the entire spatial dimension) works well because by that point, strong features have already been isolated.

> **Numerical Example: Pooling Dimension Tracking**
>
> ```python
> # Track ImageNet-standard 224×224 through pooling layers
> size = 224
> print(f"Input: {size}×{size} = {size*size:,} positions")
>
> for i in range(5):
>     size = size // 2
>     reduction = (224*224) / (size*size)
>     print(f"Pool {i+1}: {size}×{size} = {size*size:,} positions ({reduction:.0f}x smaller)")
> ```
>
> **Output:**
> ```
> Input: 224×224 = 50,176 positions
> Pool 1: 112×112 = 12,544 positions (4x smaller)
> Pool 2: 56×56 = 3,136 positions (16x smaller)
> Pool 3: 28×28 = 784 positions (64x smaller)
> Pool 4: 14×14 = 196 positions (256x smaller)
> Pool 5: 7×7 = 49 positions (1024x smaller)
> ```
>
> **Interpretation:** Each 2×2 max pool halves each dimension, quartering the spatial positions. After 5 pooling layers, 50,176 positions compress to just 49—over 1000x reduction. This progressive compression forces the network to distill spatial information into increasingly abstract "what is here" representations rather than "where exactly is it."
>
> *Source: `slide_computations/module7_examples.py` - `demo_pooling_dimension_tracking()`*

### Classic CNN Pattern

![CNN Pipeline](../assets/module7/cnn_pipeline.png)

**Reading the diagram**: This shows the classic CNN architecture pattern as a data flow pipeline. Data enters from the left and flows through repeated blocks: **Conv** (blue) applies learned filters to detect features, **ReLU** (green) introduces non-linearity by zeroing negative values, and **Pool** (purple) reduces spatial dimensions. This Conv→ReLU→Pool pattern typically repeats 2-5 times, with each cycle detecting higher-level features while shrinking the spatial dimensions. After the final pooling layer, **Flatten** (orange) reshapes the 2D feature maps into a 1D vector, which feeds into **FC** (red)—a fully connected layer that makes the final classification. The key insight: early stages are "looking" (detecting edges, textures, shapes), while the final FC layer is "deciding" (combining features into class predictions).

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### Parameter Efficiency

**For 32×32 RGB image, 64 outputs:**

| Layer Type | Parameters |
|-----------|------------|
| Fully Connected | 196,672 |
| Conv2d (3×3) | 1,792 |

**~100x fewer parameters!**

Why?
1. **Local connectivity**: Each neuron connects only to a small patch
2. **Weight sharing**: Same filter applied everywhere

> **Numerical Example: CNN vs FC Parameter Comparison**
>
> ```python
> # Task: 32×32×3 input → 64 output features
> input_size = 32 * 32 * 3  # 3,072
> output_channels = 64
>
> # Fully connected
> fc_params = input_size * output_channels + output_channels
> print(f"FC layer: {fc_params:,} parameters")
>
> # Conv2d (3×3 filter)
> conv_params = (3 * 3 * 3) * output_channels + output_channels
> print(f"Conv layer: {conv_params:,} parameters")
> print(f"Ratio: {fc_params / conv_params:.1f}x fewer with CNN")
> ```
>
> **Output:**
> ```
> FC layer: 196,672 parameters
> Conv layer: 1,792 parameters
> Ratio: 109.8x fewer with CNN
> ```
>
> **Interpretation:** For the same input→output mapping, CNNs use ~110x fewer parameters. The FC layer needs a separate weight for every input-output pair. The CNN reuses 27 weights (3×3×3 filter) across all 1,024 spatial positions. This efficiency enables training on limited data and reduces overfitting risk.
>
> *Source: `slide_computations/module7_examples.py` - `demo_cnn_vs_fc_parameters()`*

### Historical Architectures

**AlexNet (2012)**: 8 layers, ReLU, dropout, GPU training. The breakthrough.

**VGG (2014)**: 16-19 layers, all 3×3 convolutions. Showed depth matters.

**ResNet (2015)**: Skip connections enabling 150+ layers.

### Skip (Residual) Connections

**The problem**: Very deep networks suffer from vanishing gradients.

**The solution**: Add the input directly to the output.

$$Output = F(x) + x$$

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return torch.relu(out)
```

If the network can't improve on the input, it can at least pass it through unchanged. This creates direct paths for gradients and enables training 100+ layer networks.

**The "highway on-ramp" analogy**: In a deep network without skip connections, gradients must travel through every layer sequentially—like driving through 100 stoplights to get across town. Each layer can shrink the gradient (vanishing) or explode it. Skip connections add highway on-ramps: gradients can take the direct route (the skip) or the scenic route (through the layers), or both. Even if the scenic route has problems, the highway ensures signals get through. During training, early layers actually receive useful gradient information because it doesn't have to survive passage through dozens of potentially problematic layers.

**Skip connection trade-offs**: Memory overhead (must store earlier activations) and architectural constraints (dimensions must match, may need 1×1 convolutions). In shallow networks (3-5 layers), minimal benefit—skip connections solve a deep network problem. For networks >10 layers, skip connections almost always help and are now considered essential in modern architectures.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "CNNs only work for images" | CNNs work on any grid data: audio, time series, etc. |
| "Deeper is always better" | Without skip connections, very deep nets fail. Architecture matters. |
| "You need to design CNNs from scratch" | Transfer learning is usually better. |

---

## 7.3 Transfer Learning

### The Core Idea

Pre-trained ImageNet models learned **general visual features**: edges, textures, shapes, patterns. These features are useful for almost any image task!

**Learning to see before learning your task**: A child doesn't learn "what is a cat" from scratch—they already know how to see edges, shapes, colors, and textures from years of visual experience. Teaching them "cat" is just connecting those existing visual concepts to a new label. Transfer learning works the same way: ImageNet training teaches a network "how to see" (edges, textures, shapes, object parts), and your task-specific training just connects those visual features to your labels. That's why 500 images can work: you're not teaching the network to see—you're just teaching it what to call things it can already perceive.

**Two approaches:**
1. **Feature extraction**: Freeze pre-trained layers, train only new classifier
2. **Fine-tuning**: Train all layers, but with lower learning rate for pre-trained layers

This is how most real-world computer vision is done. You rarely train from scratch anymore.

### Feature Extraction

**Freeze pre-trained layers, train only new classifier.**

```python
import torchvision.models as models

model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train new classifier
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
```

The pre-trained ResNet extracts features. You just train a simple classifier on top.

### Fine-Tuning

**Train pre-trained layers with lower learning rate.**

```python
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Different learning rates
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
])
```

Pre-trained layers get smaller learning rate (they're already good). New layers get larger learning rate.

### When to Use Which

| Dataset Size | Similarity to ImageNet | Approach |
|-------------|------------------------|----------|
| Small | High | Feature extraction |
| Small | Low | Light fine-tuning |
| Large | High | Fine-tuning |
| Large | Low | Train from scratch |

**Example**: You have 500 X-ray images. Train from scratch or transfer learning?

Transfer learning! 500 images isn't enough to train from scratch. Even though X-rays look different from ImageNet photos, early-layer features (edges, textures) are still useful.

**How similar is "similar enough"?** There's no bright line—empirically test: train a classifier on frozen pre-trained features vs. random features. If pre-trained beats random, transfer helps. Even domains that seem "completely different" (medical imaging, industrial defects) usually benefit. Start with transfer learning, try fine-tuning if unsatisfactory, consider training from scratch only with millions of examples AND truly foreign image statistics.

> **Numerical Example: Transfer Learning vs Random Features**
>
> ```python
> # Simulate: classify images with limited training data
> # Pre-trained CNN extracts meaningful features
> # Random CNN outputs noise
>
> from sklearn.linear_model import LogisticRegression
>
> train_sizes = [25, 50, 100, 200, 500]
> for n in train_sizes:
>     # Train classifier on pre-trained features
>     acc_pretrained = train_on_features(X_pretrained[:n], y[:n])
>     # Train classifier on random features
>     acc_random = train_on_features(X_random[:n], y[:n])
>     print(f"n={n}: Pre-trained={acc_pretrained:.0%}, Random={acc_random:.0%}")
> ```
>
> **Output:**
> ```
> n=25:  Pre-trained=40%, Random=12%
> n=50:  Pre-trained=61%, Random=18%
> n=100: Pre-trained=74%, Random=18%
> n=200: Pre-trained=82%, Random=25%
> n=500: Pre-trained=89%, Random=19%
> ```
>
> **Interpretation:** With only 25 training examples, pre-trained features achieve 40% accuracy vs 12% for random (5-class chance = 20%). The gap widens with more data. Random features plateau near chance because they contain no useful information—the classifier is guessing. Pre-trained features capture real visual patterns that generalize to new images.
>
> *Source: `slide_computations/module7_examples.py` - `demo_transfer_learning_comparison()`*

### Business Value of Transfer Learning

- **Cost savings**: Days of training → hours
- **Data efficiency**: Good results with hundreds of images (not millions)
- **Time to deployment**: Quick proof-of-concept
- **No massive compute**: Fine-tuning on a laptop is possible

---

## 7.4 Modern Vision Applications

### Object Detection

**Task**: Find objects AND their locations (bounding boxes)

Not just "there's a dog" but "there's a dog at coordinates (x, y, w, h)."

**Key architectures:**
- **YOLO**: Fast, single-pass detection ("You Only Look Once")
- **Faster R-CNN**: Two-stage, more accurate but slower

**Applications**: Autonomous vehicles, security cameras, retail inventory

### Image Segmentation

**Semantic segmentation**: Label every pixel with a class (road, car, person)

**Instance segmentation**: Separate individual objects (this car vs that car)

**Key architecture**: U-Net—encoder-decoder with skip connections

**Applications**: Medical imaging, autonomous driving, photo editing

### Vision Transformers (ViT)

The latest revolution: apply transformer architecture to images.

**How it works:**
1. Split image into 16×16 patches
2. Flatten patches into sequences
3. Apply transformer encoder (same architecture as NLP!)

**Why it matters:**
- State-of-the-art on many benchmarks
- Unified architecture for vision AND language
- Enables CLIP, DALL-E, multimodal AI

**Patches as visual words**: In NLP, transformers process sequences of word tokens. ViT creates a similar setup for images: each 16×16 patch becomes a "visual word." A 224×224 image becomes a sequence of (224/16)² = 196 tokens. The transformer then asks "how does patch 45 relate to patch 120?" just like it asks "how does word 3 relate to word 15?" in text. This unification is powerful: the same attention mechanism that learns "the word 'cat' relates to 'furry'" can learn "this patch of fur relates to that patch showing ears." It's why models like CLIP can connect images and text—they're processing both as sequences of tokens.

### Business Applications

| Industry | Application |
|----------|-------------|
| Retail | Inventory monitoring, checkout-free stores |
| Manufacturing | Defect detection, quality control |
| Healthcare | Radiology, pathology analysis |
| Agriculture | Crop monitoring, disease detection |

---

## Reflection Questions

1. An image is 1000×1000 pixels RGB. How many input features? Why is this problematic for fully connected networks?

2. If you shift a cat 10 pixels to the right, how would a fully connected network's perception change vs. a CNN?

3. A 3×3 conv filter has 9 weights per channel. How does this compare to fully connected for the same output?

4. After 3 max pooling layers of 2×2, what happens to a 224×224 image?

5. How do skip connections help train very deep networks?

6. You have 500 X-ray images. Train from scratch or transfer learning? Why?

7. Why fine-tune later layers before earlier layers?

---

## Practice Problems

1. Calculate output size: 64×64 input, 3×3 kernel, stride=1, padding=0

2. Calculate parameters: Conv2d with in_channels=32, out_channels=64, kernel_size=3

3. Design a CNN for 28×28 grayscale images (MNIST) with 3 conv layers

4. Set up transfer learning code for a 5-class classification problem using ResNet18

5. Explain why a 7×7 filter might be replaced by two 3×3 filters

---

## Chapter Summary

**Six key takeaways from Module 7:**

1. **Images** are high-dimensional; FC networks don't scale

2. **CNNs** use local filters with weight sharing (100x fewer parameters)

3. **Pooling** reduces dimensions and adds translation invariance

4. **Skip connections** enable training very deep networks

5. **Transfer learning** is usually better than training from scratch

6. **Modern CV**: detection, segmentation, Vision Transformers

---

## What's Next

In Module 8, we tackle **Natural Language Processing**:
- Text as sequences
- Word embeddings
- Transformers and attention
- Pre-trained language models

Vision Transformers connect both domains—the same architecture that powers GPT and BERT can also process images!
