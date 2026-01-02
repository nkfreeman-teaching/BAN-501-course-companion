# Module 3: Classification Methods

## Introduction

We've covered a lot of ground—foundations in Module 1, regression in Module 2. Now we move to classification, which is arguably even more prevalent in business applications.

Think about the decisions businesses make every day: Should we approve this loan? Is this transaction fraudulent? Will this customer cancel their subscription? Is this email spam? These are all classification problems—predicting a category, not a number.

The concepts extend to **multiclass classification**: logistic regression uses softmax instead of sigmoid; decision trees handle it naturally; evaluation uses per-class precision/recall and NxN confusion matrices. The fundamentals transfer directly—the mechanics get more complex but the reasoning stays the same.

This module covers four major topics: logistic regression (extending regression concepts to classification), decision trees (intuitive classifiers that set us up for ensemble methods), handling imbalanced data (because when 99% of transactions are legitimate, accuracy is meaningless), and hyperparameter optimization.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the mechanics and interpretation of logistic regression, including log odds and probability
2. **Build** and interpret decision tree classifiers, understanding their tendency to overfit
3. **Apply** appropriate techniques for handling imbalanced classification problems
4. **Use** hyperparameter optimization techniques to improve model performance
5. **Select** appropriate evaluation metrics based on business context

---

## 3.1 Logistic Regression

### Three Components: Logistic Regression

Connecting to the framework from Module 2:

| Component | Logistic Regression |
|-----------|---------------------|
| **Decision Model** | $P(Y=1) = \sigma(\beta_0 + \beta_1 x_1 + ...)$ — sigmoid of linear combination |
| **Quality Measure** | Cross-entropy (log loss) — penalizes confident wrong predictions |
| **Update Method** | Gradient descent on log-likelihood |

The decision model changes from a line to a sigmoid curve, and the quality measure changes from SSE to cross-entropy—but the overall structure is identical to linear regression.

### Why Linear Regression Fails for Classification

Binary outcomes are coded as 0 or 1. If we fit a line, predictions can be less than 0 or greater than 1. "There's a -15% chance of churn" is meaningless.

**The solution**: Transform the output so it's always between 0 and 1.

Other functions map to (0,1)—probit, scaled tanh—but sigmoid has unique advantages: its derivative is expressible in terms of the output (efficient gradients), its inverse is the logit (clean coefficient interpretation as log-odds), and it arises from maximum entropy principles. Tools and practices are standardized around it.

### The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where $z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$

**Key properties:**
- Output is always between 0 and 1—valid probability
- S-shaped curve—small changes in x have the biggest effect near 0.5
- At z=0, output is exactly 0.5

The math:
- When z is large and positive: $e^{-z} \to 0$, so $\sigma(z) \to 1$
- When z is large and negative: $e^{-z} \to \infty$, so $\sigma(z) \to 0$
- When z = 0: $e^{0} = 1$, so $\sigma(0) = 0.5$

**Why this particular S-shape?** The sigmoid isn't arbitrary—it's the only function that: (1) maps any real number to (0,1), (2) is symmetric around 0.5, and (3) has a derivative expressible in terms of itself (making gradient descent efficient). Think of z as a "confidence score": strongly negative z means the model is confident in class 0, strongly positive means confident in class 1, and z near 0 means the model is uncertain. The sigmoid converts this confidence into a probability. The "action zone" where probability changes meaningfully is roughly z ∈ [-4, 4]—outside this range, the model is essentially certain.

> **Numerical Example: Sigmoid Function in Action**
>
> ```python
> import numpy as np
>
> def sigmoid(z):
>     return 1 / (1 + np.exp(-z))
>
> z_values = [-6, -4, -2, 0, 2, 4, 6]
> for z in z_values:
>     prob = sigmoid(z)
>     print(f"z = {z:>3}: σ(z) = {prob:.4f}")
> ```
>
> **Output:**
> ```
> z =  -6: σ(z) = 0.0025
> z =  -4: σ(z) = 0.0180
> z =  -2: σ(z) = 0.1192
> z =   0: σ(z) = 0.5000
> z =   2: σ(z) = 0.8808
> z =   4: σ(z) = 0.9820
> z =   6: σ(z) = 0.9975
> ```
>
> **Interpretation:** At z=0, probability is exactly 0.5 (maximum uncertainty). Moving to z=±4 brings probability within 2% of the extremes (0 or 1). At z=±6, the model is 99.75% confident. The "interesting range" where probability changes meaningfully is roughly z ∈ [-4, 4].
>
> *Source: `slide_computations/module3_examples.py` - `demo_sigmoid_function()`*

### Understanding Odds and Log Odds

**Step 1: Odds**

$$Odds = \frac{P(Y=1)}{P(Y=0)} = \frac{p}{1-p}$$

If P(churn) = 0.75, odds = 0.75/0.25 = 3. "3 to 1 odds of churning."

**Step 2: Log Odds (Logit)**

$$\log\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + ...$$

**Key insight**: The log odds ARE linear in the predictors. This is where the "linear" in logistic regression comes from.

**Example**: Model: $\log(odds) = -2 + 0.5 \times age$

| Age | Log Odds | Odds | Probability |
|-----|----------|------|-------------|
| 0 | -2 | $e^{-2}$ ≈ 0.14 | 12% |
| 20 | 8 | $e^8$ ≈ 2981 | 99.97% |
| 30 | 13 | $e^{13}$ ≈ 442,413 | ≈100% |

Log odds change linearly, but probabilities don't. That's the magic of the logit transform.

**Why log odds?** They provide interpretable coefficients (each β is "change in log-odds per unit"), unbounded range (the linear predictor can take any value while output stays bounded 0-1), and additive effects (effects of multiple variables sum in log-odds space, unlike in probability space). The transformation connects linear models to probability naturally.

### Coefficient Interpretation

**The coefficient $\beta_1$**: Change in log odds for a one-unit increase in $x_1$.

**The odds ratio $e^{\beta_1}$**: Multiplicative change in odds.

**Example:**
- If $\beta_1 = 0.5$, then $e^{0.5} \approx 1.65$
- "Each unit increase in X increases the odds by 65%"

**Converting to probability:**
1. Calculate log-odds: $z = \beta_0 + \beta_1 x_1 + ...$
2. Apply sigmoid: $P(Y=1) = \frac{1}{1 + e^{-z}}$

### Is It Regression or Classification?

| Aspect | Answer |
|--------|--------|
| Name | "Regression" (historical reasons) |
| What it models | Probability (continuous 0-1) |
| What we use it for | Classification (discrete classes) |
| How | Apply a threshold to the probability |

**Key insight**: Logistic regression IS a regression model (predicts continuous probability), but we USE it for classification by thresholding.

Probabilities give crucial flexibility over hard class predictions: threshold flexibility (adjust without retraining when costs change), ranking and prioritization ("which 100 customers are most likely to churn?"), confidence communication (P=0.95 vs P=0.55 both classify as positive but represent different confidence), and risk quantification (expected value calculations require probabilities). In business, you almost always benefit from probabilities.

### Decision Thresholds

**The default threshold of 0.5 is often NOT optimal!**

**Example - Fraud Detection:**
- Cost of missing fraud (false negative): $10,000
- Cost of investigating non-fraud (false positive): $100

With asymmetric costs, lower the threshold—catch more fraud, accept more false alarms.

**Cost-based threshold formula**: $t^* = \frac{C_{FP}}{C_{FP} + C_{FN}}$. For fraud costing $10,000 (FN) and investigation costing $100 (FP): threshold ≈ 100/(100+10000) ≈ 0.01—predict fraud for anyone above 1% probability! This assumes well-calibrated probabilities; verify calibration first. Alternative: Youden's J statistic (maximize TPR-FPR) when costs are unknown.

**Why does this formula work?** It comes from minimizing expected cost. At the threshold, the expected cost of a false positive equals the expected cost of a false negative:

$$P(\text{actually positive}) \times C_{FN} = P(\text{actually negative}) \times C_{FP}$$

Solving for the probability threshold where you're indifferent between predicting positive or negative gives the formula. When $C_{FN}$ is much larger than $C_{FP}$ (missing fraud is expensive), the threshold drops close to zero—you flag almost anything suspicious. When costs are equal, the threshold is 0.5 (the default).

> **Numerical Example: Cost-Based Threshold Selection**
>
> ```python
> # Fraud detection costs
> cost_fp = 50   # Investigation cost
> cost_fn = 500  # Missed fraud cost
>
> optimal_threshold = cost_fp / (cost_fp + cost_fn)
> print(f"Optimal threshold: {optimal_threshold:.3f}")
>
> # Effect on predictions
> probabilities = [0.05, 0.10, 0.20, 0.40, 0.60]
> for p in probabilities:
>     default = "Flag" if p >= 0.5 else "Clear"
>     optimal = "Flag" if p >= optimal_threshold else "Clear"
>     print(f"P={p:.2f}: Default={default}, Optimal={optimal}")
> ```
>
> **Output:**
> ```
> Optimal threshold: 0.091
> P=0.05: Default=Clear, Optimal=Clear
> P=0.10: Default=Clear, Optimal=Flag
> P=0.20: Default=Clear, Optimal=Flag
> P=0.40: Default=Clear, Optimal=Flag
> P=0.60: Default=Flag, Optimal=Flag
> ```
>
> **Interpretation:** When fraud costs 10x more than investigation, we flag any transaction with P(fraud) > 9.1%. A transaction with 10% fraud probability gets flagged—it's worth investigating because the expected loss from missing fraud ($500 × 0.1 = $50) equals the investigation cost ($50).
>
> *Source: `slide_computations/module3_examples.py` - `demo_cost_based_threshold()`*

**Threshold effects:**

| Lower Threshold | Higher Threshold |
|-----------------|------------------|
| More positive predictions | Fewer positive predictions |
| Higher recall | Higher precision |
| Lower precision | Lower recall |
| Fewer false negatives | Fewer false positives |

### ROC Curves and AUC

For each possible threshold:
1. Calculate True Positive Rate: $TPR = \frac{TP}{TP + FN}$
2. Calculate False Positive Rate: $FPR = \frac{FP}{FP + TN}$
3. Plot the point

**AUC interpretation:**
- 0.5 = Random guessing
- 1.0 = Perfect separation
- 0.8 = "80% chance that a randomly chosen positive ranks higher than a randomly chosen negative"

**Note**: AUC ≠ accuracy. AUC measures ranking ability across all thresholds.

**Ranking ability** means correctly ordering examples by likelihood—higher-risk items get higher scores—even if actual probability values are wrong. This matters for resource allocation ("call top 100 highest-risk customers"), campaign targeting (top decile by response rate), and prioritization (fraud investigators review by score). A model with AUC=0.9 and poor calibration is often more useful than AUC=0.6 with perfect calibration—you can recalibrate using Platt scaling or isotonic regression; you can't easily fix ranking ability.

**The "threshold dial" intuition:** Imagine a dial you can turn from 0 to 1. As you turn it up (higher threshold), you become more selective—fewer positive predictions, but the ones you make are more confident. As you turn it down (lower threshold), you cast a wider net—catching more true positives but also more false alarms. The ROC curve shows you every position of this dial simultaneously. A good model gives you attractive options along the curve; a poor model forces you to choose between bad options (high FPR or low TPR).

> **Numerical Example: Building an ROC Curve Step by Step**
>
> ```python
> import numpy as np
>
> # Small dataset: 4 positive, 6 negative
> true_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
> pred_proba  = np.array([0.95, 0.85, 0.70, 0.45, 0.60, 0.40, 0.35, 0.20, 0.10, 0.05])
>
> thresholds = [1.0, 0.70, 0.45, 0.0]
> for thresh in thresholds:
>     pred = (pred_proba >= thresh).astype(int)
>     tp = np.sum((pred == 1) & (true_labels == 1))
>     fp = np.sum((pred == 1) & (true_labels == 0))
>     tpr = tp / 4  # 4 actual positives
>     fpr = fp / 6  # 6 actual negatives
>     print(f"Threshold {thresh:.2f}: TPR={tpr:.2f}, FPR={fpr:.2f}")
> ```
>
> **Output:**
> ```
> Threshold 1.00: TPR=0.00, FPR=0.00
> Threshold 0.70: TPR=0.75, FPR=0.00
> Threshold 0.45: TPR=1.00, FPR=0.17
> Threshold 0.00: TPR=1.00, FPR=1.00
> ```
>
> **Interpretation:** The ROC curve plots these (FPR, TPR) points as threshold varies. At threshold 0.70, we achieve TPR=75% with zero false positives—an excellent operating point. Lowering to 0.45 catches all positives but introduces one false positive (FPR=17%). The AUC measures the area under this curve; higher is better.
>
> *Source: `slide_computations/module3_examples.py` - `demo_roc_curve_construction()`*

**Choosing optimal threshold:**
- Youden's J statistic: Maximize (TPR - FPR)
- Cost-based: Minimize expected cost given FP/FN costs
- Precision-Recall trade-off: Use PR curve for imbalanced data

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc

log_reg = LogisticRegression(penalty='l2', C=1.0, random_state=42)
log_reg.fit(X_train, y_train)

# Get probabilities
y_proba = log_reg.predict_proba(X_test)[:, 1]

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Optimal threshold (Youden's J)
optimal_idx = (tpr - fpr).argmax()
optimal_threshold = thresholds[optimal_idx]

# Interpret coefficients as odds ratios
for feature, coef in zip(feature_names, log_reg.coef_[0]):
    odds_ratio = np.exp(coef)
    print(f"{feature}: odds ratio = {odds_ratio:.3f}")
```

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Logistic regression outputs are well-calibrated probabilities" | Outputs may need calibration (Platt scaling, isotonic regression) for reliable probability estimates. |
| "Higher AUC always means better model" | AUC ignores calibration and threshold choice. A model with lower AUC but better calibration might be preferable. |
| "Logistic regression requires linear relationships" | It requires linearity in LOG ODDS, not probability. Add polynomial terms for non-linear relationships. |
| "Logistic regression can't handle multiple classes" | Multinomial logistic regression extends to multiple classes (one-vs-rest or softmax). |

---

## 3.2 Decision Trees (CART)

### Three Components: Decision Trees

| Component | Decision Trees |
|-----------|----------------|
| **Decision Model** | Tree of if-then rules — follow branches based on feature thresholds |
| **Quality Measure** | Gini impurity or entropy — measures class mixture in nodes |
| **Update Method** | Greedy recursive splitting — find best split at each node |

**Key difference**: We're not doing gradient descent. Trees use a greedy algorithm that builds one split at a time.

### Decision Tree Intuition

Imagine you're a loan officer:
- First: Is income > $50,000?
  - Yes → Check debt-to-income ratio
  - No → Check employment history...

**Decision trees formalize this intuitive process.** They automatically learn which questions to ask, in what order, and what thresholds to use.

The tree picks the feature and threshold that best separates classes (maximally reduces impurity). This is a greedy algorithm—locally best splits without looking ahead. The first feature is often important but not always "most important": a feature might matter most after controlling for another, or correlated features might be interchanged. Feature importance scores (aggregating across all nodes) are more reliable than just the root split.

### Splitting Criteria

**Gini Impurity** (scikit-learn default):

$$Gini = 1 - \sum_{i=1}^{C} p_i^2$$

Where $p_i$ is the proportion of class $i$ in the node.

- Gini = 0: Pure node (all same class)
- Gini = 0.5: Maximum impurity for binary (50-50)

**Gini as "certainty of a random guess":** If you randomly pick an example from a node and randomly assign it a class based on the node's distribution, Gini measures how often you'd be wrong. For a pure node (100% class A), you'd always guess A and always be right—Gini = 0. For a 50-50 node, you'd be wrong half the time on average—Gini = 0.5 (maximum uncertainty). The tree seeks splits that create nodes where a random guess would more often be correct.

**Hand-calculating Gini:** For a node with 60% class A, 40% class B:

$$Gini = 1 - (0.6^2 + 0.4^2) = 1 - (0.36 + 0.16) = 1 - 0.52 = 0.48$$

This is close to maximum impurity (0.5), indicating the node is nearly evenly split.

> **Numerical Example: Evaluating a Split with Gini**
>
> ```python
> def gini(class_proportions):
>     return 1 - sum(p**2 for p in class_proportions)
>
> # Parent node: 100 samples, 50-50 split
> parent_gini = gini([0.5, 0.5])
> print(f"Parent Gini: {parent_gini:.4f}")
>
> # After split: Left (60 samples, 80-20), Right (40 samples, 10-90)
> left_gini = gini([0.8, 0.2])
> right_gini = gini([0.1, 0.9])
> weighted_gini = (60 * left_gini + 40 * right_gini) / 100
> info_gain = parent_gini - weighted_gini
>
> print(f"Left child Gini: {left_gini:.4f}")
> print(f"Right child Gini: {right_gini:.4f}")
> print(f"Weighted child Gini: {weighted_gini:.4f}")
> print(f"Information gain: {info_gain:.4f}")
> ```
>
> **Output:**
> ```
> Parent Gini: 0.5000
> Left child Gini: 0.3200
> Right child Gini: 0.1800
> Weighted child Gini: 0.2640
> Information gain: 0.2360
> ```
>
> **Interpretation:** The parent node has maximum impurity (0.5). After the split, both children are more "pure"—the left child is 80% one class (Gini=0.32), the right is 90% the other (Gini=0.18). The weighted average (0.264) is much lower than the parent (0.5), yielding high information gain (0.236). The tree picks the split that maximizes this gain.
>
> *Source: `slide_computations/module3_examples.py` - `demo_gini_impurity_calculation()`*

**Entropy**:

$$Entropy = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

Usually produces similar results. Gini is slightly faster (no logarithms).

In practice, Gini vs entropy rarely matters. Entropy penalizes near-equal splits slightly more; with many classes, Gini can favor isolating one class while entropy prefers balanced information gain. Default to Gini (slightly faster); if hyperparameter tuning, include criterion and let cross-validation decide.

### The scikit-learn API Pattern

This pattern is consistent across almost ALL scikit-learn models:

```python
# 1. Instantiate
model = DecisionTreeClassifier(max_depth=5)

# 2. Fit
model.fit(X_train, y_train)

# 3. Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### Decision Boundaries

Trees create rectangular decision regions:
- Each split creates a horizontal or vertical line
- Deep trees create many small rectangles
- Different from logistic regression's smooth boundary

### Demonstrating Overfitting

**Deep Tree (no limit):**
- Train accuracy: 100%
- Test accuracy: 75%
- Hundreds of nodes

**Shallow Tree (depth=3):**
- Train accuracy: 85%
- Test accuracy: 82%
- ~15 nodes

**Key insight**: Deep trees memorize training data including noise. 100% training accuracy almost certainly means overfitting.

> **Numerical Example: Decision Tree Overfitting**
>
> ```python
> from sklearn.datasets import make_classification
> from sklearn.model_selection import train_test_split
> from sklearn.tree import DecisionTreeClassifier
>
> # Data with 10% label noise
> X, y = make_classification(
>     n_samples=300, n_features=10, n_informative=5,
>     flip_y=0.1, random_state=42
> )
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
>
> for depth in [3, 5, 10, None]:
>     tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
>     tree.fit(X_train, y_train)
>     print(f"Depth {str(depth):>4}: Train={tree.score(X_train, y_train):.1%}, "
>           f"Test={tree.score(X_test, y_test):.1%}, Leaves={tree.get_n_leaves()}")
> ```
>
> **Output:**
> ```
> Depth    3: Train=81.9%, Test=71.1%, Leaves=7
> Depth    5: Train=92.4%, Test=77.8%, Leaves=16
> Depth   10: Train=100.0%, Test=80.0%, Leaves=29
> Depth None: Train=100.0%, Test=80.0%, Leaves=29
> ```
>
> **Interpretation:** The unlimited tree achieves 100% training accuracy—but the data has 10% label noise, so perfect training accuracy means it memorized the noise! The train-test gap (20%) signals overfitting. Shallow trees (depth 3-5) have lower training accuracy but smaller gaps, indicating better generalization.
>
> *Source: `slide_computations/module3_examples.py` - `demo_tree_overfitting()`*

100% training accuracy is occasionally okay: perfectly separable data (predicting even/odd from last digit), very small clean datasets, or memorization tasks. Verify by checking test accuracy (also very high?), the train-test gap (small vs large?), complexity (10 leaves for 10,000 samples = simple rules; 5,000 leaves = memorized), and cross-validation consistency. The heuristic remains useful: 100% training accuracy should trigger suspicion.

### Pruning Strategies

**Pre-pruning (early stopping):**
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split a node
- `min_samples_leaf`: Minimum samples in a leaf

**Post-pruning:**
- Grow full tree, then prune back
- Use `ccp_alpha` parameter
- Higher alpha = more pruning

**Recommendation**: Start with pre-pruning. Set `max_depth=5` as starting point, use cross-validation to optimize.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

tree = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
tree.fit(X_train, y_train)

# Check for overfitting
train_acc = tree.score(X_train, y_train)
test_acc = tree.score(X_test, y_test)
print(f"Train: {train_acc:.3f}, Test: {test_acc:.3f}")

# Cross-validation for depth selection
for depth in range(1, 15):
    tree_cv = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree_cv, X_train, y_train, cv=5)
    print(f"Depth {depth}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Feature Importance

$$Importance = \sum_{nodes} (impurity\ reduction \times samples)$$

**Caveats:**
- Importance is relative (sums to 1)
- Correlated features split importance between them
- Doesn't indicate direction of effect or causation

To understand importance with correlated features: use domain knowledge (which is more causal?), remove one and retrain (does importance transfer?), or use permutation importance (shuffles independently). For prediction, keeping both adds complexity without benefit. For interpretation, report both but note correlation. Consider reporting "this cluster of correlated features is important" rather than attributing to one.

**The "weighted vote" intuition:** Think of feature importance as a weighted vote. Every time a feature is used to split a node, it gets "votes" equal to (impurity reduction × number of samples affected). A feature used at the root affects all samples—many votes. A feature used deep in the tree affects few samples—fewer votes. The final importance is each feature's total votes divided by all votes. This explains why the root feature often has high importance even if it's not the "most important" conceptually—it affects the most samples.

> **Numerical Example: Feature Importance**
>
> ```python
> import numpy as np
> from sklearn.tree import DecisionTreeClassifier
>
> np.random.seed(42)
> n = 500
> x_strong = np.random.randn(n)  # Strong predictor (coef 2.0)
> x_medium = np.random.randn(n)  # Medium predictor (coef 1.0)
> x_weak = np.random.randn(n)    # Weak predictor (coef 0.3)
> x_noise = np.random.randn(n)   # Pure noise
>
> y = (2.0*x_strong + 1.0*x_medium + 0.3*x_weak + np.random.randn(n)*0.5 > 0).astype(int)
> X = np.column_stack([x_strong, x_medium, x_weak, x_noise])
>
> tree = DecisionTreeClassifier(max_depth=5, random_state=42)
> tree.fit(X, y)
>
> names = ['strong', 'medium', 'weak', 'noise']
> for name, imp in zip(names, tree.feature_importances_):
>     print(f"{name:>8}: {imp:.3f}")
> ```
>
> **Output:**
> ```
>   strong: 0.661
>   medium: 0.280
>     weak: 0.046
>    noise: 0.013
> ```
>
> **Interpretation:** The tree correctly identifies the strong predictor as most important (66%), followed by medium (28%). The weak predictor and noise have minimal importance. Note: importance is relative (sums to 1) and doesn't indicate causation—just predictive value in this tree.
>
> *Source: `slide_computations/module3_examples.py` - `demo_feature_importance()`*

### Why Decision Trees Are Popular

1. **Explainable**: Show decision rules to stakeholders
2. **No preprocessing**: Handle different scales, categorical variables, missing values
3. **Non-linear**: Capture complex relationships automatically
4. **Visual**: Tree diagrams are intuitive for non-technical audiences

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Deeper trees are always better" | Deeper trees overfit. Find the sweet spot via cross-validation. |
| "Decision trees require feature scaling" | Trees are scale-invariant! One of their advantages. |
| "Feature importance = causal importance" | Importance only shows predictive power, not causation. |
| "Trees can't capture interactions" | Trees naturally capture interactions through hierarchical structure. |

---

## 3.3 Handling Imbalanced Data

### Why Accuracy is Misleading

**Fraud detection:**
- 99.9% of transactions are legitimate
- 0.1% are fraudulent

**A model that predicts "legitimate" for EVERYTHING:**
- Accuracy: 99.9%
- Catches zero fraud!

**Accuracy is useless for imbalanced classes.**

**When is it "imbalanced"?** 60/40 is typically fine; 70/30 is mild; 80/20 starts requiring attention; 90/10 likely needs specialized techniques; 95/5 definitely needs SMOTE, class weights, or threshold adjustment. But it's not just about ratio—absolute numbers matter (90/10 with 10,000 minority samples is fine; with 100 is problematic). The practical test: does your model learn anything about the minority class? If accuracy comes from ignoring the minority entirely, you have a problem.

### Better Metrics

**Precision**: Of those we flagged as positive, how many actually were?

$$Precision = \frac{TP}{TP + FP}$$

**Recall**: Of actual positives, how many did we catch?

$$Recall = \frac{TP}{TP + FN}$$

**F1 Score**: Harmonic mean balancing both

$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**Why harmonic mean?** It punishes extreme imbalance:
- Precision = 100%, Recall = 1% → F1 = 2%
- Precision = 50%, Recall = 50% → F1 = 50%

The harmonic mean (as opposed to arithmetic mean) has a special property: it's dominated by the smaller value. If you have P=100% and R=1%, the arithmetic mean would be 50.5%—making it look decent. But the harmonic mean is 2%—revealing that one metric is terrible. This is what we want! A model that achieves high precision by predicting almost nothing (low recall) shouldn't score well. The harmonic mean enforces balance: both metrics must be reasonable for F1 to be high.

**The precision-recall "fishing net" analogy:** Imagine you're fishing for a specific type of fish (positives) in a lake (your data). Precision asks: "Of the fish in your net, what fraction are the type you wanted?" Recall asks: "Of all the target fish in the lake, what fraction did you catch?" A tight net (high threshold) catches few fish but mostly the right kind—high precision, low recall. A wide net (low threshold) catches more target fish but also lots of other fish—high recall, low precision. You can't optimize both without a better model (or more target fish).

> **Numerical Example: Precision-Recall at Different Thresholds**
>
> ```python
> from sklearn.datasets import make_classification
> from sklearn.linear_model import LogisticRegression
> from sklearn.metrics import precision_score, recall_score, f1_score
>
> # Imbalanced dataset: 90% negative, 10% positive
> X, y = make_classification(n_samples=1000, weights=[0.9, 0.1], random_state=42)
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
>
> model = LogisticRegression(random_state=42, max_iter=1000)
> model.fit(X_train, y_train)
> y_proba = model.predict_proba(X_test)[:, 1]
>
> for thresh in [0.1, 0.3, 0.5, 0.7]:
>     y_pred = (y_proba >= thresh).astype(int)
>     prec = precision_score(y_test, y_pred, zero_division=0)
>     rec = recall_score(y_test, y_pred, zero_division=0)
>     print(f"Threshold {thresh}: Precision={prec:.0%}, Recall={rec:.0%}")
> ```
>
> **Output:**
> ```
> Threshold 0.1: Precision=29%, Recall=69%
> Threshold 0.3: Precision=70%, Recall=44%
> Threshold 0.5: Precision=86%, Recall=19%
> Threshold 0.7: Precision=100%, Recall=9%
> ```
>
> **Interpretation:** As threshold increases, precision rises (fewer false alarms) but recall falls (more missed positives). At threshold 0.1, we catch 69% of positives but 71% of our "positive" predictions are wrong. At 0.7, we're always right when we predict positive, but we miss 91% of actual positives. Choose based on business costs!
>
> *Source: `slide_computations/module3_examples.py` - `demo_precision_recall_threshold()`*

### The Precision-Recall Trade-off

Usually you can't maximize both:
- High precision → few false alarms but miss some positives
- High recall → catch most positives but more false alarms

**Business context determines priority:**
- **High precision**: Email marketing (don't waste budget)
- **High recall**: Medical screening (don't miss sick patients)

### Resampling: SMOTE

**SMOTE** (Synthetic Minority Over-sampling Technique):
- Creates synthetic minority examples
- Interpolates between existing minority points
- Better than simple duplication

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

**How SMOTE "fills in the gaps":** SMOTE doesn't just copy existing minority examples (which would teach the model to memorize specific points). Instead, it creates synthetic examples by: (1) picking a minority example, (2) finding its k nearest minority neighbors, (3) drawing a line to one neighbor, and (4) placing a new point somewhere along that line. This "fills in" the feature space between existing minority examples, helping the model learn the general region where minority examples live rather than just memorizing specific cases. Think of it as sketching between the dots to reveal the underlying shape.

**Important**: Only apply SMOTE to training data, never test data! The test set must reflect real-world conditions—your deployed model will face the true class distribution. SMOTE is a training trick to help the model learn about the minority class, not a data transformation. The correct workflow: (1) Split data first. (2) Apply SMOTE only to training set. (3) Evaluate on original, imbalanced test set. (4) Use appropriate metrics (F1, precision, recall) that work for imbalanced data.

### Class Weights

Many algorithms have built-in support:

```python
model = LogisticRegression(class_weight='balanced')
model = DecisionTreeClassifier(class_weight='balanced')
```

**Effect**: Increases penalty for misclassifying minority class. Often simpler than resampling.

> **Numerical Example: Effect of Class Weights**
>
> ```python
> from sklearn.datasets import make_classification
> from sklearn.linear_model import LogisticRegression
> from sklearn.metrics import recall_score, f1_score
>
> # Severely imbalanced: 95% negative, 5% positive
> X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
>
> # Without class weights
> model_uw = LogisticRegression(class_weight=None, random_state=42, max_iter=1000)
> model_uw.fit(X_train, y_train)
> pred_uw = model_uw.predict(X_test)
>
> # With balanced class weights
> model_w = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
> model_w.fit(X_train, y_train)
> pred_w = model_w.predict(X_test)
>
> print(f"No weights:  Recall={recall_score(y_test, pred_uw):.0%}, F1={f1_score(y_test, pred_uw):.0%}")
> print(f"Balanced:    Recall={recall_score(y_test, pred_w):.0%}, F1={f1_score(y_test, pred_w):.0%}")
> ```
>
> **Output:**
> ```
> No weights:  Recall=0%, F1=0%
> Balanced:    Recall=78%, F1=19%
> ```
>
> **Interpretation:** Without weights, the model learns to predict "negative" for everything—achieving 96% accuracy by ignoring the minority class entirely (0% recall). With balanced weights, the model actually tries to find positives, achieving 78% recall. Accuracy drops because of more false positives, but F1 improves because we're actually solving the problem!
>
> *Source: `slide_computations/module3_examples.py` - `demo_class_weights()`*

### Threshold Adjustment

```python
y_proba = model.predict_proba(X_test)[:, 1]
threshold = 0.3  # Instead of 0.5
y_pred = (y_proba >= threshold).astype(int)
```

Lower threshold → predict positive more often → higher recall, lower precision.

### Business Context Examples

| Domain | Priority | Reason |
|--------|----------|--------|
| Fraud Detection | High recall | Cost of fraud >> investigation cost |
| Medical Diagnosis | High recall | Don't miss sick patients |
| Churn Prediction | Balance | Retention cost vs customer value |
| Manufacturing QC | Depends | Defect severity vs discard cost |

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Always balance classes to 50-50" | Optimal ratio depends on the problem. Original distribution may be meaningful. |
| "SMOTE is always better than oversampling" | SMOTE can create unrealistic synthetic examples. Test both. |
| "Class weights and resampling do the same thing" | Similar effect but different mechanisms. Results can differ. |
| "Imbalanced data is always a problem" | If minority class is well-separated, imbalance may not hurt. Always check metrics. |

---

## 3.4 Hyperparameter Optimization

### Parameters vs Hyperparameters

| Parameters | Hyperparameters |
|------------|-----------------|
| Learned during training | Set before training |
| Model learns via .fit() | You choose before .fit() |
| Example: Coefficients | Example: Regularization strength |
| Example: Split points | Example: Max tree depth |

**Hyperparameters control HOW the model learns.**

**Finding hyperparameters**: Use official documentation (search "sklearn DecisionTreeClassifier"), in-code exploration (`model.get_params()`, `help(DecisionTreeClassifier)`), or IDE autocomplete. Not all hyperparameters matter equally—most algorithms have 3-5 "important" ones: for decision trees, focus on `max_depth`, `min_samples_split`, `min_samples_leaf`; for Random Forests add `n_estimators`, `max_features`; for XGBoost: `learning_rate`, `max_depth`, `n_estimators`, `subsample`.

### Grid Search

Try every combination in a predefined grid:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
}
# Total: 4 × 3 = 12 combinations

grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='f1'
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
```

**Pros**: Exhaustive, reproducible
**Cons**: Exponential growth, wastes time on bad regions

### Random Search

Sample random combinations from distributions:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distributions = {
    'max_depth': randint(2, 20),
    'min_samples_split': randint(2, 50),
}

random_search = RandomizedSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=5,
    scoring='f1',
    random_state=42
)
random_search.fit(X_train, y_train)
```

### Why Random Often Beats Grid

**Key insight (Bergstra & Bengio, 2012):**
- Not all hyperparameters are equally important
- Grid search wastes trials on unimportant parameters
- Random search explores more values of what matters

**In practice, random search often beats grid search with the same computational budget.**

**Why random beats grid—the geometric intuition:** Imagine a 2D hyperparameter space where only one dimension matters (common in practice). Grid search with 9 trials might try 3 values per dimension, giving you only 3 unique values of the important parameter. Random search with 9 trials gives you 9 unique values of the important parameter! When you don't know which parameters matter most (and you usually don't), random search automatically allocates more trials to exploring variation in every dimension. Grid search wastes trials exploring combinations of unimportant parameters.

**Standard ranges for common hyperparameters**: `max_depth`: 2-20 for trees; `n_estimators`: 50-500 for forests/boosting; `learning_rate`: 0.001-0.3 for boosting; `min_samples_split`: 2-50; `C` (regularization): 0.001-100 (log scale). If the best value is at the edge of your range, extend that direction. Start with wide, log-spaced ranges, do a coarse search (10 values), then refine in the promising region.

### Bayesian Optimization (optuna)

Use past results to guide future trials:

```python
import optuna

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 2, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
    }
    model = DecisionTreeClassifier(**params, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

More efficient than random search—learns from previous trials.

### Never Use Test Set for Tuning!

**Correct workflow:**
1. Split into train/test
2. Use cross-validation on training set for tuning
3. Select best hyperparameters via CV score
4. Retrain on full training set
5. Evaluate **once** on test set

If you tune on test set, your estimate is no longer unbiased.

**After tuning**: Retrain on all training data with the best hyperparameters. Cross-validation models were trained on only (K-1)/K of your data. Retraining on 100% gives the model more examples. `GridSearchCV` does this automatically—`grid_search.best_estimator_` is already retrained on the full training set.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "More hyperparameter tuning always helps" | Diminishing returns. 50-100 trials often enough. Risk overfitting to validation data. |
| "Grid search is more thorough" | Grid is exhaustive only for values you specify. Random can find values between grid points. |
| "Best hyperparameters are universal" | Optimal hyperparameters depend on your specific dataset. |
| "Use test set to choose hyperparameters" | Never! Use cross-validation on training data. |

---

## Reflection Questions

1. A model predicts P(churn) = 0.6 for a customer. What does this actually mean? How confident should we be?

2. Why might you choose a threshold other than 0.5? Give scenarios for very low and very high thresholds.

3. A logistic regression coefficient for 'number of support tickets' is 0.3. How would you explain this to a stakeholder?

4. You build a decision tree with 100% training accuracy. Is this good or bad? What would you do next?

5. In fraud detection with 0.1% fraud rate, a model achieves 99.9% accuracy. What's wrong with celebrating this?

6. When would you prefer high precision over high recall? Give a business example.

7. Why might random search find better hyperparameters than grid search with the same budget?

---

## Practice Problems

1. Calculate odds and log-odds for P = 0.8

2. Given coefficients β₀ = -2, β₁ = 0.5, β₂ = -0.3, calculate P(Y=1) when x₁ = 4, x₂ = 2

3. Draw what a decision tree boundary would look like for 2D data with 2 splits

4. Given a 95% legitimate / 5% fraud dataset: if we predict all legitimate, what's accuracy? Precision for fraud? Recall for fraud?

5. Choose between precision and recall priority for: (a) spam filter, (b) cancer screening, (c) loan approval

---

## Chapter Summary

**Six key takeaways from Module 3:**

1. **Logistic regression** outputs probabilities via sigmoid; threshold for classification

2. **Odds ratios** (exponentiate coefficients) translate to business-friendly interpretation

3. **Decision trees** are intuitive but overfit easily—use pruning

4. **Accuracy is misleading** for imbalanced data—use precision/recall/F1

5. **Handle imbalance** with SMOTE, class weights, or threshold adjustment

6. **Hyperparameter tuning** via cross-validation, never on test set

---

## What's Next

In Module 4, we tackle **Ensemble Methods**:
- Random Forests (ensembles of decision trees)
- Gradient Boosting (XGBoost, LightGBM)
- Why combining weak learners creates strong models

Understanding decision trees is essential—Random Forests take everything we learned about trees and combine many of them for better performance.
