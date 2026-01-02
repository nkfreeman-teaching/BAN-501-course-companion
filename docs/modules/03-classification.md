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

**Important**: Only apply SMOTE to training data, never test data! The test set must reflect real-world conditions—your deployed model will face the true class distribution. SMOTE is a training trick to help the model learn about the minority class, not a data transformation. The correct workflow: (1) Split data first. (2) Apply SMOTE only to training set. (3) Evaluate on original, imbalanced test set. (4) Use appropriate metrics (F1, precision, recall) that work for imbalanced data.

### Class Weights

Many algorithms have built-in support:

```python
model = LogisticRegression(class_weight='balanced')
model = DecisionTreeClassifier(class_weight='balanced')
```

**Effect**: Increases penalty for misclassifying minority class. Often simpler than resampling.

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
