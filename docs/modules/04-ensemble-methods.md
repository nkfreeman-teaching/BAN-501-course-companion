# Module 4: Ensemble Methods

## Introduction

In Module 3, we learned about decision trees—intuitive classifiers that are easy to interpret but prone to overfitting. Deep trees memorize training data; shallow trees underfit.

This module answers a natural question: **What if we could get the benefits of deep trees without the overfitting?**

The answer is ensemble methods. Instead of training one model, we train many models and combine their predictions. This simple idea—the wisdom of crowds—turns out to be one of the most powerful techniques in machine learning.

By the end of this module, you'll understand two major ensemble paradigms: **bagging** (where Random Forests come from) and **boosting** (where XGBoost comes from). These methods dominate tabular data competitions and are workhorses in industry.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the intuition behind ensemble methods and why combining models outperforms individuals
2. **Implement** bagging (Random Forests) and boosting (XGBoost)
3. **Interpret** feature importance for business stakeholders
4. **Select** appropriate ensemble strategies based on problem characteristics

---

## 4.1 Ensemble Learning Concepts

### The Wisdom of Crowds

**Galton's Ox Experiment (1907):**

At a county fair, 787 people tried to guess the weight of an ox. Individual guesses varied wildly—some way too high, some way too low.

- **Median of all guesses: 1,207 lbs**
- **Actual weight: 1,198 lbs** (< 1% error!)

How can a crowd of non-experts outperform individuals?

**Key insight**: Errors cancel out when they're uncorrelated. Some people guessed too high, some too low. The errors went in different directions. When you average, errors cancel and the true signal remains.

This is exactly the principle behind ensemble machine learning.

**Correlation matters**: Ensembles work best with uncorrelated errors, but help even with partially correlated errors. If individual models have variance σ² and correlation ρ between errors, ensemble variance is ρσ² + (1-ρ)σ²/n. With perfect independence (ρ=0), variance drops as 1/n. With perfect correlation (ρ=1), averaging doesn't help. In practice, even 50% correlation provides substantial benefit.

### How Ensembles Improve Predictions

**Variance Reduction (Bagging):**
- Single decision trees are high-variance estimators
- Small changes in training data → very different trees
- Averaging multiple trees reduces instability
- Mathematically: $Var(average) = Var(individual) / n$ when predictions are uncorrelated

**Bias Reduction (Boosting):**
- Each new model focuses on errors of previous models
- The ensemble gradually learns patterns individual weak learners missed
- Sequential learning reduces systematic error

### Model Diversity is Critical

**Ensembles only help if the models are different!**

If all models make the same mistakes, averaging doesn't help. Think: if you ask 787 people the same leading question and they all guess the same wrong answer, the median is still wrong.

**How ensemble methods create diversity:**
- Random Forests: Random sampling of data AND features
- Boosting: Sequential focus on different examples
- Different algorithms: Different inductive biases (heterogeneous ensembles)

**Heterogeneous ensembles** combine completely different algorithms (neural network + decision tree + logistic regression). Different algorithms have different inductive biases, making them unlikely to make the same mistakes. The Netflix Prize winning solution combined 107 different models.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "More models always means better results" | Diminishing returns kick in quickly. 1 → 10 trees helps a lot; 100 → 1000 helps little. |
| "Ensembles are always better than single models" | For simple problems or when interpretability is paramount, single models may be preferable. |
| "You need sophisticated models in your ensemble" | Ensembles of simple models (shallow trees, stumps) can be remarkably effective. |

---

## 4.2 Bagging Methods

### Three Components: Random Forest

| Component | Random Forest |
|-----------|---------------|
| **Decision Model** | Ensemble of decision trees — each tree votes, majority wins |
| **Quality Measure** | Gini/entropy for individual trees; OOB error for ensemble |
| **Update Method** | Independent parallel training — no iteration between trees |

**Key insight**: Random Forest doesn't "update" traditionally. Each tree trains independently on a bootstrap sample. Learning happens through aggregation—the wisdom of crowds.

### Bootstrap Aggregating (Bagging)

**Algorithm:**

1. **Create B bootstrap samples** (sample with replacement)
   - Each sample same size as original data
   - Some observations appear multiple times, some not at all
   - ~63.2% unique observations per sample

2. **Train a separate model on each sample**

3. **Aggregate predictions:**
   - Regression: Average
   - Classification: Majority vote

**Why ~63.2%?** When sampling n observations with replacement from n, the probability any specific row is never selected is:

$$(1 - \frac{1}{n})^n \approx e^{-1} \approx 0.368$$

So ~36.8% are left out ("out-of-bag"), meaning ~63.2% are included.

**Why replacement?** Without replacement at the same size, you'd get identical datasets. With replacement: some observations appear multiple times (emphasized), some don't appear (~36.8%, providing OOB validation), and different trees emphasize different observations—creating diversity. Bootstrap sampling approximates drawing fresh samples from the true population.

### Random Forests: Double Randomness

Random Forests extend bagging with **two sources of randomness**:

1. **Row sampling** (from bagging): Each tree gets a bootstrap sample

2. **Feature sampling** (unique to RF): At each split, consider only a random subset
   - Default: $\sqrt{d}$ features for classification (where d = total features)

**Why feature sampling matters:**

Imagine one incredibly predictive feature (credit score for loan default). Without feature sampling, every tree uses it as the root split. All trees become highly correlated.

With feature sampling, each split considers a random subset. Sometimes credit score isn't available. The tree finds other splits. This creates diversity.

**The tradeoff**: Ignoring the best feature sometimes hurts individual trees (higher bias), but trees become more diverse (lower correlation). The ensemble variance formula shows reducing correlation (ρ) often helps more than the slight increase in individual variance (σ²). Random Forests typically outperform bagged trees precisely because of this tradeoff. The `max_features` hyperparameter controls this—default √d is a good starting point.

### Why Bagging Reduces Overfitting

- A single deep tree overfits to specific patterns
- Each tree in the forest also overfits, but to DIFFERENT patterns
- When we average, idiosyncratic overfitting cancels out
- True signal remains (all trees agree on it)

**The ensemble variance formula:**

$$Var(ensemble) = \rho\sigma^2 + \frac{(1-\rho)\sigma^2}{n}$$

Where:
- $\sigma^2$ = variance of individual tree predictions
- $\rho$ = average correlation between trees (0 = independent, 1 = identical)
- $n$ = number of trees

**Reading this formula:**
- First term ($\rho\sigma^2$): Irreducible variance from correlation
- Second term: Shrinks as you add trees

**Key insight:** Lower correlation between trees = better ensemble. Feature sampling specifically reduces $\rho$.

**Number of trees**: 100-500 trees usually sufficient. Plot OOB error vs. n_estimators—it decreases rapidly then flattens. Unlike boosting, more RF trees never hurt performance; they just stop helping. More trees mean more memory and slower inference, so balance accuracy against cost.

### Feature Importance

**Mean Decrease in Impurity (MDI):**
- Sum of impurity decreases from splits using each feature, averaged across trees
- Fast to compute
- Can favor high-cardinality features

**Permutation Importance:**
- Shuffle each feature and measure accuracy decrease
- More reliable, slower
- Preferred for stakeholder communication

**Important caveat**: Importance ≠ direction of effect! Importance tells you which features the model relies on, not HOW they affect predictions. For that, use SHAP values (Module 9).

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    oob_score=True,
    random_state=42
)
rf.fit(X_train, y_train)

# OOB score (free validation!)
print(f"OOB Accuracy: {rf.oob_score_:.3f}")

# MDI importance (fast)
importance_mdi = rf.feature_importances_

# Permutation importance (more reliable)
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10)
```

### Out-of-Bag (OOB) Error

Each bootstrap sample leaves out ~36.8% of observations. These "out-of-bag" samples provide free validation:

- For each observation, predict using only trees that didn't train on it
- OOB error ≈ cross-validation error, but FREE!

```python
rf = RandomForestClassifier(oob_score=True)
rf.fit(X_train, y_train)
print(f"OOB Accuracy: {rf.oob_score_}")
```

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Random Forest can't overfit" | It can! Deep trees with too few estimators still overfit. Tuning on test set causes overfitting to that. |
| "More trees is always better" | Diminishing returns. 100-500 usually sufficient. |
| "Random Forest is a black box" | Feature importance and SHAP make it reasonably interpretable. |
| "Feature importance = feature effect" | Importance shows reliance, not direction of effect. |

---

## 4.3 Boosting Methods

### Three Components: Gradient Boosting

| Component | Gradient Boosting (XGBoost) |
|-----------|----------------------------|
| **Decision Model** | Sequential ensemble — sum of many shallow trees |
| **Quality Measure** | Any differentiable loss + regularization |
| **Update Method** | Gradient descent in function space — each tree corrects previous errors |

The update method is fascinating: instead of updating parameters, we add new functions (trees). Each tree predicts the negative gradient (residuals).

### The Boosting Philosophy

Build models sequentially, where each new model focuses on mistakes of previous ones.

| Bagging | Boosting |
|---------|----------|
| Parallel (independent trees) | Sequential (dependent trees) |
| Reduces variance | Reduces bias (and variance) |
| Deep trees | Shallow trees typical |

**Visual metaphors:**
- Bagging: Committee of experts who work independently and vote
- Boosting: Relay team where each runner covers for previous weaknesses

### AdaBoost: Adaptive Boosting

1. Start with equal weights for all training examples
2. Train a weak learner (often a "stump"—one split)
3. Identify misclassified examples
4. Increase weights on misclassified examples
5. Train next weak learner on reweighted data
6. Repeat

**Key insight**: Each subsequent learner specializes in hard examples previous learners got wrong.

**Boosting and outliers**: Boosting can obsess over mislabeled or impossible-to-fit examples. Mitigation: (1) `subsample` (0.8) so outliers don't appear every round, (2) lower learning rate to limit per-iteration damage, (3) regularization (`reg_alpha`, `reg_lambda`) to prevent extreme predictions, (4) early stopping before overfitting to noise. Random Forests are more robust because outliers only affect ~63% of trees and no tree specifically focuses on them.

### Gradient Boosting Machines

**Core innovation**: Fit each new tree to the residuals (errors).

1. Make initial prediction (often the mean)
2. Calculate residuals: $actual - predicted$
3. Fit a tree to predict the residuals
4. Add this tree's predictions (with learning rate)
5. Calculate new residuals
6. Repeat

**Why "gradient"?** For MSE loss:

$$\frac{\partial L}{\partial \hat{y}} = -(y - \hat{y}) = -\text{residual}$$

The residual IS the negative gradient of the loss. When we fit trees to residuals, we're following the gradient in function space.

**Gradient in function space**: Normal gradient descent optimizes parameters (adjust θ). Gradient boosting optimizes functions (add a new tree). For squared error, the negative gradient is simply the residual. Fitting a tree to residuals approximates "what should I add to reduce error?" The learning rate works like in gradient descent—taking fractional steps (0.1 × tree_prediction) prevents overshooting. So: F_new(x) = F_old(x) + learning_rate × new_tree(x). Each tree is a step in function space toward lower loss.

### Key Boosting Hyperparameters

| Parameter | Effect |
|-----------|--------|
| `n_estimators` | More → more capacity, but overfit risk |
| `learning_rate` | Smaller → need more trees, often better |
| `max_depth` | Usually 3-8 (much shallower than RF) |

**Trade-off**: Lower learning rate + more trees often gives best results but takes longer.

**Tree depth difference:**
- Random Forest: Deep, fully-grown trees (low bias, high variance). Averaging reduces variance.
- Boosting: Shallow trees (high bias). Sequential correction reduces bias.

### XGBoost: The Competition Champion

XGBoost adds optimizations that make it dominant:

1. **Regularization**: L1/L2 penalties on leaf weights
2. **Parallel processing**: Split evaluation parallelized within trees
3. **Missing value handling**: Learns optimal direction for missing values
4. **Histogram-based splitting**: Bins features for speed

**Why it dominates**: Won more Kaggle competitions than any other algorithm. Widely adopted in finance, insurance, tech.

**When Random Forest is better**: (1) Noisy labels—RF more robust, noise doesn't compound; (2) Limited tuning time—RF works well with defaults; (3) Parallelization—RF trees train independently; (4) Small datasets—boosting can overfit quickly. A well-tuned XGBoost beats a well-tuned RF, but default RF often beats default XGBoost. In many real-world scenarios, the difference is 1-2%.

### XGBoost with Early Stopping

**Always use early stopping with boosting!**

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
    verbose=False
)

print(f"Best iteration: {xgb_model.best_iteration}")
```

Without early stopping, boosting overfits. With it, training stops when validation plateaus.

**Why early stopping beats fixed n_estimators**: The optimal number depends on learning rate, tree depth, data complexity, and sample size—a fixed number can't adapt. Set a large n_estimators as an upper limit, monitor validation loss, stop when no improvement for N consecutive rounds. The model finds its own stopping point, works with any learning rate, and prevents overfitting automatically. Always use a separate validation set for early stopping—not your final test set.

### LightGBM and CatBoost

**LightGBM:**
- Even faster than XGBoost
- Histogram-based splitting
- Great for very large datasets

**CatBoost:**
- Excellent categorical feature handling
- No one-hot encoding needed
- Often works well with defaults

**Rule of thumb**: Start with XGBoost. Try LightGBM for very large data. Try CatBoost for many categorical features.

### Bagging vs Boosting: When to Use Each

| Scenario | Recommendation |
|----------|----------------|
| High-variance (deep trees) | Bagging (RF) |
| High-bias (shallow trees) | Boosting |
| Fast training needed | Bagging (parallelizable) |
| Best accuracy needed | Boosting (often wins) |
| Noisy labels | Bagging (more robust) |
| Need interpretability | Random Forest |

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "XGBoost is always best" | No Free Lunch. Linear models beat it on linear data. Neural networks beat it on images/text. |
| "Boosting can't overfit" | Very much can! Use early stopping. |
| "More boosting rounds = better" | Unlike RF, more rounds increases overfit risk. |
| "XGBoost, LightGBM, CatBoost are completely different" | All gradient boosting variants. Similar core ideas. |

---

## 4.4 Other Ensemble Techniques

### Stacking

Use model predictions as features for a "meta-learner."

```
Level 0:   RF_pred    XGB_pred    LR_pred
              ↓           ↓          ↓
Level 1:     Meta-model (e.g., Logistic Regression)
                         ↓
                  Final Prediction
```

**How it works:**
1. Train several level-0 models (RF, XGBoost, logistic regression)
2. Generate predictions using cross-validation (out-of-fold)
3. Use predictions as features for level-1 meta-model
4. Meta-model learns which base models to trust

**Critical**: Must use out-of-fold predictions to avoid leakage!

**Why out-of-fold matters:**

If you train RF on all training data and use its predictions on that same data as meta-features, RF makes artificially confident predictions (it's seen those examples). This won't generalize.

**Correct approach:**
1. Split into K folds
2. For fold 1: Train on folds 2-5, predict fold 1
3. For fold 2: Train on folds 1,3-5, predict fold 2
4. Continue for all folds
5. Meta-model trains on these honest predictions

**Multi-level stacking**: Going deeper is possible but rarely worthwhile. Two levels is usually sufficient (Netflix Prize used two). Each additional level requires proper out-of-fold predictions (complex bookkeeping), increases overfitting risk, and slows inference. In production, a single well-tuned XGBoost or simple two-level stack is almost always preferred.

### Voting Classifiers

Simpler than stacking: combine predictions directly.

**Hard voting**: Each model votes; majority wins.

**Soft voting**: Average probability estimates; pick highest.

```python
# Model 1: P(A)=0.7, P(B)=0.3
# Model 2: P(A)=0.4, P(B)=0.6
# Model 3: P(A)=0.8, P(B)=0.2
# Average: P(A)=0.63 → Class A
```

**Soft voting usually performs better** (uses more information).

```python
from sklearn.ensemble import VotingClassifier, StackingClassifier

# Soft Voting
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100)),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)

# Stacking
stacking_clf = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('gb', GradientBoostingClassifier(n_estimators=100))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
```

### When to Use Each Approach

| Method | Use When |
|--------|----------|
| Simple voting | Models roughly equal; quick solution |
| Weighted voting | Some models clearly better |
| Stacking | Time for complexity; competition setting |

**Avoid sophisticated ensembles when:**
- Explainability is crucial
- Fast inference needed
- Limited compute

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Stacking always improves performance" | If base models are highly correlated, stacking adds complexity without benefit. |
| "More diverse base models = better" | Diversity helps, but models still need to be individually competent. |
| "Stacking is just averaging with extra steps" | Meta-learner can learn complex patterns like "trust RF for certain input ranges." |

---

## Reflection Questions

1. Why does the wisdom of crowds work? Under what conditions would it fail?

2. A colleague says Random Forest can never overfit. How would you respond?

3. Why sample features at each split rather than once per tree?

4. When might boosting overfit more easily than bagging? What would you adjust?

5. A data scientist says they always use XGBoost because "it wins Kaggle." What's your response?

6. You have 5 models with accuracies 82%, 81%, 79%, 78%, 75%. Would you ensemble all 5? Why or why not?

---

## Practice Problems

1. Derive why ~63.2% of observations appear in each bootstrap sample

2. If you have 100 features in a classification problem, how many are considered at each split in Random Forest (default)?

3. Explain why Random Forest feature importance might differ from permutation importance

4. Draw a diagram showing how 5 stumps combine in AdaBoost vs how 5 shallow trees combine in Gradient Boosting

5. You train XGBoost without early stopping and see training accuracy at 99% but test accuracy at 75%. Diagnose and fix.

---

## Chapter Summary

**Six key takeaways from Module 4:**

1. **Ensembles** work by combining diverse models—errors cancel out

2. **Bagging** (Random Forests) reduces variance through averaging independent trees

3. **Boosting** (XGBoost) reduces bias through sequential learning from errors

4. **Feature importance** shows predictive power, not effect direction

5. **Early stopping** is essential for boosting methods

6. **Choose wisely**: Random Forest for robustness, XGBoost for accuracy

---

## What's Next

In Module 5, we tackle **Unsupervised Learning**:
- Clustering (K-Means, hierarchical)
- Dimensionality reduction (PCA)
- Finding structure without labels

So far, we've had a target variable to predict. In unsupervised learning, there's no target—we're discovering hidden patterns in the data.
