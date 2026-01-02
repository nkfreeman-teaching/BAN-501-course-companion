# Module 2: Classical Machine Learning - Regression

## Introduction

Module 1 established the foundation: what ML is, how to prepare data, and how to evaluate models. Now we put that foundation to work.

Regression is the workhorse of predictive analytics. When a business wants to predict sales, estimate prices, or forecast demand, regression is often the first tool they reach for. But we're not just going to use regression as a black box—we're going to understand *how* it works, including implementing gradient descent from scratch.

Why learn gradient descent? Because gradient descent is the foundation for training neural networks. Every deep learning model you've heard of—GPT, image classifiers, everything—learns through gradient descent. Understanding it for linear regression means understanding it for neural networks.

The closed-form solution for linear regression requires matrix inversion—O(n³) complexity that's impossible for neural networks with millions of parameters. Neural networks also have non-convex loss surfaces with many local minima; there's no mathematical formula to jump to the optimal weights. Gradient descent works for any differentiable function and scales to billions of parameters—the linear regression closed-form is a special case where we can skip the search.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Explain** the mechanics of linear regression including the least squares method
2. **Implement** gradient descent from scratch and understand its trade-offs
3. **Interpret** regression coefficients for business insights
4. **Diagnose** model issues through residual analysis
5. **Apply** regularization techniques (L1, L2, Elastic Net) to prevent overfitting
6. **Communicate** regression findings to non-technical stakeholders

---

## 2.1 Simple Linear Regression

### The Three Components of Every ML Model

Before diving into linear regression, here's a framework that applies to *every* supervised learning algorithm:

| Component | Question It Answers | Linear Regression |
|-----------|---------------------|-------------------|
| **Decision Model** | How do we transform inputs into predictions? | $\hat{y} = \beta_0 + \beta_1 x$ |
| **Quality Measure** | How do we evaluate prediction quality? | Sum of Squared Errors (SSE) |
| **Update Method** | How do we improve the model? | Gradient descent (or closed-form) |

This same pattern applies to every algorithm: logistic regression, decision trees, random forests, neural networks. The decision model changes, the quality measure may change, but the structure is always the same.

Different quality measures encode different assumptions about what "good" means. MSE assumes symmetric, quadratic costs—but in demand forecasting, under-predicting (stockouts) might cost more than over-predicting. For classification, cross-entropy penalizes confident wrong predictions. For outlier-heavy data, MAE or Huber loss are more robust. Choose a quality measure that aligns with your actual business costs.

### The Goal of Linear Regression

Given input features, we want to predict a continuous output. We assume the relationship can be approximated by a line:

$$\hat{y} = \beta_0 + \beta_1 x$$

Where:
- $\hat{y}$ (y-hat) is the **predicted value**
- $\beta_0$ (beta-zero) is the **intercept**—the baseline prediction when x = 0
- $\beta_1$ (beta-one) is the **slope**—how much y changes for a one-unit change in x
- $x$ is the **input feature**

The key assumption is that a straight line is a reasonable approximation of the true relationship.

Check linearity visually: scatter plots should show points around an imaginary straight line. Use `sns.pairplot()` for multiple regression. Correlation measures only *linear* association—a perfect U-shaped relationship has r=0. Always check residual plots after fitting; curved patterns reveal non-linearity. If non-linear, try transforms (log, square root), polynomial terms (x², x³), or inherently non-linear models (trees, neural networks).

### The Least Squares Method

How do we find the *best* line? We find the coefficients that minimize squared prediction errors:

$$\text{minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

For each data point, calculate the error (actual minus predicted), square it, and add them all up. The best line is the one that makes this sum as small as possible.

**Why squared errors?**

1. **Penalizes large errors more than small ones.** An error of 10 contributes 100; an error of 1 contributes only 1. The algorithm cares about avoiding big mistakes.

2. **Mathematically tractable.** The function is differentiable and convex, so we can find the minimum using calculus.

3. **Nice statistical properties.** Under certain assumptions, least squares gives the Best Linear Unbiased Estimator (BLUE).

For asymmetric costs, use **weighted least squares** (assign higher weights to observations where errors are more costly) or **quantile regression** (systematically over/under-predicts—useful for safety stock). In deep learning, you can define arbitrary custom loss functions. Start simple; only add complexity when you have clear business justification for asymmetric costs.

**Closed-form solution:**

$$\beta_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

$$\beta_0 = \bar{y} - \beta_1\bar{x}$$

### Linear Regression Assumptions

For statistical inference to be valid, certain assumptions must hold:

1. **Linearity**: The relationship between X and Y is linear
2. **Independence**: Observations are independent of each other
3. **Homoscedasticity**: Constant variance of residuals across all levels of X
4. **Normality**: Residuals are normally distributed
5. **No multicollinearity**: (for multiple regression) Predictors aren't too highly correlated

**When assumptions are violated:**

| Violation | Symptom | Solution |
|-----------|---------|----------|
| Non-linearity | Curved pattern in residuals | Transform variables, add polynomial terms |
| Heteroscedasticity | Fan-shaped residual plot | Transform Y, use robust standard errors |
| Non-normality | Q-Q plot deviates from line | Transform Y, use bootstrap |
| Autocorrelation | Patterns in time-ordered residuals | Time series methods |

**Common transformations:**

| Transform | Formula | Best for |
|-----------|---------|----------|
| Log | $\log(x)$ | Right-skewed data, multiplicative relationships |
| Square root | $\sqrt{x}$ | Count data, mild right skew |
| Box-Cox | $(x^\lambda - 1)/\lambda$ | Automated selection—finds optimal λ |

The log transform is the workhorse—it handles right-skewed distributions (common in business data like income, prices, counts) and converts multiplicative relationships to additive ones.

With a log-transformed target, coefficients represent **percentage changes**: a one-unit increase in x multiplies y by $e^{\beta_1}$. For small β (roughly |β| < 0.2), this approximates β × 100% change. If both x and y are logged, coefficients represent **elasticities**: a 1% change in x → β₁% change in y. Always back-transform predictions before evaluating metrics, and document which scale coefficients are interpreted on.

### Gradient Descent

We could use the closed-form solution, but gradient descent is worth learning because it's the foundation for all neural network training.

**The algorithm:**
1. Start with random values for $\beta_0$ and $\beta_1$
2. Calculate the gradient—the direction of steepest error increase
3. Update parameters in the **opposite** direction—downhill toward lower error
4. Repeat until convergence

**The gradients:**

$$\frac{\partial MSE}{\partial \beta_0} = -\frac{2}{n}\sum(y_i - \hat{y}_i)$$

$$\frac{\partial MSE}{\partial \beta_1} = -\frac{2}{n}\sum(y_i - \hat{y}_i) \cdot x_i$$

**Update rules:**

$$\beta_0 \leftarrow \beta_0 - \alpha \cdot \frac{\partial MSE}{\partial \beta_0}$$

$$\beta_1 \leftarrow \beta_1 - \alpha \cdot \frac{\partial MSE}{\partial \beta_1}$$

Where $\alpha$ is the **learning rate**—how big a step we take each iteration.

**Convergence** means parameters have stabilized—further iterations don't meaningfully improve the solution. Common stopping criteria: loss change below threshold (1e-6), small gradient magnitude, or maximum iterations. Use a combination. For linear regression, the loss surface is convex with one global minimum; for neural networks, you'll find a local minimum (usually good enough). Monitor the loss curve—oscillating or increasing loss suggests the learning rate is too high.

**Implementation:**

```python
def gradient_descent_linear_regression(
    X, y,
    learning_rate=0.01,
    n_iterations=1000,
    tolerance=1e-6
):
    n = len(X)

    # Initialize parameters randomly
    beta_0 = np.random.randn()
    beta_1 = np.random.randn()

    history = []

    for i in range(n_iterations):
        # Predictions
        y_pred = beta_0 + beta_1 * X

        # Compute gradients
        d_beta_0 = -2/n * np.sum(y - y_pred)
        d_beta_1 = -2/n * np.sum((y - y_pred) * X)

        # Update parameters
        beta_0 = beta_0 - learning_rate * d_beta_0
        beta_1 = beta_1 - learning_rate * d_beta_1

        # Track loss
        mse = np.mean((y - y_pred)**2)
        history.append(mse)

        # Check convergence
        if i > 0 and abs(history[-1] - history[-2]) < tolerance:
            break

    return beta_0, beta_1, history
```

### Learning Rate Trade-offs

The learning rate $\alpha$ is crucial:

| Too Small | Just Right | Too Large |
|-----------|------------|-----------|
| Very slow convergence | Converges in reasonable time | Overshoots the minimum |
| Safe but inefficient | Reaches good solution | Can diverge (loss increases!) |

If your loss keeps *increasing*, the learning rate is too high. Reduce it by a factor of 10.

**Adaptive learning rate methods** automatically adjust during training. Learning rate schedules (step decay, exponential decay, cosine annealing) decrease the rate over time—large steps initially, smaller steps later. **Adaptive optimizers** (AdaGrad, RMSprop, Adam) adjust per parameter based on gradient history. **Adam** is the default for deep learning—it works well with default learning rate 0.001. For scikit-learn's linear regression, optimization is handled automatically.

### Using statsmodels for Regression

In practice, we use libraries. For regression with good statistical output, use statsmodels:

```python
import statsmodels.formula.api as smf

# R-style formula interface
model = smf.ols(
    formula='sales ~ advertising + price',
    data=df
)
results = model.fit()
print(results.summary())
```

**Key statistics in the output:**

**R² (Coefficient of Determination):**

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

R² tells you what proportion of variance your model explains. R² = 0.75 means 75% explained, 25% unexplained.

**p-values for coefficients:**
- Tests: "Is this coefficient different from zero?"
- p < 0.05 is conventional threshold for "statistically significant"
- Caution: p-values don't tell you effect *size*

The 0.05 threshold is a historical convention from R.A. Fisher, not a magic number. Problems: with enough data, trivial effects become "significant"; with little data, real effects may not be. Modern practice: report exact p-values, consider effect sizes and confidence intervals, and remember that practical significance matters more than statistical significance in business—a statistically significant 0.1% improvement might not be worth implementing.

**Confidence intervals:**
- 95% CI of [1.8, 3.2] means: "We're 95% confident the true effect is between $1.80 and $3.20"
- If the CI includes zero, the effect is not statistically significant

**F-statistic:**
- Tests whether the model as a whole is useful
- "Is this model better than just predicting the mean?"

**When to use scikit-learn instead:** When building ML pipelines, when prediction is the main goal, when you need cross-validation and hyperparameter tuning.

### Interpreting Coefficients

**The standard interpretation:**
"A one-unit increase in X is associated with a $\beta_1$ change in Y, *holding all else constant*."

**Example:**
```
sales = 50,000 + 2.5 × advertising + 1,200 × sales_staff
```

Translation:
- **Baseline sales: $50,000** (when advertising = 0 and sales_staff = 0)
- **Each $1 in advertising → $2.50 more in sales** (250% ROI)
- **Each additional sales staff → $1,200 more in sales**

**Caution: Correlation ≠ Causation**

Regression shows *association*, not *causation*. When we say "Each $1 in advertising → $2.50 more in sales," we mean they're associated. We haven't proven advertising *causes* sales.

Classic example: Ice cream sales and drowning deaths are positively correlated. Both are caused by summer heat—a confounding variable.

Establishing causation requires experimental design or careful causal inference methods. **Randomized experiments** (A/B tests) are the gold standard. **Natural experiments** (policy changes affecting some regions) create quasi-random groups. **Instrumental variables** find factors that affect treatment but not outcome directly. **Causal inference frameworks** (propensity score matching, difference-in-differences) try to estimate effects from observational data. With observational regression alone, you have association—to claim causation, you need a convincing argument for why confounders are controlled.

### Residual Analysis

Residuals reveal problems:

$$e_i = y_i - \hat{y}_i$$

If the model is good and assumptions hold, residuals should look like random noise.

**Diagnostic plots:**

1. **Residuals vs. Fitted Values**: Should show random scatter around zero
   - Curve = non-linearity
   - Funnel = heteroscedasticity

2. **Q-Q Plot**: Residuals vs. theoretical normal quantiles
   - Straight line = normality satisfied
   - Curves at ends = heavy/light tails

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')

plt.subplot(1, 3, 2)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q Plot')

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Residuals')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()
```

**Key insight:** High R² with a patterned residual plot is still a bad model. The pattern means you're missing structure in the data.

To fix a curved residual pattern: (1) Identify which feature causes it by plotting residuals against each predictor. (2) Try transforms—log for diminishing returns, square root for counts, polynomial terms (x², x³). (3) Use `PolynomialFeatures(degree=2)` with regularization. (4) If transforms don't help, consider non-linear models (trees, GAMs, neural networks). (5) Verify the fix—residuals should show random scatter.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Correlation implies causation" | Regression shows association only. Causation requires experimental design or causal inference methods. |
| "High R² means the model is good" | R² can be high due to overfitting. Must check test set performance and residual plots. |
| "The intercept is always meaningful" | Often it's not (e.g., salary when experience = 0 years). Focus on slopes for interpretation. |
| "Larger coefficients mean more important features" | Only true if features are on the same scale. Use standardized coefficients to compare. |

---

## 2.2 Multiple Linear Regression

### Multiple Predictors

In the real world, we rarely have just one predictor:

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_d x_d$$

**Why multiple predictors?**
1. Single predictor rarely captures the full story
2. Control for confounding variables
3. Improve prediction accuracy

**Key insight:** Each coefficient represents the effect of that variable *while controlling for the others*. This is different from running separate simple regressions!

Multiple regression coefficients are **partial effects**—the effect of one variable holding others constant. Simple regression captures both direct and indirect effects through correlated variables. If experience and education are correlated, simple regression conflates their effects; multiple regression "controls for" education when estimating experience's effect. Sometimes adding variables can even flip coefficient signs (Simpson's paradox)—the Berkeley admissions example showed apparent disadvantage for women overall that reversed within each department.

### Confounding Variables

**Example:**
- **Simple regression:** Salary ~ Experience → $\beta = \$5,000$ per year
- **Multiple regression:** Salary ~ Experience + Education → $\beta_{exp} = \$3,500$ per year

The coefficient for experience dropped because education was **confounded** with experience. People with more experience often have more education. The simple regression was attributing some of education's effect to experience.

To identify confounders, ask: "What could affect both X and Y?" Draw causal diagrams—confounders have arrows TO both predictor and outcome. Check correlations, but correlation alone isn't sufficient; you need domain reasoning. Don't throw every variable in—mediators (on the causal path) or colliders (affected by both) can introduce bias. Include variables that theory suggests are confounders and were measured before the treatment.

### Multicollinearity

**What is multicollinearity?** High correlation between predictors.

**Symptoms:**
- Coefficients change dramatically when you add/remove features
- High R² but few significant individual predictors
- Signs of coefficients seem wrong

### Detecting Multicollinearity: VIF

**Variance Inflation Factor (VIF):**

$$VIF_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is R² from regressing feature j on all other features.

| VIF Value | Interpretation |
|-----------|----------------|
| VIF = 1 | No correlation with other features |
| VIF > 5 | Moderate multicollinearity—investigate |
| VIF > 10 | Serious multicollinearity—must address |

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculate VIF for each feature
for i in range(X.shape[1]):
    vif = variance_inflation_factor(X.values, i)
    print(f"{feature_names[i]}: VIF = {vif:.2f}")
```

**Solutions for multicollinearity:**
1. Remove one of the correlated features
2. Combine features (average or PCA)
3. Use regularization (Ridge handles this well)
4. Accept it if prediction is your only goal

Multicollinearity doesn't affect prediction accuracy—but it creates problems for interpretation. Coefficients become unstable (jumping around with small data changes), standard errors inflate (true effects appear "not significant"), and signs may reverse. If your goal is purely prediction, ignore it. If you need interpretation, address it.

### Why Regularize?

Regularization prevents overfitting by penalizing large coefficients:

$$\text{minimize } \sum(y_i - \hat{y}_i)^2 + \lambda \cdot \text{penalty}(\beta)$$

The parameter $\lambda$ controls how strong the penalty is.

Large coefficients can indicate overfitting—the model making sharp adjustments for individual data points (memorizing). Signs of overfitting: large coefficients only with small samples, huge standard errors, or poor test performance. Regularization forces the model to justify large coefficients—if fitting noise, the penalty isn't worth it; if fitting real patterns, the benefit outweighs the penalty. Note: "large" depends on feature scale—always standardize before judging.

### L1 Regularization (Lasso)

**Penalty:** $\lambda \sum|\beta_j|$

**Effect:** Can shrink coefficients **exactly to zero** → automatic feature selection!

The L1 constraint region is a diamond with corners on the axes; L2 is a smooth circle. Loss function contours typically hit the L1 diamond at corners (coefficients exactly zero), but touch the L2 circle tangentially (rarely on an axis). Additionally, the L1 gradient is constant (±1) regardless of how small β gets—always pulling toward zero—while the L2 gradient (2β) weakens as β approaches zero. L1 performs automatic feature selection; L2 keeps all features with small coefficients.

**When to use:**
- You suspect many features are irrelevant
- You want an interpretable, sparse model
- Feature selection is an explicit goal

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1, random_state=42)
lasso.fit(X_train_scaled, y_train)

# See which features were selected (non-zero coefficients)
selected_features = [f for f, c in zip(feature_names, lasso.coef_) if c != 0]
print(f"Selected {len(selected_features)} features")
```

### L2 Regularization (Ridge)

**Penalty:** $\lambda \sum\beta_j^2$

**Effect:** Shrinks all coefficients toward zero, but **never exactly zero**

**When to use:**
- Multicollinearity is present
- All features are potentially relevant
- Prediction accuracy is the main goal

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
```

### Elastic Net: Combining L1 and L2

**Penalty:** $\lambda_1 \sum|\beta_j| + \lambda_2 \sum\beta_j^2$

**Benefits:**
- Feature selection from L1 (can zero out coefficients)
- Stability from L2 (handles correlated features better)
- More flexible than either alone

```python
from sklearn.linear_model import ElasticNet

elastic = ElasticNet(
    alpha=0.1,       # Overall regularization strength
    l1_ratio=0.5,    # Balance between L1 and L2
    random_state=42
)
elastic.fit(X_train_scaled, y_train)
```

### Choosing Regularization Strength

Use cross-validation:

```python
from sklearn.linear_model import LassoCV, RidgeCV

# Automatic alpha selection via CV
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 50),
    cv=5
)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {lasso_cv.alpha_}")
```

**Important:** Always scale your features before regularization! Regularization penalizes large coefficients, and features on different scales are penalized unfairly.

To convert scaled coefficients back to original units: $\beta_{original} = \frac{\beta_{scaled}}{\sigma}$. In code: `original_coefs = model.coef_ / scaler.scale_`. Back-transform for business communication ("each $1000 in spending → X more sales"); keep scaled for comparing feature importance. Save your scaler object so you have access to `mean_` and `scale_` attributes.

### Linear Regression as a Neural Network

Here's something that will pay off in Module 6:

```
Input Layer          Output Layer

   x₁ ──── w₁ ────┐
                   ├──→ Σ + b ──→ ŷ
   x₂ ──── w₂ ────┘
```

Linear regression is just a neural network with no hidden layers!
- The weights (w) are our coefficients ($\beta$)
- The bias (b) is our intercept ($\beta_0$)
- We sum the weighted inputs and add the bias

When we add hidden layers and non-linear activations, we get deep learning. But the foundation—weighted sums optimized by gradient descent—is exactly what we learned here.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "Regularization always hurts training performance" | True, but that's the point! We sacrifice training fit for better generalization. |
| "Lasso always performs feature selection" | Only with sufficient regularization. Very small alpha may keep all features. |
| "More features always improve the model" | Only if they're informative. Irrelevant features add noise and overfitting risk. |
| "Ridge is inferior because it doesn't zero out coefficients" | Ridge is often better when all features matter. Lasso is for sparse solutions. |

---

## 2.3 Business Application

### End-to-End Regression Workflow

1. **Business problem definition** - What are we predicting? Why does it matter?
2. **Data collection and exploration** - EDA, quality assessment
3. **Feature engineering** - Create informative variables from raw data
4. **Model building and evaluation** - Compare approaches, cross-validate
5. **Interpretation and communication** - Translate for stakeholders

Most of your time should go into steps 1-3. The modeling itself is almost mechanical once you have good data and features.

**When is "enough" feature engineering?** Track validation performance as you add features—stop when new features don't improve it. The 80/20 rule applies: basic features (raw data, simple transforms) usually dominate; exotic interactions rarely add much. Don't exceed n/10 to n/20 features for n samples without regularization. Start simple, add complexity where residual analysis suggests it, and stop when validation performance plateaus.

### Feature Engineering Patterns

**Time-based features:**
```python
import polars as pl

df = df.with_columns([
    pl.col('date').dt.weekday().alias('day_of_week'),
    pl.col('date').dt.month().alias('month'),
    pl.col('date').dt.weekday().is_in([5, 6]).cast(pl.Int32).alias('is_weekend'),
])
```

**Aggregations:**
```python
customer_features = df.group_by('customer_id').agg([
    pl.col('amount').sum().alias('total_spend'),
    pl.col('amount').mean().alias('avg_order'),
    pl.col('amount').count().alias('order_count'),
])
```

**Interactions:**
```python
df = df.with_columns([
    (pl.col('price') / pl.col('sqft')).alias('price_per_sqft'),
])
```

**Polynomial features:**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

### Translating Statistics to Business Language

| Statistical Term | Business Translation |
|-----------------|---------------------|
| Coefficient = 2.5 | "Every $1 more in advertising is associated with $2.50 more in sales" |
| R² = 0.75 | "Our model explains about 75% of the variation in sales" |
| p-value < 0.05 | "We're confident this factor genuinely affects sales, not just by chance" |
| 95% CI: [1.8, 3.2] | "We estimate the effect is between $1.80 and $3.20 per dollar spent" |

**Standardized vs. unstandardized coefficients:**
- **Unstandardized** (original units): For business interpretation—"Every $1 in advertising → $2.50 in sales"
- **Standardized** (z-scores): For comparing feature importance—"Which variable has the biggest effect?"

### Sensitivity Analysis

Stakeholders love "what-if" scenarios:

```python
# Base prediction
base = model.predict(X_current)

# Modified scenario
X_modified = X_current.copy()
X_modified['advertising'] *= 1.10
new = model.predict(X_modified)

print(f"10% more advertising → ${new - base:,.0f} more sales")
```

### The Business Memo

**Structure:**

1. **Executive Summary** - 2-3 sentences. Main finding plus recommendation.
2. **Key Findings** - Bullet points with business impact.
3. **Recommendations** - Specific, actionable steps.
4. **Limitations** - What the analysis cannot tell us.

**Rules:**
- No code
- No jargon without explanation
- No p-values without context
- Always include limitations

The limitations section matters. Every analysis has limitations. If you don't acknowledge them, you're either unaware (bad) or hiding them (worse).

Frame limitations as **scope definition, not weakness**. Instead of "The model doesn't account for competitor pricing," say "The model predicts based on our historical data. For decisions involving major competitor moves, supplement with competitive intelligence." Quantify uncertainty ("accurate to within ±15% 80% of the time") rather than saying "predictions might be wrong." Lead with capabilities, then contextualize what's left. Provide recommendations within limitations ("Given ±15% uncertainty, keep 20% buffer inventory"). Models presented as perfect lose credibility when they fail; honest limitations build trust.

---

## Reflection Questions

1. You implement gradient descent and the loss keeps increasing. What's likely wrong? How would you fix it?

2. A regression coefficient for 'ice cream sales' on 'drowning deaths' is positive and statistically significant. Should ice cream vendors be concerned about causing drownings?

3. Your model has R² = 0.95 but the residual plot shows a clear curved pattern. Is this a good model?

4. When would you prefer Lasso over Ridge regression? Give a business scenario.

5. You add more features to your model and R² on training data increases, but test set performance decreases. What's happening?

6. How would you explain regularization to a business stakeholder without using math?

---

## Practice Problems

1. Given a fitted model equation, interpret each coefficient in business terms.

2. Diagnose issues from residual plots (identify non-linearity, heteroscedasticity).

3. Calculate VIF and decide which features to remove.

4. Choose between Lasso, Ridge, and ElasticNet for different scenarios.

5. Write a business memo interpreting regression results for a non-technical audience.

---

## Chapter Summary

**Six key takeaways from Module 2:**

1. **Linear regression minimizes squared errors** to find the best-fit line. The math is elegant, but the intuition is simple: find the line that makes predictions as close as possible to reality.

2. **Gradient descent iteratively optimizes parameters**. This is the foundation for all neural network training. Learning rate matters: too small is slow, too large is unstable.

3. **Coefficients show association, not causation**. Be careful how you communicate this. "Associated with" is not "causes."

4. **Residual analysis reveals assumption violations**. Always check your residual plots. High R² with a patterned residual plot is a bad model.

5. **Regularization prevents overfitting**. Lasso selects features, Ridge handles multicollinearity, Elastic Net does both.

6. **Communication matters**. Translate statistics to business impact. Include limitations. Make it actionable.

---

## What's Next

In Module 3, we tackle **Classification Methods**:
- Logistic regression (extends linear regression to classification)
- Decision boundaries and probability estimation
- Classification metrics in depth
- Handling imbalanced classes

You'll apply everything from Module 2:
- Same data preparation workflow
- Gradient descent concepts
- Regularization techniques

The difference: we're predicting categories instead of numbers.
