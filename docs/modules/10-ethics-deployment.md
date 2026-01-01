# Module 10: Ethics, Deployment & Real-World ML

## Introduction

We've covered an incredible amount of ground in this course. You can build regression models, classification models, ensemble methods, neural networks, CNNs for images, transformers for text. You can interpret models with SHAP and LIME. You know how to evaluate, tune, and avoid common pitfalls.

But here's the thing: **Building a model is only half the journey.**

This module tackles the other half—getting models into the real world responsibly and effectively. This means grappling with ethics and fairness, learning time series forecasting, understanding deployment, and calculating business value.

These topics bridge technical skills to real-world impact. Every data scientist who wants to make a difference needs to master them.

**Responsibility for fairness**: All three share responsibility. Data scientists are the first line of defense and should raise concerns. Companies set culture, allocate resources for fairness audits, and establish review processes—they're culpable for pressuring fast deployment without ethical review. Regulators provide external accountability that markets fail to create. The healthiest ecosystem has all three layers; relying on any single one is insufficient.

---

## Learning Objectives

By the end of this module, you should be able to:

1. **Identify** sources of bias in ML systems and propose mitigation strategies
2. **Apply** fairness metrics to evaluate model equity across groups
3. **Build** time series forecasting models using appropriate techniques
4. **Explain** the basics of model deployment and MLOps
5. **Calculate** business value and ROI of ML projects
6. **Communicate** uncertainty and manage stakeholder expectations

---

## 10.1 Ethics & Responsible AI

### The Stakes Are High

ML systems are making consequential decisions: who gets a loan, who gets a job, who gets parole, who gets medical treatment. These aren't abstract technical problems—they affect real people's lives.

And here's the uncomfortable truth: **ML systems can be biased. They can be unfair. They can cause harm.**

Not because the engineers are malicious, but because bias creeps in through data, design choices, and blind spots. If we're going to deploy these systems, we need to understand how bias arises and how to mitigate it.

### Sources of Bias in ML

**Historical Bias**: Training data reflects past discrimination.
- If you train on 10 years of hiring data, and that data reflects historical biases against women or minorities, your model learns those biases.
- The model isn't "biased by itself"—it's learning patterns from biased data.

**Selection Bias**: Training data doesn't represent the population.
- A medical AI trained mostly on data from white patients may perform worse on underrepresented groups.
- The model has never learned the patterns for those populations.

**Measurement Bias**: Features are measured differently across groups.
- "Years of experience" penalizes career gaps, which disproportionately affects women.
- "Arrests" doesn't mean "crimes committed"—it reflects policing patterns.

**Aggregation Bias**: One model for heterogeneous populations.
- A single diabetes prediction model may work differently across ethnicities.
- Sometimes you need separate models or careful feature engineering.

**Feedback Loops**: Model predictions affect future data.
- Predictive policing sends more officers to certain neighborhoods → more arrests → more "crime" data → model sends even more officers.
- The bias becomes self-reinforcing.

### Case Study: Amazon Hiring Tool

In 2018, it was reported that Amazon had built a hiring tool trained on 10 years of resume data.

**What went wrong:**
- The model learned to penalize words like "women's" (as in "women's chess club captain")
- It downgraded graduates of women's colleges
- It effectively discriminated against female applicants

**The lesson**: Historical data encodes historical bias. Amazon's tech workforce was predominantly male. The model learned that being male correlated with getting hired. It wasn't explicitly told "penalize women," but it learned it from the patterns.

Amazon scrapped the tool.

### Case Study: COMPAS

COMPAS is a recidivism prediction algorithm used in the US criminal justice system to predict whether defendants will reoffend.

**ProPublica's analysis found:**
- Black defendants had a higher **false positive rate** (incorrectly flagged as high risk)
- White defendants had a higher **false negative rate** (incorrectly flagged as low risk)
- Same overall accuracy, very different error patterns

This shows that identical accuracy can hide profoundly different impacts on different groups.

**Choosing between fairness criteria**: This is an ethical decision, not technical—it shouldn't be made solely by data scientists. The data scientist's role is to make tradeoffs transparent ("if we optimize for A, here's what happens to X and Y"), not to unilaterally decide. These decisions should involve domain experts, affected communities, legal experts, and ethicists. Document the decision, reasoning, and who was involved.

### Fairness Metrics

There are multiple mathematical definitions of fairness:

**Demographic Parity** (Statistical Parity):
$$P(\hat{Y}=1|A=0) = P(\hat{Y}=1|A=1)$$
- Equal positive prediction rates across groups
- If 30% of men get approved, 30% of women should get approved
- **Limitation**: Ignores actual qualification rates

**Equalized Odds**:
$$P(\hat{Y}=1|Y=1, A=0) = P(\hat{Y}=1|Y=1, A=1)$$
$$P(\hat{Y}=1|Y=0, A=0) = P(\hat{Y}=1|Y=0, A=1)$$
- Equal true positive rates AND equal false positive rates across groups
- If you're qualified, you should have equal chance of being accepted regardless of group
- If you're unqualified, you should have equal chance of being rejected

**Predictive Parity**:
$$P(Y=1|\hat{Y}=1, A=0) = P(Y=1|\hat{Y}=1, A=1)$$
- Equal precision across groups
- If the model says "yes," the probability of actually being qualified should be the same across groups

### The Impossibility Theorem

**You cannot satisfy all fairness criteria simultaneously** (except in special cases).

This isn't a technical limitation—it's mathematically proven. If groups have different base rates (different proportions of positive outcomes), you have to choose which fairness criterion matters most.

**Example:**

| Group | Accuracy | FPR | FNR |
|-------|----------|-----|-----|
| A | 85% | 10% | 20% |
| B | 85% | 25% | 5% |

Same accuracy. But Group B has more false positives (more people incorrectly flagged). Group A has more false negatives (more people incorrectly missed).

Which is worse depends on context. In criminal justice, high FPR means innocent people in jail. In medical diagnosis, high FNR means sick people going untreated.

### Calculating Fairness Metrics

```python
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Calculate metrics by group
metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'precision': precision_score,
        'recall': recall_score
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=demographic_feature
)

# View metrics by group
print(metric_frame.by_group)

# View maximum difference between groups
print(metric_frame.difference())
```

The `fairlearn` library makes this straightforward. Calculate your metrics by demographic group and look for disparities.

### Proxy Variables

"We don't use race in our model, so it can't be biased."

This is wrong.

**Proxy variables** are features that correlate with protected attributes:
- ZIP code correlates with race and income
- Name can indicate gender or ethnicity
- Arrest history correlates with race (due to policing patterns)
- "Years since last job" correlates with gender (career gaps)

Removing the protected attribute doesn't remove the bias if proxies remain.

### When NOT to Use ML

Not every problem needs machine learning.

**Consider avoiding ML when:**
- Stakes are very high and errors are catastrophic
- Accountability and explanation are paramount
- Training data is fundamentally biased
- The problem is better solved by policy
- Human judgment is essential

**Questions to ask:**
- Who is affected by this system?
- What happens when it's wrong?
- Can we explain decisions to affected parties?
- Is the training data representative?
- Are we automating an already unfair process?

Sometimes the right answer is "don't build this model."

**Pushing back on unethical projects**: Document concerns and frame in terms of business risk (legal liability, reputational damage). Escalate through appropriate channels—ethics hotlines, ombudspersons. If internal advocacy fails: comply under protest (documented), refuse the project (accept consequences), or leave. Building a financial cushion gives leverage. Long-term: seek employers whose values align with yours—ask about ethics review processes during interviews.

### Common Misconceptions

| Misconception | Reality |
|--------------|---------|
| "ML is objective because it's math" | ML learns patterns from data, including human biases |
| "Equal accuracy = fairness" | Same accuracy can hide very different error patterns across groups |
| "Just remove protected attributes" | Proxy variables can encode same information |
| "Fairness is a purely technical problem" | Requires ethical choices that should involve diverse stakeholders |

---

## 10.2 Time Series Forecasting

### Why Time Series Is Different

Time series data has a unique property: **temporal ordering matters**.

In standard ML, we assume observations are independent—shuffling rows shouldn't matter. In time series, shuffling destroys the information. Yesterday's sales tell you something about today's sales. January's patterns repeat every January.

This changes everything about how we model and validate.

### Time Series Components

Time series can be decomposed into components:

**Trend**: Long-term direction (sales growing over years)

**Seasonality**: Regular patterns (sales spike every December)

**Cyclical**: Irregular longer-term fluctuations (economic cycles)

**Noise**: Random variation

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    series,
    model='additive',
    period=12
)
decomposition.plot()
```

Understanding these components helps you choose the right model and spot problems.

### ARIMA Models

ARIMA is the classic statistical approach to time series.

**ARIMA(p, d, q):**
- **AR (AutoRegressive)**: Predict from past values (how many lags to use = p)
- **I (Integrated)**: Differencing for stationarity (how many times to difference = d)
- **MA (Moving Average)**: Predict from past errors (how many lag errors to use = q)

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train_series, order=(1, 1, 1))
results = model.fit()
forecast = results.forecast(steps=30)
```

**Choosing p, d, q:**
- ACF/PACF plots give guidance
- AIC/BIC criteria for model selection
- Or use auto_arima:

```python
from pmdarima import auto_arima

model = auto_arima(
    train_series,
    seasonal=True,
    m=12,  # Monthly seasonality
    trace=True
)
```

Auto_arima searches through parameter combinations and picks the best one.

### Prophet

Facebook's Prophet is a popular alternative to ARIMA.

**Advantages:**
- Handles seasonality automatically (multiple seasonalities!)
- Robust to missing data
- Interpretable components
- Easy to add holidays and special events

```python
from prophet import Prophet

# Data must have columns 'ds' (date) and 'y' (value)
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True
)
model.fit(train_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

model.plot(forecast)
model.plot_components(forecast)  # Shows trend, seasonality, etc.
```

Prophet is particularly good for business applications with strong seasonal patterns.

### When to Choose What

- **Prophet**: Multiple seasonalities, missing data, holidays, interpretable components
- **ARIMA**: More control, complex series that don't fit Prophet's assumptions, very short series
- **LSTM**: Complex non-linear patterns, multiple input features, long sequences

### Time Series Validation

**You cannot use standard k-fold cross-validation for time series.**

Why? Because it would leak future information into training. If your test set contains January 2024 and your training set contains February 2024, you're cheating—you're using the future to predict the past.

**Walk-forward validation:**
```
Train: [----]          Test: [-]
Train: [------]        Test: [-]
Train: [--------]      Test: [-]
```

Always train on the past, test on the future. Never the reverse.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(data):
    train = data[train_idx]
    test = data[test_idx]
    # Train and evaluate
```

### Business Applications

Time series forecasting is everywhere in business:
- **Sales forecasting**: Budget planning, resource allocation
- **Demand planning**: Inventory management, supply chain
- **Capacity planning**: Staffing, infrastructure
- **Financial forecasting**: Revenue projections, cash flow

---

## 10.3 Model Deployment

### From Notebook to Production

You've built a model in a Jupyter notebook. It works great. Now what?

**The gap between "model works" and "model is deployed" is significant:**
- How do other systems call your model?
- How do you handle errors?
- How do you scale?
- How do you update the model?
- How do you monitor performance?

This is where software engineering meets data science.

### Model Serialization

First, you need to save your model so it can be loaded elsewhere.

**Pickle/Joblib** (for scikit-learn models):
```python
import joblib

# Save model
joblib.dump(model, 'model.pkl')

# Load model
model = joblib.load('model.pkl')
```

**ONNX** (cross-platform format):
- Works across different frameworks (PyTorch, TensorFlow, scikit-learn)
- Optimized for inference
- Useful when production environment differs from development

```python
import torch.onnx

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)
```

### Creating APIs

To let other systems use your model, wrap it in an API.

**Flask** (simple, widely used):
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['features']
    prediction = model.predict([features])
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**FastAPI** (modern, automatic documentation):
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load('model.pkl')

class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
def predict(request: PredictionRequest):
    prediction = model.predict([request.features])
    return {"prediction": prediction[0]}
```

Now other applications can send HTTP requests to get predictions.

### Containerization with Docker

Docker packages your application with all its dependencies.

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 5000
CMD ["python", "app.py"]
```

**Benefits:**
- **Reproducibility**: Same environment everywhere
- **Portability**: Runs on any system with Docker
- **Scalability**: Easy to run multiple containers
- **Isolation**: Dependencies don't conflict

### Cloud Deployment Options

| Platform | Use Case | Complexity |
|----------|----------|------------|
| AWS SageMaker | Full ML platform | Medium |
| Google Vertex AI | Full ML platform | Medium |
| Azure ML | Full ML platform | Medium |
| Heroku | Simple web apps | Low |
| AWS Lambda | Serverless, simple models | Low |

For production ML at scale, the major cloud platforms provide integrated solutions. For simple models or prototypes, Heroku or Lambda can be quick to set up.

---

## 10.4 Production & Business Value

### Monitoring in Production

Deployment isn't the end—it's the beginning of a new phase.

**Metrics to track continuously:**
- **Model accuracy over time**: Is performance degrading?
- **Feature distributions**: Are inputs changing?
- **Latency and throughput**: Is the system fast enough?
- **Error rates**: Are there failures?

Set up alerts. If accuracy drops below a threshold, you want to know immediately.

### Model Drift

**Data drift**: Input distribution changes.
- Customer demographics shift
- Seasonality wasn't captured
- Data collection process changed

**Concept drift**: The relationship between inputs and outputs changes.
- Customer behavior changed (e.g., during COVID)
- What used to predict churn no longer does

```python
from evidently import Report
from evidently.metrics import DataDriftTable

report = Report(metrics=[DataDriftTable()])
report.run(
    reference_data=training_data,
    current_data=production_data
)
report.show()
```

Evidently AI and similar tools help detect when your data has shifted.

### Retraining Strategies

When drift is detected (or just periodically), you need to retrain.

**Scheduled retraining**: Weekly, monthly—on a fixed schedule

**Triggered retraining**: When drift exceeds a threshold

**Continuous training**: Update with each new batch of data

**Automate the pipeline:**
1. Data validation
2. Feature engineering
3. Model training
4. Evaluation (reject if metrics don't meet threshold)
5. Deployment

This is where MLOps comes in—applying DevOps principles to ML.

### A/B Testing

**Purpose**: Compare new model against current model

**Implementation:**
1. Route percentage of traffic to new model
2. Track metrics for both models
3. Statistical test for significance
4. Roll out winner

**Key considerations:**
- Sample size requirements
- Duration of test
- Guardrail metrics (don't harm user experience)

### ROI Calculation

How do you justify an ML project?

**Example: Churn prevention model**

```
Annual churning customers: 10,000
Customer lifetime value: $500
Churn cost without model: $5,000,000

Model performance:
- Identifies 75% of churners
- Intervention success rate: 30%
- Customers saved: 10,000 × 0.75 × 0.30 = 2,250
- Value saved: 2,250 × $500 = $1,125,000

Costs:
- Development: $100,000
- Annual maintenance: $20,000
- Intervention cost: $50 per flagged customer
- Total intervention cost: 7,500 × $50 = $375,000
- Total costs: $495,000

First year ROI: ($1,125,000 - $495,000) / $495,000 = 127%
```

**The key**: Quantify business impact, not just accuracy. "95% accuracy" means nothing to a CFO. "$630,000 net value in year one" does.

### Communicating Uncertainty

**Be honest about limitations:**
- Model accuracy is not 100%
- Performance varies across segments
- Future performance is not guaranteed
- Edge cases exist

**Use confidence intervals:**
- "We predict revenue of $1.2M ± $150K"
- "The model is 85% confident this customer will churn"

**Scenario analysis:**
- Best case / Base case / Worst case

Stakeholders appreciate honesty. Overpromising leads to disappointment and loss of trust.

### Managing Expectations

**Common pitfalls:**
- Overpromising accuracy
- Underestimating timeline
- Ignoring maintenance needs
- Assuming it's a one-time effort

**Best practices:**
- Start with a pilot project
- Set realistic expectations upfront
- Plan for iteration—first version won't be perfect
- Communicate progress regularly
- Budget for ongoing maintenance

ML models are more like products than projects. The world changes, data shifts, and models degrade. You wouldn't build a website and never update it. Same with ML models.

**Estimating maintenance costs**: Rule of thumb: 15-25% of initial development cost per year. Break down components: monitoring infrastructure, data pipeline maintenance, periodic retraining, model auditing, incident response. Staff time is usually the largest cost (one person at 20% time ≈ $30-50K/year). Present stakeholders with scenarios: "minimum maintenance" costs X with degradation risk; "recommended maintenance" costs Y with better reliability.

---

## Reflection Questions

1. Amazon's hiring tool was trained on successful employees. Why did it still produce biased results?

2. A model has 85% accuracy for both men and women. Is it fair? What else would you check?

3. Your model uses ZIP code, which correlates with race. Should you remove it? What are the trade-offs?

4. A hospital wants to use ML to allocate scarce medical resources. What ethical considerations arise?

5. Why can't we use regular k-fold cross-validation for time series?

6. When would you choose Prophet over ARIMA?

7. Your model works on your laptop but fails in production. What might cause this?

8. A model improves accuracy by 2% but costs $500K to develop. How do you decide if it's worth it?

9. How do you explain to a CFO that ML requires ongoing investment, not a one-time cost?

---

## Practice Problems

1. Calculate fairness metrics from confusion matrices for two demographic groups

2. Identify bias sources in a case study scenario

3. Design a monitoring plan for a deployed fraud detection model

4. Calculate ROI for an ML project given costs and projected benefits

5. Write a brief stakeholder communication explaining a model's limitations

---

## Chapter Summary

**Six key takeaways from Module 10:**

1. **Bias** enters ML through data and design; measure and mitigate it actively

2. **Fairness** has multiple incompatible definitions—you must choose

3. **Time series** requires temporal validation; never use k-fold

4. **Deployment** needs APIs, containers, and monitoring infrastructure

5. **Drift** happens; plan for detection and retraining

6. **Business value** must be quantified and communicated in dollars, not accuracy

---

## Course Summary

Take a moment to appreciate how far you've come.

**You've learned to:**
- Build ML models (regression, classification, clustering)
- Train ensemble methods and understand their power
- Train neural networks for structured data, images, and text
- Interpret and explain model decisions
- Deploy responsibly with fairness considerations and monitoring

**You understand:**
- The ML workflow from data to deployment
- Evaluation, validation, and avoiding common pitfalls
- When to use which technique
- How to communicate with stakeholders

**You're ready for the Capstone Project!**

---

## What's Next: Capstone Project

The capstone project is your chance to apply everything you've learned to a real problem.

You'll:
- Define a business problem
- Collect and prepare data
- Build and evaluate models
- Interpret and explain results
- Create deployment and monitoring plans
- Present to stakeholders

This is the culmination of the course—showing that you can take a problem from start to finish.
