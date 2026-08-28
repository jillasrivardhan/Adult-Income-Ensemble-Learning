# 💰 Adult Income Prediction — Ensemble Learning

A machine learning classification project that predicts whether a person's annual income is **greater than $50K or less than or equal to $50K** using demographic and employment-related features from the **Adult Income Dataset**.

This project focuses on **data preprocessing, categorical encoding, ensemble learning algorithms, and model evaluation** using Python and Scikit-learn.

---

## 📌 Project Overview

The goal of this project is to build machine learning models that predict an individual's income category based on features such as:

* Age
* Workclass
* Education
* Marital Status
* Occupation
* Relationship
* Race
* Sex
* Capital Gain
* Capital Loss
* Hours per Week
* Native Country

The target variable is:

```text
income
```

with two possible classes:

```text
<=50K
>50K
```

Several **ensemble learning techniques** are implemented and compared to determine which approach performs best on the dataset.

---

## 🎯 Objectives

The main objectives of this project are:

* Perform data cleaning and preprocessing.
* Handle missing values represented by `?`.
* Separate features and target variables.
* Apply **Label Encoding** to the target variable.
* Apply **One-Hot Encoding** to categorical feature columns.
* Split the dataset into training and testing sets.
* Implement multiple ensemble learning algorithms.
* Compare model performance.
* Evaluate models using accuracy, precision, recall, F1-score, and confusion matrix.
* Understand the advantages and differences between ensemble techniques.

---

## 📂 Dataset

The project uses the **Adult Income Dataset**, commonly known as the **Census Income Dataset**.

Each row represents an individual and contains demographic, employment, and financial information.

### Target Variable

```text
income
```

Possible values:

```text
<=50K
>50K
```

The machine learning problem is therefore a **binary classification problem**.

---

## 🧹 Data Preprocessing

Before training the models, several preprocessing steps are performed.

### 1. Handling Missing Values

The dataset represents missing categorical values using:

```text
?
```

These values are converted to `NaN` and rows containing missing values are removed.

```python
df = df.replace("?", np.nan)
df = df.dropna()
```

---

### 2. Removing Redundant Feature

The dataset contains both:

```text
education
education-num
```

`education-num` already provides a numerical representation of education level, so the original `education` column is removed in this project.

```python
df = df.drop("education", axis=1)
```

---

## 🔢 Encoding

Categorical variables cannot be directly used by most machine learning algorithms, so they need to be converted into numerical representations.

### Label Encoding

**LabelEncoder** is used for the target variable:

```text
<=50K → 0
>50K  → 1
```

Example:

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)
```

### Why Label Encoding?

The target contains only two categories, so representing them as `0` and `1` is appropriate.

---

### One-Hot Encoding

Categorical feature columns are converted using `OneHotEncoder`.

The categorical columns include:

```text
workclass
marital-status
occupation
relationship
race
sex
native-country
```

Example:

```python
from sklearn.preprocessing import OneHotEncoder

OneHotEncoder(handle_unknown="ignore")
```

Instead of assigning artificial numerical values such as:

```text
Private → 0
State-gov → 1
Self-emp → 2
```

One-Hot Encoding creates separate binary columns.

For example:

```text
workclass_Private
workclass_State-gov
workclass_Self-emp
```

This prevents the model from assuming that one category is numerically greater or smaller than another.

---

## 🔀 Train-Test Split

The dataset is divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Split

```text
80% → Training
20% → Testing
```

`stratify=y` is used to maintain a similar target-class distribution in both sets.

---

# 🌳 Ensemble Learning

Ensemble learning combines multiple machine learning models to produce a stronger predictive model.

Instead of relying on one model, ensemble methods use multiple models and combine their predictions.

This project explores five ensemble techniques:

1. Random Forest
2. Extra Trees
3. Gradient Boosting
4. Voting Classifier
5. Stacking Classifier

---

# 1️⃣ Random Forest 🌲

Random Forest is a **Bagging-based ensemble algorithm**.

It creates many decision trees and combines their predictions.

### How it works

```text
Training Data
     │
     ├──→ Decision Tree 1 ──→ Prediction
     │
     ├──→ Decision Tree 2 ──→ Prediction
     │
     ├──→ Decision Tree 3 ──→ Prediction
     │
     ├──→ ...
     │
     └──→ Decision Tree N ──→ Prediction
                    │
                    ▼
             Final Prediction
```

For classification, the trees vote for the final class.

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

### Advantages

* Reduces overfitting compared with a single decision tree.
* Handles nonlinear relationships.
* Works well with high-dimensional data.
* Generally provides strong baseline performance.

---

# 2️⃣ Extra Trees 🌳

Extra Trees stands for **Extremely Randomized Trees**.

It is similar to Random Forest but introduces additional randomness when constructing decision trees.

### Random Forest vs Extra Trees

```text
Random Forest
      ↓
Random samples + randomized feature selection
      ↓
Multiple Decision Trees
      ↓
Final Prediction
```

```text
Extra Trees
      ↓
Random samples/features + highly randomized splits
      ↓
Multiple Decision Trees
      ↓
Final Prediction
```

### Implementation

```python
from sklearn.ensemble import ExtraTreesClassifier

model = ExtraTreesClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

### Advantages

* Often fast to train.
* Reduces variance.
* Strong performance on many tabular datasets.
* Adds more randomness than Random Forest.

---

# 3️⃣ Gradient Boosting 🚀

Gradient Boosting is a **Boosting algorithm**.

Unlike Random Forest, where trees are generally built independently, Gradient Boosting builds models **sequentially**.

Each new model attempts to improve upon the errors made by previous models.

### How it works

```text
Training Data
     │
     ▼
Decision Tree 1
     │
     ▼
Find Errors
     │
     ▼
Decision Tree 2
     │
     ▼
Correct Previous Errors
     │
     ▼
Decision Tree 3
     │
     ▼
Continue...
     │
     ▼
Final Prediction
```

### Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
```

### Important Parameters

**`n_estimators`**

Number of boosting stages/trees.

**`learning_rate`**

Controls how much each new tree contributes to the final model.

**`max_depth`**

Controls the complexity of individual trees.

### Advantages

* Excellent performance on tabular data.
* Can capture complex relationships.
* Often produces highly accurate models.

---

# 4️⃣ Voting Classifier 🗳️

Voting combines several different machine learning models.

In this project, the Voting Classifier combines:

```text
Random Forest
Extra Trees
Gradient Boosting
```

### How it works

```text
              ┌── Random Forest ──→ Prediction
              │
Input Data ───┼── Extra Trees ────→ Prediction
              │
              └── Gradient Boost ─→ Prediction
                       │
                       ▼
                 Voting System
                       │
                       ▼
                Final Prediction
```

This project uses:

```python
voting="soft"
```

### Soft Voting

Soft voting uses the predicted probabilities from each model and combines them.

```python
VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier()),
        ("et", ExtraTreesClassifier()),
        ("gb", GradientBoostingClassifier())
    ],
    voting="soft"
)
```

### Advantages

* Combines different models.
* Can improve stability.
* Uses information from multiple classifiers.
* Soft voting can benefit from probability estimates.

---

# 5️⃣ Stacking Classifier 🧠

Stacking is a more advanced ensemble technique.

It trains several **base models** and then uses another model called the **meta-model** to learn from their predictions.

### Structure

```text
                    Input Data
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     Random Forest  Extra Trees  Gradient Boosting
          │             │             │
          ▼             ▼             ▼
       Prediction   Prediction    Prediction
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                  Meta Model
                        │
                        ▼
                 Final Prediction
```

In this project:

```text
Base Models:
Random Forest
Extra Trees
Gradient Boosting

Meta Model:
Random Forest
```

### Implementation

```python
from sklearn.ensemble import StackingClassifier

model = StackingClassifier(
    estimators=[
        ("rf", RandomForestClassifier()),
        ("et", ExtraTreesClassifier()),
        ("gb", GradientBoostingClassifier())
    ],
    final_estimator=RandomForestClassifier(),
    n_jobs=-1
)
```

### Advantages

* Combines different model strengths.
* Can learn how to combine predictions.
* Often useful when different models make different types of errors.

---

# ⚖️ Bagging vs Boosting vs Voting vs Stacking

| Technique             | Main Idea                                          |
| --------------------- | -------------------------------------------------- |
| **Random Forest**     | Many decision trees trained independently          |
| **Extra Trees**       | Highly randomized decision trees                   |
| **Gradient Boosting** | Trees built sequentially to correct errors         |
| **Voting**            | Multiple models vote/combine predictions           |
| **Stacking**          | Multiple models feed predictions into a meta-model |

### Simple way to remember

```text
Bagging
→ Build many models independently
→ Combine their predictions

Boosting
→ Build models sequentially
→ Each model improves previous mistakes

Voting
→ Multiple different models vote

Stacking
→ Multiple models make predictions
→ Another model learns how to combine them
```

---

# 📊 Model Evaluation

The models are evaluated using:

### Accuracy

Percentage of correctly classified samples.

```text
Accuracy =
Correct Predictions / Total Predictions
```

---

### Precision

Of the samples predicted as a particular class, how many were actually that class?

```text
Precision =
True Positives / (True Positives + False Positives)
```

---

### Recall

Of all actual samples belonging to a class, how many did the model correctly identify?

```text
Recall =
True Positives / (True Positives + False Negatives)
```

---

### F1-Score

The harmonic mean of precision and recall.

```text
F1 =
2 × (Precision × Recall)
-------------------------
   Precision + Recall
```

---

### Confusion Matrix

A confusion matrix shows how many predictions were:

* True Positive
* True Negative
* False Positive
* False Negative

It helps understand which classes the model is confusing.

---

# 📈 Model Comparison

The project compares the performance of:

```text
Random Forest
Extra Trees
Gradient Boosting
Voting Classifier
Stacking Classifier
```

Example comparison table:

| Model               | Accuracy |
| ------------------- | -------: |
| Random Forest       |        — |
| Extra Trees         |        — |
| Gradient Boosting   |        — |
| Voting Classifier   |        — |
| Stacking Classifier |        — |

The actual values are generated when the notebook is executed.

The best model should not necessarily be selected based only on accuracy. Precision, recall, F1-score, and the confusion matrix should also be considered.

---

# ⚠️ Class Imbalance

The Adult Income dataset contains more samples in the `<=50K` class than in the `>50K` class.

Therefore, accuracy alone may not provide the complete picture.

For a more detailed evaluation, this project uses:

```text
Precision
Recall
F1-score
```

especially for the minority class.

Future versions of the project can experiment with:

* `class_weight="balanced"`
* Oversampling
* SMOTE
* Undersampling
* Threshold tuning

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Google Colab**

---

# 📁 Project Structure

```text
Adult-Income-Ensemble-Learning/
│
├── adult.data
├── Adult_Income_Ensemble_Learning.ipynb
├── README.md
└── requirements.txt
```

---

# 🚀 How to Run

### Option 1 — Google Colab

1. Open Google Colab.
2. Upload the notebook.
3. Upload `adult.data`.
4. Run the cells sequentially.
5. Review the model evaluation results.
6. Compare the ensemble algorithms.

---

### Option 2 — Local Environment

Clone the repository:

```bash
git clone https://github.com/your-username/Adult-Income-Ensemble-Learning.git
```

Navigate to the project:

```bash
cd Adult-Income-Ensemble-Learning
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

Run the Jupyter Notebook:

```bash
jupyter notebook
```

---

# 📦 Requirements

Create a `requirements.txt` file:

```text
pandas
numpy
scikit-learn
matplotlib
```

---

# 💡 Key Learnings

Through this project, I practiced and learned:

* Data loading and cleaning
* Handling missing values
* Feature and target separation
* Label Encoding
* One-Hot Encoding
* Train-test splitting
* Pipelines
* ColumnTransformer
* Random Forest
* Extra Trees
* Gradient Boosting
* Voting Ensemble
* Stacking Ensemble
* Classification metrics
* Confusion matrices
* Model comparison
* Class imbalance
* Ensemble learning concepts

---

# 🔮 Future Improvements

Possible improvements include:

* Hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`
* Cross-validation
* SMOTE for handling class imbalance
* Feature importance analysis
* ROC-AUC evaluation
* Precision-Recall curves
* Model explainability using SHAP
* Deployment using Streamlit
* Building an interactive prediction interface
* Comparing additional boosting algorithms such as XGBoost, LightGBM, or CatBoost

---

# 👨‍💻 Author

**Jillasri Vardhan**

This project was created as part of my machine learning and ensemble learning practice.

---

## ⭐ Conclusion

This project demonstrates how ensemble learning can be applied to a real-world binary classification problem.

By implementing and comparing **Random Forest, Extra Trees, Gradient Boosting, Voting, and Stacking**, the project provides practical experience with different ensemble strategies and shows how combining multiple models can improve the robustness of machine learning solutions.

If you found this project useful, consider giving the repository a ⭐.
