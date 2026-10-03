# 🚴 DoorDash Support Ticket Triage

## 📌 Project Overview

This project builds a text classification system to automatically route DoorDash-style customer support tickets to the appropriate support team.

Customer messages can contain spelling mistakes, inconsistent capitalization, informal language, and multiple issues in the same ticket. The goal is to use the ticket text to predict one of six support categories and reduce the amount of manual routing required by support agents.

The project uses **TF-IDF with word unigrams and bigrams** and compares multiple traditional machine-learning classifiers.

> **Dataset:** 3,000 labelled training tickets and 1,000 held-out test tickets from August–September 2025.

---

## 🎯 Business Problem

Support tickets initially enter a common queue. An agent must read each message and manually route it to the correct team.

This creates several operational problems:

- Manual routing takes agent time.
- Urgent delivery issues may wait in the queue.
- Informal customer language makes keyword-only routing unreliable.
- Tickets can contain spelling errors and multiple problems.
- Inconsistent routing can affect downstream support performance.

### Objective

Build a machine-learning model that:

1. Reads the customer ticket text.
2. Predicts the appropriate support category.
3. Is evaluated using classification metrics.
4. Can be used to predict categories for unseen tickets.

---

## 🗂️ Dataset

### Training Dataset — `tickets_train.csv`

**3,000 rows × 6 columns**

| Column | Description |
|---|---|
| `ticket_id` | Unique ticket identifier |
| `created_at` | Timestamp when the ticket was created |
| `channel` | Ticket source: `chat`, `email`, or `app_form` |
| `order_value` | Related order value in USD |
| `text` | Customer's support message |
| `category` | Target support category |

### Test Dataset — `tickets_test.csv`

**1,000 rows × 5 columns**

The test dataset contains the same information except that `category` is not provided.

---

## 🏷️ Target Categories

The training data contains six support categories:

| Category | Tickets | Share |
|---|---:|---:|
| `late_delivery` | 833 | 27.77% |
| `missing_item` | 638 | 21.27% |
| `payment_issue` | 467 | 15.57% |
| `dasher_feedback` | 388 | 12.93% |
| `account_access` | 341 | 11.37% |
| `wrong_order` | 333 | 11.10% |
| **Total** | **3,000** | **100%** |

`late_delivery` is the largest class, while `wrong_order` is the smallest. The classes are therefore not perfectly balanced, making **macro F1** useful for evaluating whether performance is reasonable across all categories rather than being dominated by the largest class.

---

# 🔎 Exploratory Data Analysis

The notebook checks the structure of the training data and confirms:

- 3,000 training records.
- 6 columns.
- 3,000 unique `ticket_id` values.
- 6 target categories.
- 3 ticket channels: `chat`, `email`, and `app_form`.
- No missing values in the training dataset.
- `order_value` ranges from approximately **$9.01 to $119.99**.
- Customer messages contain informal language, spelling mistakes, capitalization differences, and punctuation variations.

### Examples of noisy customer text

The dataset contains examples such as:

- `forogt` instead of `forgot`
- `supoprt` instead of `support`
- `oredr` instead of `order`
- Fully capitalized complaints
- Messages mentioning more than one problem

This type of noise is important because a support-routing model must work with real customer-written text rather than perfectly cleaned sentences.

---

# 🧹 Text Preprocessing

A preprocessing function was applied to the ticket text.

### Steps

1. Convert text to lowercase.
2. Remove order numbers such as:
   - `order #123456`
   - `order 654321`
3. Remove prices such as:
   - `$75`
   - `$75.50`
4. Remove punctuation.
5. Normalize repeated whitespace.

Example transformation:

```text
Order #123456 missing item, cost $75.50
```

becomes approximately:

```text
missing item cost
```

The preprocessing removes identifiers and formatting noise while retaining the words needed for classification.

---

# 🧠 Feature Engineering

The project uses **TF-IDF (Term Frequency–Inverse Document Frequency)**.

```python
TfidfVectorizer(
    max_features=50000,
    stop_words='english',
    ngram_range=(1,2)
)
```

### Configuration

| Parameter | Value |
|---|---|
| Maximum features | 50,000 |
| Stop words | English |
| N-grams | Unigrams + bigrams |
| Feature representation | TF-IDF |

### Why TF-IDF?

TF-IDF gives more importance to words that are useful for distinguishing tickets while reducing the influence of very common words.

Using `(1,2)` n-grams also allows the model to capture short phrases such as:

- `late delivery`
- `missing item`
- `wrong order`
- `card declined`
- `account locked`

This is useful for support-ticket classification because the meaning of a phrase can be more informative than an individual word.

---

# 🤖 Models Evaluated

The notebook evaluates six classifiers:

1. Multinomial Naive Bayes
2. Complement Naive Bayes
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost

The data is split using an **80/20 stratified train-test split**, resulting in:

- Training set: 2,400 tickets
- Evaluation set: 600 tickets

Stratification preserves the relative category distribution in the split.

---

# 📊 Model Performance

The notebook reports the following results on the 600-ticket evaluation set.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| **MultinomialNB** | **0.90** | **0.89** | **0.90** |
| ComplementNB | 0.90 | 0.89 | 0.90 |
| DecisionTree | 0.85 | 0.84 | 0.85 |
| RandomForest | 0.89 | 0.89 | 0.89 |
| GradientBoosting | 0.90 | 0.89 | 0.90 |
| XGBClassifier | 0.90 | 0.88 | 0.89 |

### Selected model

The notebook uses **Multinomial Naive Bayes** as the final model.

Its evaluation results were:

- **Accuracy:** 90%
- **Macro F1:** 0.89
- **Weighted F1:** 0.90

The macro F1 of 0.89 indicates that the model performs reasonably consistently across the six categories rather than relying only on the largest class.

---

# 🔬 Multinomial Naive Bayes — Classification Results

The class labels produced by `LabelEncoder` correspond to:

| Encoded Label | Category |
|---:|---|
| 0 | `account_access` |
| 1 | `dasher_feedback` |
| 2 | `late_delivery` |
| 3 | `missing_item` |
| 4 | `payment_issue` |
| 5 | `wrong_order` |

The Multinomial Naive Bayes classification report from the notebook is:

| Category | Precision | Recall | F1 |
|---|---:|---:|---:|
| `account_access` | 0.89 | 0.91 | 0.90 |
| `dasher_feedback` | 0.89 | 0.83 | 0.86 |
| `late_delivery` | 0.90 | 0.96 | 0.93 |
| `missing_item` | 0.93 | 0.91 | 0.92 |
| `payment_issue` | 0.88 | 0.92 | 0.90 |
| `wrong_order` | 0.88 | 0.78 | 0.83 |
| **Macro Average** | **0.90** | **0.88** | **0.89** |

The strongest recall is for `late_delivery` at 0.96.

The lowest recall is for `wrong_order` at 0.78, indicating that this category is relatively more difficult for the model to identify.

---

# 🧩 Confusion Matrix

The Multinomial Naive Bayes confusion matrix is:

```text
[[62, 2, 1, 0, 1, 2],
 [ 1,65, 6, 1, 3, 2],
 [ 1, 0,160, 2, 3, 1],
 [ 1, 4, 4,115, 1, 2],
 [ 2, 2, 3, 0,86, 0],
 [ 3, 0, 3, 5, 4,52]]
```

Rows represent the actual class and columns represent the predicted class, following the encoded label mapping above.

The diagonal values show correctly classified tickets. Most errors occur between semantically related support categories, which is expected for messages containing overlapping problems.

---

# 🧪 Example Prediction

The notebook also demonstrates prediction on a new customer message:

```text
HI, RECEIVED GARLIC BREAD INSTEAD OF THE DUMPLINGS
```

Prediction:

```text
Category: wrong_order
Confidence: 44.41%
```

The relatively modest confidence is useful operationally: a ticket like this could be considered for human review rather than being treated as an unquestionable automated decision.

---

# 🚨 Important Evaluation Notes

The original task asks for three additional evaluation components:

1. A manually created **keyword baseline** evaluated using macro F1.
2. **Cross-validation** to compare models more robustly.
3. A manual review of **20 model mistakes**, grouped by root cause.

These steps are **not implemented in the supplied notebook**.

The current notebook instead uses a single **80/20 stratified train-test split** and reports the resulting classification metrics.

Therefore, the reported 0.89 macro F1 should be described as a **single hold-out evaluation result**, not as a cross-validated estimate.

Similarly, no verified keyword-baseline score or 20-error taxonomy is included here because those results cannot be derived from the notebook outputs without running the additional analysis.

---

# 🏁 Final Model and Test Predictions

The notebook selects:

```python
final_model = MultinomialNB()
```

The final model is trained using the TF-IDF representation.

The notebook then loads:

```python
tickets_test.csv
```

and generates predictions for all **1,000 test tickets**.

The prediction process is:

```text
tickets_test.csv
       ↓
Ticket text
       ↓
TF-IDF vectorization
       ↓
Multinomial Naive Bayes
       ↓
Predicted category
```

The notebook saves the resulting file as:

```text
test_predictions.csv
```

### Example predicted records

| Ticket ID | Predicted Category |
|---|---|
| `T100009` | `wrong_order` |
| `T100013` | `missing_item` |
| `T100018` | `late_delivery` |
| `T100019` | `payment_issue` |
| `T100021` | `missing_item` |

The original task specifies an output named `predictions.csv` containing exactly:

```text
ticket_id,category
```

The supplied notebook currently writes `test_predictions.csv` containing the original test columns plus `category`. This should be adjusted if the submission must follow the exact requested format.

---

# 💼 Business Recommendation

The model is best treated as a **routing assistant**, not a complete replacement for support agents.

### Recommended workflow

```text
Customer Ticket
      ↓
Text Preprocessing
      ↓
TF-IDF
      ↓
MultinomialNB
      ↓
Prediction + Confidence
      ↓
 ┌─────────────────────────────┐
 │ High confidence             │
 │ → Automatic routing         │
 └─────────────────────────────┘
              │
              │ Low confidence /
              │ ambiguous issue
              ↓
       Human review
```

### Suitable for automated routing

Tickets with clear language such as:

- Clearly late orders
- Clearly missing items
- Clearly wrong orders
- Clearly identifiable payment problems
- Clearly identifiable account-access problems

### Keep a human in the loop when

- The ticket contains multiple problems.
- The message is very short or ambiguous.
- The model confidence is low.
- The customer describes an unusual situation.
- The predicted category could have a significant operational consequence.
- The ticket appears to contain conflicting information.

A practical deployment could use a confidence threshold and route low-confidence cases to human agents.

---

# 🔧 Recommended Improvements

To fully satisfy the original project specification, the following additions should be made.

## 1. Add a keyword baseline

Create a simple rule-based classifier using manually selected keywords for each category.

For example:

```text
late_delivery   → late, delayed, waiting, ETA
missing_item    → missing, forgot, absent, didn't receive
payment_issue   → card, charged, refund, payment
wrong_order     → wrong, incorrect, different order
account_access  → login, password, account, locked
dasher_feedback → dasher, driver, delivery person
```

Evaluate this baseline using **macro F1** and compare it against the TF-IDF model.

---

## 2. Use cross-validation

Instead of relying only on one train-test split, use stratified cross-validation.

For example:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

Report:

- Mean macro F1
- Standard deviation
- Mean accuracy

This gives a more reliable estimate of model performance.

---

## 3. Perform the requested 20-error analysis

Collect 20 incorrectly classified validation tickets and categorize the mistakes.

Possible groups include:

- Multiple problems in one ticket
- Similar wording between categories
- Spelling errors
- Very short messages
- Ambiguous customer language
- Potentially incorrect human labels

This analysis can identify whether future improvements should focus on preprocessing, training data quality, feature engineering, or label definitions.

---

## 4. Improve output format

Create the requested final submission:

```python
predictions = test_df[['ticket_id', 'category']]
predictions.to_csv('predictions.csv', index=False)
```

This produces exactly:

```text
ticket_id,category
T100009,wrong_order
T100013,missing_item
...
```

---

# 🛠️ Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **NLTK**
- **XGBoost**
- **Matplotlib**
- **Seaborn**
- **TF-IDF**
- **Multinomial Naive Bayes**
- **Complement Naive Bayes**
- **Decision Tree**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**

---

# 📁 Project Structure

A recommended repository structure is:

```text
DoorDash-Support-Ticket-Triage/
│
├── DoorDash_Support_Ticket_Traige.ipynb
├── tickets_train.csv
├── tickets_test.csv
├── predictions.csv
├── README.md
└── requirements.txt
```

---

# 📈 Key Takeaways

- The dataset contains **3,000 labelled support tickets** across six categories.
- There are **no missing values** in the training data.
- Customer messages contain realistic noise such as typos, capitalization differences, and informal wording.
- TF-IDF with **unigrams and bigrams** was used for text representation.
- Six traditional ML classifiers were evaluated.
- **Multinomial Naive Bayes achieved 90% accuracy and 0.89 macro F1** on the 600-ticket hold-out set.
- `late_delivery` was the strongest-performing category by recall.
- `wrong_order` had the lowest recall among the six categories.
- The model generated predictions for all **1,000 held-out test tickets**.
- The notebook demonstrates that traditional NLP/ML can provide a practical baseline for automated support-ticket routing without large language models or paid APIs.

---

## ⚠️ Project Status

**Core modeling:** ✅ Completed  
**EDA:** ✅ Completed  
**Text preprocessing:** ✅ Completed  
**TF-IDF feature engineering:** ✅ Completed  
**Multiple classifiers:** ✅ Completed  
**Confusion matrix:** ✅ Completed  
**Test-set predictions:** ✅ Completed  
**Keyword baseline:** ⚠️ Not implemented in supplied notebook  
**Cross-validation:** ⚠️ Not implemented in supplied notebook  
**20-error analysis:** ⚠️ Not implemented in supplied notebook  
**Exact `predictions.csv` format:** ⚠️ Needs adjustment from current `test_predictions.csv`

