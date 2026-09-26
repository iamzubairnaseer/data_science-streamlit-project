# Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to **churn (Yes/No)** based on demographic, subscription, billing, and customer-service information.

The project compares **Logistic Regression** and **Random Forest** classifiers and deploys the selected model through an interactive **Streamlit web application**.

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. By identifying potential churners in advance, organizations can take proactive retention measures.

In this project, customer information such as contract type, tenure, monthly charges, support calls, payment method, and internet service is used to predict customer churn.

### Objectives

* Explore and understand customer churn data.
* Perform data preprocessing and exploratory data analysis (EDA).
* Handle missing values.
* Encode categorical variables.
* Scale numerical features.
* Train and compare multiple classification models.
* Evaluate models using Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.
* Select the better-performing model.
* Save the trained machine learning pipeline.
* Deploy the model using Streamlit.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── churn_pipeline.pkl
├── requirements.txt
├── README.md
│
└── notebooks/
    └── customer_churn_prediction.ipynb
```

> The exact structure may vary depending on how the project is organized.

---

## 📊 Dataset

The dataset contains customer-level information related to demographics, service usage, contracts, billing, and customer support.

### Features

| Feature                   | Description                                             |
| ------------------------- | ------------------------------------------------------- |
| `customer_id`             | Unique customer identifier                              |
| `age`                     | Customer age                                            |
| `gender`                  | Customer gender                                         |
| `region`                  | Customer region                                         |
| `tenure_months`           | Number of months the customer has been with the company |
| `monthly_charges`         | Customer's monthly charges                              |
| `total_charges`           | Total charges incurred by the customer                  |
| `contract_type`           | Type of customer contract                               |
| `internet_service`        | Type of internet service                                |
| `tech_support`            | Whether the customer has technical support              |
| `online_security`         | Whether the customer has online security                |
| `paperless_billing`       | Whether paperless billing is enabled                    |
| `payment_method`          | Customer payment method                                 |
| `num_support_calls`       | Number of customer support calls                        |
| `late_payments_last_year` | Number of late payments during the previous year        |
| `avg_monthly_usage_gb`    | Average monthly internet usage                          |
| `churn`                   | Target variable indicating whether the customer churned |

The `customer_id` feature is excluded from model training because it is an identifier rather than a meaningful predictive feature.

---

## 🔍 Exploratory Data Analysis

The project includes exploratory analysis to understand patterns and relationships within the dataset.

Key visualizations include:

* Churn class distribution
* Churn by contract type
* Churn by internet service
* Monthly charges distribution
* Tenure distribution
* Monthly charges by churn status

EDA helps identify potential relationships between customer characteristics and churn behavior before model training.

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Missing Values

Numerical missing values were handled using **median imputation**, while categorical missing values were handled using the **most frequent value (mode)**.

### Categorical Encoding

Categorical features were converted into numerical representations using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Using `handle_unknown='ignore'` allows the model to process previously unseen categorical values during prediction.

### Numerical Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

### Pipeline

Preprocessing and model training were combined into a single scikit-learn `Pipeline`.

This ensures that the same preprocessing steps used during training are automatically applied when making predictions.

---

## 🤖 Machine Learning Models

Two classification algorithms were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier

### Logistic Regression

Logistic Regression was used as a baseline classification model and performed well on the churn prediction task.

### Random Forest

Random Forest was also trained to capture potentially non-linear relationships between customer characteristics and churn.

---

## 📈 Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

### Results

| Model               | Accuracy | Precision | Recall | F1 Score |
| ------------------- | -------: | --------: | -----: | -------: |
| Logistic Regression |   75.83% |    76.71% | 82.35% |   79.43% |
| Random Forest       |   71.67% |    74.29% | 76.47% |   75.36% |

Based on the test-set results, **Logistic Regression was selected as the final model** because it achieved higher Accuracy, Precision, Recall, and F1 Score than the Random Forest model.

It also produced fewer false negatives:

| Model               | False Negatives |
| ------------------- | --------------: |
| Logistic Regression |              24 |
| Random Forest       |              32 |

Reducing false negatives is particularly relevant for churn prediction because a false negative represents a customer who is predicted not to churn but actually churns.

---

## 💾 Model Saving

The complete trained pipeline was saved using `joblib`:

```python
import joblib

joblib.dump(logistic_model, 'churn_pipeline.pkl')
```

The saved pipeline contains both:

* Data preprocessing
* Trained Logistic Regression model

Therefore, the Streamlit application does not need to recreate or retrain the model.

---

## 🌐 Streamlit Application

The trained model is deployed through a Streamlit web application.

The application allows users to enter customer information through interactive widgets such as:

* Number inputs
* Select boxes
* Sliders

The application then sends the entered customer information to the trained pipeline and displays the predicted churn status.

### Prediction

The application provides:

```text
Churn Prediction: Yes
```

or

```text
Churn Prediction: No
```

Where supported, the application also displays the predicted probability of churn.

---

## 🖥️ Running the Application Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ⚠️ Model Compatibility

The `churn_pipeline.pkl` file was created using scikit-learn and should ideally be loaded with the **same scikit-learn version used during model training**.

The project's `requirements.txt` should therefore pin the scikit-learn version used to create the saved model.

For example:

```text
scikit-learn==1.x.x
```

This helps prevent compatibility issues when loading the serialized pipeline.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Joblib**
* **Streamlit**
* **Jupyter Notebook**

---

## 🔄 Machine Learning Workflow

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Missing Value Handling
     ↓
Exploratory Data Analysis
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Feature Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Logistic Regression Selected
     ↓
Save Pipeline
     ↓
Streamlit Deployment
     ↓
Customer Churn Prediction
```

---

## 🚀 Future Improvements

Possible improvements to the project include:

* Hyperparameter tuning
* Cross-validation
* Feature engineering
* Class imbalance analysis
* Threshold optimization
* Explainable AI using SHAP
* Model monitoring
* Automated retraining
* Cloud deployment
* Integration with a customer database
* Adding customer retention recommendations based on prediction results

---

## 👨‍💻 Author

**Zubair Naseer**

Data Engineer | AI & Automation Specialist

---

## 📄 License

This project is intended for educational and portfolio purposes.
