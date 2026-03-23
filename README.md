#  Titanic Survival Prediction | Machine Learning Project

##  Overview
This project applies machine learning techniques to predict passenger survival on the Titanic dataset. It demonstrates a complete ML pipeline including data analysis, preprocessing, model building, and evaluation.

---

## 📊 Dataset
- Source: Titanic Dataset (Kaggle)
- The dataset contains information about passengers such as:
  - Age
  - Sex
  - Passenger Class (Pclass)
  - Fare
  - Embarked
  - Survival (Target variable)

##  Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Analyzed dataset using `.head()`, `.info()`, `.describe()`
- Visualized survival distribution
- Plotted:
  - Survival count
  - Age distribution (survivors vs non-survivors)
  - Fare distribution
  - Correlation heatmap

---

### 2. Data Preprocessing
- Handled missing values
- Converted categorical features into numerical format
- Selected relevant features for modeling

---

### 3. Machine Learning Models

The following classification models were implemented and compared:

- Logistic Regression (baseline model)  
- Decision Tree Classifier  
- Random Forest Classifier (ensemble method)  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)  
- Naive Bayes  

---

### 4. Model Evaluation
- Models were evaluated based on accuracy
- Compared performance to identify the best model

---

## Results
- Achieved good prediction accuracy
- Observed key insights:
  - Female passengers had higher survival rates
  - Passenger class significantly impacted survival
  - Fare and age also influenced outcomes

---

##  Tools & Technologies
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

##  Key Learnings
- Data cleaning and preprocessing techniques  
- Exploratory Data Analysis (EDA)  
- Visualization for insights  
- Machine learning model building and evaluation  

---

##  Future Improvements
- Feature engineering for better performance  
- Hyperparameter tuning  
- Implementation of advanced models (XGBoost, Gradient Boosting)  
