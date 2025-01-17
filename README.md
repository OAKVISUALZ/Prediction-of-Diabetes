# Diabetes Prediction with Machine Learning

## Objective
This project aims to predict whether an individual has diabetes based on various diagnostic measures, including age, glucose levels, blood pressure, and other medical attributes. The project uses a supervised machine learning approach to build a predictive model.

### Skills Developed
- Techniques for data preprocessing and cleaning.
- Feature engineering and selection.
- Model training and hyperparameter tuning.
- Evaluation of classification models.


### Tools Utilized
- Python (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn)
- Jupyter Notebook for interactive development
- Git for version control, with repositories on GitHub

### Project Execution Steps
1. Clone the repository.
2. Install the necessary Python libraries.
3. Run the Jupyter Notebook to explore the dataset and train models.
4. Review the model's results and predictions.

```python
# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load Dataset
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(data_url, header=None, names=columns)

# Data Overview
print("Dataset Shape:", df.shape)
print("First 5 Rows:\n", df.head())

# Data Preprocessing
# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Splitting features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Standardizing the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
plt.barh(range(X.shape[1]), feature_importances, align='center')
plt.yticks(range(X.shape[1]), X.columns)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Diabetes Prediction")
plt.show()

# Save the Model (Optional)
import joblib
joblib.dump(model, "diabetes_predictor.pkl")

# Load and Test Saved Model
loaded_model = joblib.load("diabetes_predictor.pkl")
sample_data = np.array([X.iloc[0]])
prediction = loaded_model.predict(sample_data)
print("Prediction for Sample Data (0=No Diabetes, 1=Diabetes):", prediction)
```

### Results

![image](https://github.com/user-attachments/assets/f88f8784-37db-4b4b-9cb5-ea5373f8e1f4)


---

This project can be further enhanced by experimenting with different machine learning models or incorporating deep learning techniques for improved prediction accuracy.

---

