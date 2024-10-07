# Iris Flower Classification using Logistic Regression

This project uses the Iris flower dataset to predict flower species using the Logistic Regression model. The dataset includes three species: Setosa, Versicolour, and Virginica.

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- joblib

## Dataset

The dataset used is the `Iris` dataset from `sklearn.datasets`, which contains measurements of iris flowers.

## Steps

1. **Load Dataset**: Load the iris dataset using `load_iris()` from `sklearn.datasets`.
2. **Data Preparation**: Prepare the dataset by creating a DataFrame.
3. **Model Training**: Split the data into training and testing sets, and train the Logistic Regression model.
4. **Evaluation**: Measure the prediction score and plot the confusion matrix.

## Code

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt
import joblib

# Load dataset
data_set = load_iris()
target_array = data_set.target_names

# Create DataFrame
df = pd.DataFrame(data_set.data, columns=data_set.feature_names)
target = pd.DataFrame(data_set.target, columns=['target'])
df = pd.concat([df, target], axis='columns')

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df.drop(['target'], axis='columns'), df['target'], test_size=0.2, random_state=10)

# Train Logistic Regression model
model = linear_model.LogisticRegression()
model.fit(x_train, y_train)

# Measure prediction score
score = model.score(x_test, y_test)
print(f'Prediction Score: {score}')

# Plot confusion matrix
y_pred = model.predict(x_test)
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='Greens', annot=True, xticklabels=target_array, yticklabels=target_array)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()
