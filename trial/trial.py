import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Load the dataset
heart_data = pd.read_csv('heart_disease_data.csv')

# Split data into features and target variable
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Predict on the input data
input_data = (63,1,2,130,231,0,1,146,0,1.8,1,3,3)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

# Print the outcome based on prediction
if (prediction[0] == 0):
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Disease')
