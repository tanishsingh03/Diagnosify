import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load the dataset
data_path = '.data/heart_disease_data.csv'
heart_data = pd.read_csv(data_path)

# Split the data into features (X) and target (Y)
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=500)
model.fit(X_train, Y_train)

# Save the trained model to a file using pickle
model_filename = 'trained_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f'Model trained and saved as {model_filename}')
