import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read data file and Define column name of the data
def read_data(file_path):
    column_names = ['Label','number','Peak Height for 567 Hz','Peak Height for 1134 Hz', 'Peak Height for 1702 Hz', 'Peak Height for 2269 Hz', 
                    'Peak Height for 2836 Hz','Peak Height for 3403 Hz','Peak Height for 3970 Hz','Peak Height for 4537 Hz']
    df = pd.read_csv(file_path,header = None, names = column_names)
    return df

dataset = read_data(r'/Users/seulki_han/Desktop/Ph.D/Research/Gerber Tech/Model/data.csv')

# Set input & output dimensions
x = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 0].values

# Split train data (80%) and test data (20%)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# Normalize features (values between 0 and 1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# KNN model (k value = 5)
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Print confusion matrix for testing data
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

