from pandas import read_csv, DataFrame
from matplotlib.pyplot import show, plot, scatter, subplot 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.neighbors import KNeighborsClassifier


#load the data
data = read_csv("E:\ML\Intro to Deep Learning\Labs\Codes\Applied ML\Pima Indians Diabetes\diabetes.csv")

# Replace zeros with NaN
zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[zero_columns] = data[zero_columns].replace(0, np.nan)


# Fill NaN using KNN imputer
imputer = KNNImputer(n_neighbors=5)
data[zero_columns] = imputer.fit_transform(data[zero_columns])
"""
# Fill NaN using mean imputer for classifier problems
imputer = SimpleImputer(strategy='median')
data[zero_columns] = imputer.fit_transform(data[zero_columns])
"""

# Split features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

#Scale the data using standard scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#split the features to train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Use KNN model
#Initialize the model
knn_model = KNeighborsClassifier()

#train the model
knn_model.fit(X_train, y_train)

#compute the accurcy
knn_accuracy = knn_model.score(X_test, y_test)
#print(f"The Accuracy befor hyperparameters tunning = {knn_accuracy:.2f}")

#Hyperparameter Tuning with GridSearchCV
param_grid = {'n_neighbors': range(1, 20),'metric': ['euclidean', 'manhattan']}

#grid search with 5-fold cross valdation
knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)

#calculate gridsearch accuracy
#print(f"Accuracy after Grid Search = {knn_grid.best_score_:.2f}")

# Predict on test set
y_pred = knn_grid.predict(X_test)

# Classification report
#print(classification_report(y_test, y_pred))