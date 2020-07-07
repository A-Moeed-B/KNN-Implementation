# Apply KNN algorithm,  
# Split orginal data into 85% training and 15 % test data 
# Compute confusion matrix and accuracy metrics
# importing necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import zero_one_loss
from sklearn.metrics import classification_report
# reading the data set file
df = pd.read_csv('tmdb_5000_movies.csv')
# the following loop will convert the object(string) data type columns into category data type
for col in df.columns.values:
    if df[col].dtypes == 'object':
        df[col] = df[col].astype('category')
# the following loop will convert the category data type into int for processing.
# this is because I couldn't convert object directly into integer and needed to convert into category first
for col in df.columns.values:
    if str(df.dtypes[col]) == 'category':
        df[col] = df[col].cat.codes
# the following two lines are used to clean the data
# it will remove Null NaN infinite or other type of non-processable values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(-1, inplace=True)
# this will select the columns that will predict the outcome of y (revenue)
# the budget and popularity columns which are at 0 and 8 index respectively
x = df.iloc[:, [0, 8]]
# this will select the column to be predicted (revenue)
y = df.iloc[:, 12]
# this value of k=69 because sqroot(4803) = 69.3....
knn = KNeighborsClassifier(n_neighbors=69)
# this will split the data into test and training data with  ratio of 0.85:0.15 i.e. 85% data split into training and
# rest for test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=1)
# this will apply the KNN algorithm on the data set
knn.fit(x_train, y_train)
y_test_pred = knn.predict(x_test)
prediction_final = pd.DataFrame(data=[y_test_pred, y_test.values])
prediction_final.transpose()
print('Final Prediction: ')
print(prediction_final)
# this will create the confusion matrix
print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_test_pred))
print('Confusion Matrix in table form: ')
print(pd.crosstab(y_test, y_test_pred, rownames=['True'], colnames=['Predicted'], margins=True))
# this will print out the accuracy metrics
print("Accuracy Metrics: ")
# maximum accuracy out of 1.0 from the 15% test data
print('Maximum Accuracy achieved by test data (out of 1.0): ')
print(accuracy_score(y_test, y_test_pred))
# this will show the number of accurately identified samples from the 15% test data.
print('Number of accurately identified samples: ')
print(accuracy_score(y_test, y_test_pred, normalize=False))
# this will show the error rate
print('Error Rate: ')
print(zero_one_loss(y_test, y_test_pred))
