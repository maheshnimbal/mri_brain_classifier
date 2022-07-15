import csv
import numpy as np

# read age data from csv
# age = np.loadtxt()
with open('../input/ell319-data/ELL319_kaggle/age_final.csv', 'r') as f:
    csvreader = csv.reader(f)
    age = next(csvreader)

# sort the ages into bins of developmental categories (from paper: 9-18, 19-32, 33-51, 52-83)
for i in range(len(age)):
    n = float(age[i])
    if n < 19:
        age[i] = 1
    elif n < 33:
        age[i] = 2
    elif n < 52:
        age[i] = 3
    else:
        age[i] = 4
age = np.array(age)
# now age contains the correct labels for training the svm
# print(age)

# Read data
# how to combine all the different csv files as one input data file X
# x = file of features of different subjects (each value is a timeseries and not a single float)

# assuming x is only the ApEn data of subjects
x = np.loadtxt('../input/ell319-apen/apEn.csv', delimiter = ',')

# split the data (80-20)
from sklearn.model_selection import train_test_split
train_mri, test_mri, train_lbl, test_lbl = train_test_split(x, age, test_size = 0.2, random_state=0)

print(train_mri.shape)
print(test_mri.shape)
print(train_lbl.shape)
print(test_lbl.shape)

# standardise data
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)

# pre-process the data using PRINCIPAL COMPONENT ANALYSIS
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
from sklearn.decomposition import PCA

pca = PCA(0.9)
pca.fit(train_mri)

train_mri = pca.transform(train_mri)
test_mri = pca.transform(test_mri)

# train the SVM using pre-processed data
# https://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf', gamma=20, C=10)
svclassifier.fit(train_mri, train_lbl)

test_pred = svclassifier.predict(test_mri)

# Performance Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print('Model Accuracy:', accuracy_score(test_lbl, test_pred))
print()
print('Classification Report')
# print(confusion_matrix(test_lbl, test_pred))
print(classification_report(test_lbl, test_pred))
