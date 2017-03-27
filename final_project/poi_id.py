#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                'deferred_income', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                'restricted_stock', 'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                'from_this_person_to_poi', 'shared_receipt_with_poi'] # You will need to use more features

# There are a total of 17 features that were picked, while removing the total as they would create a problem of overfitting
print("number of features to start with: {0}".format(len(features_list) - 1))

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# There are 146 data points for this dataset
print("\nSize of the dataset: " + str(len(data_dict)))

# The number of POI/non-POI are counted here based on the the value of the POI feature.
npoi = 0
for p in data_dict.values():
    if p['poi']:
        npoi += 1

# We have 18  POIs and 128 non-POIs
print("number of pois: {0}".format(npoi))
print("number of non-pois: {0}".format(len(data_dict) - npoi))

# In the following lines, the features and number of NaNs are listed.
print("print out the number of missing values in each feature: ")
NaNInFeatures = [0 for i in range(len(features_list))]
for i, person in enumerate(data_dict.values()):
    for j, feature in enumerate(features_list):
        if person[feature] == 'NaN':
            NaNInFeatures[j] += 1

totalNaN = 0

for i, feature in enumerate(features_list):
    totalNaN += NaNInFeatures[i]
    print(feature, NaNInFeatures[i])

print("Total NaN: " + str(totalNaN))

print("\nThe list of people in the database: ")
for person in sorted(data_dict.keys()):
    print(person)

# Remove the user 'TOTAL'
data_dict.pop('TOTAL')
print(len(features_list))
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# I have an assumption that people with absurdly high to/from messages from POI maybe POI themselves. So, it makes sense to look at the 
# to/from percentages.
# Adding these features increases the precision by 0.004 with no change in recall in case of Nearest Centroid
# For Ada boost the precision decreases by 0.07 and the recall increses by 0.01
# So, it makes sense to add these features only in case of Nearest Centroid.
print("New features created are: to_poi_messages_percent and from_poi_messages_percent")

for person, feature in data_dict.items():
    if float(feature['to_messages']) > 0:
        my_dataset[person]["to_poi_messages_percent"] = float(feature['from_this_person_to_poi'])/float(feature['to_messages']) 
    else:
        my_dataset[person]["to_poi_messages_percent"] = 0
    if float(feature['from_messages']) > 0:
        my_dataset[person]["from_poi_messages_percent"] = float(feature['from_poi_to_this_person'])/float(feature['from_messages'])
    else:
        my_dataset[person]["from_poi_messages_percent"] = 0

features_list += ["to_poi_messages_percent", "from_poi_messages_percent"]
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# print(features)
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler

min_max_scaler = MinMaxScaler()

nc = NearestCentroid()
adc = AdaBoostClassifier()

nc_report = {'accuracy': list(), 'precision': list(), 'recall': list()}
adc_report = {'accuracy': list(), 'precision': list(), 'recall': list()}

# this function is derived from tester.py
def create_report(clf, features, labels):
    cv = StratifiedShuffleSplit(labels, 100, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = round(1.0*(true_positives + true_negatives)/total_predictions,2)
    try:
        precision = round(1.0*true_positives/(true_positives+false_positives),2)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = round(1.0*true_positives/(true_positives+false_negatives),2)
    except ZeroDivisionError:
        recall = 0.0
    return accuracy, precision, recall

print("Select features:")
t0 = time()
for i in range(17):
    selector = SelectKBest(f_classif, k = i+1)
    selector.fit(features, labels)
    reduced_features = selector.fit_transform(features, labels)

    # gather data for the report
    a, p, r = create_report(adc, reduced_features, labels)
    adc_report['accuracy'].append(a)
    adc_report['precision'].append(p)
    adc_report['recall'].append(r)

    a, p, r = create_report(nc, min_max_scaler.fit_transform(reduced_features), labels)
    nc_report['accuracy'].append(a)
    nc_report['precision'].append(p)
    nc_report['recall'].append(r)
print "report time: {0}".format(round(time()-t0, 3))

print(adc_report)
print(nc_report)

# ploting the accuracy, precision and recalls
adc_plot = pd.DataFrame(adc_report)
adc_plot.plot(title='AdaBoost Report', kind='bar')
plt.savefig('adc_report.png')

nc_plot = pd.DataFrame(nc_report)
nc_plot.plot(title='Nearest Centroid Report', kind='bar')
plt.savefig('nc_report.png')

# Form the above plots we can see that the most balanced values of precision and recall are for:
# AdaBoost: k = 11
# NearestCentroid: k = 11

selector = SelectKBest(f_classif, k = 11)
selector.fit(features, labels)
reduced_features = selector.fit_transform(features, labels)

print("Scores of the features: ")
for score in sorted(selector.scores_):
    print(score)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

features_train, features_test, labels_train, labels_test = \
    train_test_split(reduced_features, labels, test_size=0.3, random_state=42)

# Ada Boost
t0 = time()
# Parameters to be tuned for Ada Boost
parameters = {'n_estimators': [50,100,200], 'learning_rate': [0.4,0.6,1]}

clf = GridSearchCV(adc, parameters, scoring = 'recall')
clf.fit(features_train, labels_train)
print("Best parameters are:")
print(clf.best_params_)
print "training time: {0}".format(round(time()-t0, 3))
clf = clf.best_estimator_

# Nearest Centroid
t0 = time()
# Parameters to be tuned for NearestCentroid
parameters = {'metric':('cosine', 'euclidean', 'l1', 'l2'), 
'shrink_threshold':[0.9, 0.7, 0.5]}

clf = GridSearchCV(nc, parameters, scoring = 'recall')
clf.fit(features_train, labels_train)
print("Best parameters are:")
print(clf.best_params_)
print "training time: {0}".format(round(time()-t0, 3))
clf = clf.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)