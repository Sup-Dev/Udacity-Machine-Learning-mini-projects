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

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print("\nSize of the dataset: " + str(len(data_dict)))

# print("\nThe list of people in the database: ")
# for person in sorted(data_dict.keys()):
#     print(person)

# Remove the user 'TOTAL'
data_dict.pop('TOTAL')
print(len(features_list))
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

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

nc = NearestCentroid()
adc = AdaBoostClassifier()

print("Select 7 features by importance:")
selector = SelectKBest(f_classif, k = 7)
selector.fit(features, labels)
reduced_features = selector.fit_transform(features, labels)


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

# Nearest Centroid
t0 = time()
parameters = {'metric':('cosine', 'euclidean', 'l1', 'l2'), 
'shrink_threshold':[0.9, 0.7, 0.5]}

clf = GridSearchCV(nc, parameters, scoring = 'recall')
clf.fit(features_train, labels_train)
print("Best parameters are:")
print(clf.best_params_)
print "training time: {0}".format(round(time()-t0, 3))
clf = clf.best_estimator_

# Ada Boost
# t0 = time()
# parameters = {'n_estimators': [50,100,200], 'learning_rate': [0.4,0.6,1]}

# clf = GridSearchCV(adc, parameters, scoring = 'recall')
# clf.fit(features_train, labels_train)
# print("Best parameters are:")
# print(clf.best_params_)
# print "training time: {0}".format(round(time()-t0, 3))
# clf = clf.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)