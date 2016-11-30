#!/usr/bin/python

import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from visualize import DrawPlot

sys.path.append("../final_project/")
from tester import dump_classifier_and_data, test_classifier


### Task 1: Select the features I'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                      'salary', 'deferred_income', 'long_term_incentive',
                      'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
                      'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi']

### See select_k_best.py to see how these features were chosen

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)

### Create new feature
### See feature_creation.py to see the code for the creation of the two features

sys.path.append("../tools/")
from feature_creation import CreatePoiEmailRatio, CreateExercisedStockRatio

CreatePoiEmailRatio(data_dict, features_list)
CreateExercisedStockRatio(data_dict, features_list)

### After running Select_K_Best on the new features, poi_email_ratio comes 5th,
### but exercised_stock_ratio comes last. Let's remove it from the feature list.

features_list.remove('exercised_stock_ratio')

### Store to my_dataset for easy export.

my_dataset = data_dict

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Scale features

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Task 4: Try a varity of classifiers

from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

###### Uncomment to run PCA and/or GridSearchCV using Pipelines ######

# n_components = 8
# estimators = [('reduce_dim', PCA(n_components=n_components)), ('clf', DecisionTreeClassifier(min_samples_split=3))]
# clf = Pipeline(estimators)

### OR ###
# estimators = [('reduce_dim', PCA()), ('clf', DecisionTreeClassifier())]
# parameters = {'reduce_dim__n_components': [4, 6, 8, 10], 'clf__min_samples_split': [3, 4, 5, 6, 7, 8]}
# clf = GridSearchCV(pipe, parameters, scoring='f1')

######################################################################


###### Decision Tree Classifier ######

clf = DecisionTreeClassifier(min_samples_split=7)

######################################


###### AdaBoost Classifier ######

# clf = AdaBoostClassifier(n_estimators=100)

#################################


###### K-Nearest Neighbors Classifier ######

# clf = KNeighborsClassifier(n_neighbors=6)

############################################



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

clf.fit(features_train, labels_train)

### If GridSearchCV, uncomment to run attributes

# print clf.cv_results_
# print clf.best_estimator_
# print clf.best_score_
# print clf.best_params_


pred = clf.predict(features_test)

### Uncomment to run local testing. Used to check if classifier was working fine,
### proper testing was done using test_classifier.

# print accuracy_score(labels_test, pred)
# print precision_score(labels_test, pred)
# print recall_score(labels_test, pred)
# print f1_score(labels_test, pred)
# print classification_report(labels_test, pred)
# print confusion_matrix(labels_test, pred)


### Trying out other feature lists, particularly for the DecisionTreeClassifier

DT_features_list = ['poi','exercised_stock_options', 'total_stock_value', 'bonus',
                 'salary', 'deferred_income', 'long_term_incentive',
                 'poi_email_ratio']

DT_features_list_email = ['poi', 'exercised_stock_options', 'poi_email_ratio']


### Proper testing of classifier

test_classifier(clf, my_dataset, DT_features_list_email, folds=1000)


### Uncomment to see Plot of final features

#DrawPlot(my_dataset, 'exercised_stock_options', 'poi_email_ratio')


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, DT_features_list_email)
