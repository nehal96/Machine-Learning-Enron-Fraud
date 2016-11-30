import sys
import pickle
from pprint import pprint
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from select_k_best import Select_K_Best

def CreatePoiEmailRatio(data_dict, features_list):
    """
    Adds a new feature to the feature list: POI Email Ratio.
    """
    features = ['from_messages', 'to_messages', 'from_poi_to_this_person',
                'from_this_person_to_poi']

    for key in data_dict:
        employee = data_dict[key]
        is_valid = True
        for feature in features:
            if employee[feature] == 'NaN':
                is_valid = False
        if is_valid:
            total_from = employee['from_poi_to_this_person'] + employee['from_messages']
            total_to = employee['from_this_person_to_poi'] + employee['to_messages']
            to_poi_ratio = float(employee['from_this_person_to_poi']) / total_to
            from_poi_ratio = float(employee['from_poi_to_this_person']) / total_from
            employee['poi_email_ratio'] = to_poi_ratio + from_poi_ratio
        else:
            employee['poi_email_ratio'] = 'NaN'

    features_list.append('poi_email_ratio')

def CreateExercisedStockRatio(data_dict, features_list):
    """
    Adds a new feature to the feature list: Exercised Stock Ratio
    """

    features = ['exercised_stock_options', 'total_stock_value']

    for key in data_dict:
        employee = data_dict[key]
        is_valid = True
        for feature in features:
            if employee[feature] == 'NaN':
                is_valid = False
        if is_valid:
            employee['exercised_stock_ratio'] = float(employee['exercised_stock_options']) / employee['total_stock_value']
        else:
            employee['exercised_stock_ratio'] = 'NaN'

    features_list.append('exercised_stock_ratio')

