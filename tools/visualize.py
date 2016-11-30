import sys
import pickle
import matplotlib.pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

def DrawPlot(data_dict, feature_x, feature_y):
    """
    Draws a plot of the two features selected and colors the POIs.
    feature_list must be of the form ["poi", feature_y, feature_x] (i.e. label
    comes first, then y variable, then x variable (see FeatureFormat write-up)
    """

    feature_list = [feature_x, feature_y, 'poi']
    data_array = featureFormat(data_dict, feature_list)
    poi_color = "r"
    non_poi_color = "g"

    for point in data_array:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            plt.scatter(x, y, color="r")
        else:
            plt.scatter(x, y, color="g")
    plt.scatter(x, y, color="r", label="poi")
    plt.scatter(x, y, color="g", label="non-poi")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend()
    plt.show()
