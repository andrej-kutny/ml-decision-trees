import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

y_true = [1,1,1,1,1] # Pretend labels
y_pred = [1,1,2,2,1] # Pretend prediction

data_wine = pd.read_csv("wine.csv").to_numpy()

data_wine_feature_names, data_wine_target_name = pd.read_csv("wine.csv", nrows=0).columns.to_numpy()[0:-1], pd.read_csv("wine.csv", nrows=0).columns.to_numpy()[-1]

# TODO: Set up the data and split it into train and test-sets.
data_wine_feats_train, data_wine_feats_test, data_wine_labels_train, data_wine_labels_test = train_test_split(data_wine[1:, 0:-1], data_wine[1:, -1], random_state=0)

# TODO: Train and test your implemented tree model.
# NOTE: Use the same train/test split for your tree model and the scikit learn model'

# FUNCTIONS
def most_appropriate_label(data):
    label_values = np.unique(data)
    label_app_perc = np.zeros(len(label_values))
    for label in data:
        pos_idx = np.where(label_values == label)[0][0]
        label_app_perc[pos_idx] += 1
    label_app_perc /= len(data)
    return int(label_values[np.argmax(label_app_perc)]), np.max(label_app_perc)

def check_if_homogeneous(data, threshold = 0.8):
    if most_appropriate_label(data)[1] >= threshold:
        return True
    return False

def extract_feat(feat_idx, feature_set):
    feature = np.array([])
    for features in feature_set:
        feature = np.append(feature, float(features[feat_idx]))
    return feature

def extract_subsets(data, feature_set, feature, threshold):
    index_order = np.argsort(feature)
    subset_l_pos, subset_r_pos = index_order[threshold:], index_order[:threshold]
    subset_l_feats, subset_r_feats, subset_l_labels, subset_r_labels = np.asarray([]), np.asarray([]), np.asarray([]), np.asarray([])
    for pos in subset_l_pos:
        subset_l_feats = np.append(subset_l_feats, feature_set[pos], axis=0)
        subset_l_labels = np.append(subset_l_labels, int(data[pos]))
    for pos in subset_r_pos:
        subset_r_feats = np.append(subset_r_feats, feature_set[pos], axis=0)
        subset_r_labels = np.append(subset_r_labels, int(data[pos]))
    return subset_l_feats.reshape(len(subset_l_pos), feature_set.shape[1]).astype(float), subset_r_feats.reshape(len(subset_r_pos), feature_set.shape[1]).astype(float), subset_l_labels, subset_r_labels

def calculate_probability(value, data):
    pb_value = np.where(data == value)[0].shape[0] / data.shape[0]
    return pb_value

def entropy(data):
    ent = 0
    for value in np.unique(data):
        ent += calculate_probability(value, data) * np.log(calculate_probability(value, data))
    ent = -ent
    return ent

def gini_index(data):
    gini = 1
    for value in np.unique(data):
        gini -= calculate_probability(value, data)**2
    return gini

def best_split_feature(data, feature_set, gini): 
    split_thr = 1
    min_imp = 1
    for i in range(feature_set.shape[1]):
        feature = extract_feat(i, feature_set)
        for threshold in range(1, len(feature)):
            _, _, subset_l_labels, subset_r_labels = extract_subsets(data, feature_set, feature, threshold)
            if gini == False:
                imp = (subset_l_labels.shape[0] * entropy(subset_l_labels) + subset_r_labels.shape[0] * entropy(subset_r_labels)) / data.shape[0]
            else:
                imp = (subset_l_labels.shape[0] * gini_index(subset_l_labels) + subset_r_labels.shape[0] * gini_index(subset_r_labels)) / data.shape[0]
            if imp < min_imp:
                min_imp = imp
                best_feat_idx = i
                split_thr = threshold
    return best_feat_idx, split_thr

def split_data(data, feature_set, gini_instead_of_entropy):
    best_feat_idx, split_thr = best_split_feature(data, feature_set, gini_instead_of_entropy)
    best_feat = extract_feat(best_feat_idx, feature_set)
    subset_l_feats, subset_r_feats, subset_l_labels, subset_r_labels = extract_subsets(data, feature_set, best_feat, split_thr)
    return best_feat_idx, subset_l_feats, subset_r_feats, subset_l_labels, subset_r_labels

def grow_tree(data, feature_set, feature_names, target_name, gini_instead_of_entropy = False):
    if check_if_homogeneous(data):
        return {"type": "leaf", "label": target_name + "_" + str(most_appropriate_label(data)[0])}
    best_feat_idx, subset_l_feats, subset_r_feats, subset_l_labels, subset_r_labels = split_data(data, feature_set, gini_instead_of_entropy)
    node = {"type": "node", "feature": feature_names[best_feat_idx]}
    if subset_l_labels.shape[0] != 0:
        node["left"] = grow_tree(subset_l_labels, subset_l_feats, feature_names, target_name, gini_instead_of_entropy)
    else:
        node["left"] = {"type": "leaf", "label": target_name + "_" + str(most_appropriate_label(data)[0])}
    if subset_r_labels.shape[0] != 0:
        node["right"] = grow_tree(subset_r_labels, subset_r_feats, feature_names, target_name, gini_instead_of_entropy)
    else:
        node["right"] = {"type": "leaf", "label": target_name + "_" + str(most_appropriate_label(data)[0])}
    return node 

def predict_with_tree():
    
    return

data = data_wine_labels_train
feature_set = data_wine_feats_train
feature_names = data_wine_feature_names
target_name = data_wine_target_name

