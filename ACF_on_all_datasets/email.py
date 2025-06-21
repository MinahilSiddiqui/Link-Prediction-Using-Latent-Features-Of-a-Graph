import networkx as nx
import random
import matplotlib.pyplot as plt
import math
import csv
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report

def combined(g,pos_edges,neg_edges):
    result = []
    for edge in pos_edges + neg_edges:
        node_one, node_two = edge[0], edge[1]
        neighbors_one = set(g.neighbors(node_one)).union(set(g.neighbors(n) for n in g.neighbors(node_one)))
        neighbors_two = set(g.neighbors(node_two)).union(set(g.neighbors(n) for n in g.neighbors(node_two)))
        num_common_neighbors = len(neighbors_one.intersection(neighbors_two))
        
        aa = 0
        for i in set(g.neighbors(node_one)).intersection(set(g.neighbors(node_two))):
           aa += 1 / math.log(len(list(g.neighbors(i))))
           
           
        label = 1 if edge in pos_edges else 0
        result.append((num_common_neighbors,aa,label))
        
    with open("features.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)  
        writer.writerow(['num_common_neighbors','aa_score', 'label'])
        writer.writerows(result)

feature_set = [ combined ]

def produce_fake_edge(g, neg_g,num_test_edges):
    i = 0
    while i < num_test_edges:
        edge = random.sample(list(g.nodes()), 2)
        try:
            shortest_path = nx.shortest_path_length(g,source=edge[0],target=edge[1])
            if shortest_path >= 2:
                neg_g.add_edge(edge[0],edge[1])
                i += 1
        except:
            pass

def create_graph_from_file(filename):
    print("-------------------build graph-----------------------------")
    f = open(filename, "rb")
    g = nx.read_edgelist(f)
    return g

def sample_extraction(g, pos_num, neg_num):
  
    print("----------------extract positive samples--------------------")
    # randomly select pos_num as test edges
    pos_sample = random.sample(list(g.edges()), pos_num)
    sample_g = nx.Graph()
    sample_g.add_edges_from(pos_sample)
    nx.write_edgelist(sample_g, "sample_positive_" +str(pos_num)+ ".txt", data=['positive'])
    # adding non-existing edges
    print("----------------extract negative samples--------------------")
    neg_g = nx.Graph()
    produce_fake_edge(g,neg_g,neg_num)
    nx.write_edgelist(neg_g, "sample_negative_" +str(neg_num)+ ".txt", data=["positive"])
    neg_sample = neg_g.edges()
    nx.write_edgelist(neg_g, "sample_combine_" +str(pos_num + neg_num)+ ".txt", data=["positive"])
    # compute common neighbors for all samples
    combined(g,pos_sample,list(neg_sample))
    return pos_sample, neg_sample
 

def main(filename,feature_name=feature_set):
    g = create_graph_from_file(filename)
    num_edges = g.number_of_edges()
    pos_num = int(num_edges *0.09)
    neg_num = int(num_edges *0.09)
    pos_sample, neg_sample = sample_extraction(g, pos_num, neg_num)

#______________________Entry Point________________________
fn="Email.txt"
#Run this line to genrate feature Set
main(filename=fn, feature_name=feature_set)


r=np.loadtxt(open("features2.csv", "rb"), delimiter=",", skiprows=1);
l,b=r.shape;
np.random.shuffle(r);
train_l=int(0.7*l)
X_train=r[0:train_l,0:b-1]
Y_train=r[0:train_l,b-1]
X_test=r[train_l:l,0:b-1]
Y_test=r[train_l:l,b-1]
X_train = normalize(X_train, axis=0, norm='max')
X_test = normalize(X_test, axis=0, norm='max')
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

# Define the ensemble function
def ensemble_classifiers(training, training_labels, testing, testing_labels):
    result=[]
    # Train the Decision Tree classifier
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(training, training_labels)

    # Train the XGBoost classifier
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(training, training_labels)
    
    #train ANN classifier
    ann_clf = MLPClassifier(solver='adam', alpha=1e-3,hidden_layer_sizes=(16,9),max_iter=500, random_state=1)
    ann_clf.fit(training, training_labels)

    # train random forest classifier
    rfc_clf = RandomForestClassifier(n_estimators=100)
    rfc_clf.fit(training, training_labels)
    
    # Predict labels using both classifiers
    dt_preds = dt_clf.predict(testing)
    xgb_preds = xgb_clf.predict(testing) 
    ann_preds = ann_clf.predict(testing)
    ensemble_preds = []
    for dt_pred, xgb_pred, ann_preds in zip(dt_preds, xgb_preds, ann_preds):
        # Use a weighted approach based on the number of 1s predicted by each model
        num_ones = sum([dt_pred, xgb_pred,ann_preds])
        if num_ones >= 2:
            label = 1
        else:
            label = 0
        ensemble_preds.append(label)
        result.append([dt_pred, xgb_pred, ann_preds, label])
    with open("ensemble_predictions.csv", 'w', newline='') as csvfile:
           writer = csv.writer(csvfile)
           writer.writerow(['dt_pred','xgb_pred','ann_pred', 'label'])
           writer.writerows(result)
    # Compute accuracy of ensemble predictions
    accuracy= accuracy_score(testing_labels,ensemble_preds)
    # Print classification report for ensemble predictions
    print("Classification Report for Ensemble Predictions:")
    print(classification_report(testing_labels, ensemble_preds))

     # Compute and plot confusion matrix for ensemble predictions
    cf_matrix = confusion_matrix(testing_labels, ensemble_preds)
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names, group_counts)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (Ensemble Accuracy: {accuracy:.2f})')
    plt.show()
    
# Run the ensemble function
ensemble_classifiers(X_train, Y_train, X_test, Y_test)
