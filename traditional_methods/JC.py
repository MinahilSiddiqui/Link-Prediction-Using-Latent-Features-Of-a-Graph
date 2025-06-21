import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import csv
import seaborn as sns
from sklearn import svm
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.metrics import roc_curve, auc

def jaccard_coefficient(g, pos_edges, neg_edges):
    result1=[]
    for edge in pos_edges + neg_edges:
        node_one, node_two = edge[0], edge[1]
        neighbors_one = set(g.neighbors(node_one))
        neighbors_two = set(g.neighbors(node_two))
        jaccard = len(neighbors_one.intersection(neighbors_two)) / len(neighbors_one.union(neighbors_two))
        label = 1 if edge in pos_edges else 0
        result1.append((jaccard,label))
    with open("jctwt.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)  
        writer.writerow(['jaccard_coefficient', 'label'])
        writer.writerows(result1)


feature_set = [jaccard_coefficient ]

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
    print("----------------build graph--------------------")
    f = open(filename, "rb")
    g = nx.read_edgelist(f)
    return g

def sample_extraction(g, pos_num, neg_num):
  
    print("----------------extract positive samples--------------------")
    # randomly select pos_num as test edges
    pos_sample = random.sample(list(g.edges()), pos_num)
    sample_g = nx.Graph()
    sample_g.add_edges_from(pos_sample)
    #nx.write_edgelist(sample_g, "sample_positive_" +str(pos_num)+ ".txt", data=['positive'])
    # adding non-existing edges
    print("----------------extract negative samples--------------------")
    neg_g = nx.Graph()
    produce_fake_edge(g,neg_g,neg_num)
    nx.write_edgelist(neg_g, "sample_negative_" +str(neg_num)+ ".txt", data=["positive"])
    neg_sample = neg_g.edges()
    #nx.write_edgelist(neg_g, "sample_combine_" +str(pos_num + neg_num)+ ".txt", data=["positive"])
    # compute jaccard coficient for all samples
    jaccard_coefficient(g,pos_sample,list(neg_sample))
    return pos_sample, neg_sample

def main(filename,feature_name=feature_set):
    g = create_graph_from_file(filename)
    num_edges = g.number_of_edges()
    pos_num = int(num_edges * 0.009)
    neg_num = int(num_edges * 0.009)
    pos_sample, neg_sample = sample_extraction(g, pos_num, neg_num)

#______________________Entry Point________________________
fn="twitter_combined.txt"
#Run this line to genrate feature Set
main(filename=fn, feature_name=feature_set)


r=np.loadtxt(open("jctwt.csv", "rb"), delimiter=",", skiprows=1);
l,b=r.shape;
np.random.shuffle(r);
train_l=int(0.7*l)
X_train=r[0:train_l,0:b-1]
Y_train=r[0:train_l,b-1]
X_test=r[train_l:l,0:b-1]
Y_test=r[train_l:l,b-1]
X_train = normalize(X_train, axis=0, norm='max')
X_test = normalize(X_test, axis=0, norm='max')
scaler = StandardScaler()  
scaler.fit(X_train)  
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

def mySvm(training, training_labels, testing, testing_labels):
    #Support Vector Machine
    clf = svm.SVC()
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the SVM classifier ++++++++++++")
    result = clf.predict(testing)
    accuracy = accuracy_score(testing_labels, result)
    print(metrics.classification_report(Y_test, result))
   # confusion matrix
    cf_matrix = confusion_matrix(testing_labels, result)
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    #group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    # plot confusion matrix
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (SVM Accuracy: {accuracy:.2f})')
    plt.show()

#Run this to for SVM classification
mySvm(X_train,Y_train,X_test,Y_test)

def ANN(training, training_labels, testing, testing_labels):
    clf = MLPClassifier(solver='adam', alpha=1e-3,hidden_layer_sizes=(16,9),max_iter=500, random_state=1)
    clf.fit(training, training_labels)
    print ("+++++++++ Finishing training the ANN classifier ++++++++++++")
    result = clf.predict(testing)
    accuracy = accuracy_score(testing_labels, result)
    # confusion matrix
    cf_matrix = confusion_matrix(testing_labels, result)
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    # plot confusion matrix
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (ANN Accuracy: {accuracy:.2f})')
    plt.show()
    print(metrics.classification_report(testing_labels, result))
    
# Run this for ANN classification
ANN(X_train,Y_train,X_test,Y_test)

def myRandomForest(training, training_labels, testing, testing_labels):
    #Random Forest
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(training, training_labels)
    print("+++++++++ Finishing training the Random Forest classifier ++++++++++++")
    result = clf.predict(testing)
    accuracy = accuracy_score(testing_labels, result)
    print(metrics.classification_report(Y_test, result))
    # confusion matrix
    cf_matrix = confusion_matrix(testing_labels, result)
    group_names = ['TN', 'FP', 'FN', 'TP']
    group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_counts)]
    labels = np.asarray(labels).reshape(2,2)
    # plot confusion matrix
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix (Random Forest Accuracy: {accuracy:.2f})')
    plt.show()

#Run this to for Random Forest classification
myRandomForest(X_train,Y_train,X_test,Y_test)