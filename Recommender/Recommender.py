import networkx as nx
import random
import math
import csv
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
import xgboost as xgb

    
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
            if shortest_path >=3:
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
  
    #print("----------------extract positive samples--------------------")
    # randomly select pos_num as test edges
    pos_sample = random.sample(list(g.edges()), pos_num)
    sample_g = nx.Graph()
    sample_g.add_edges_from(pos_sample)
    nx.write_edgelist(sample_g, "sample_positive_" +str(pos_num)+ ".txt", data=['positive'])
    # adding non-existing edges
    #print("----------------extract negative samples--------------------")
    neg_g = nx.Graph()
    produce_fake_edge(g,neg_g,neg_num)
    nx.write_edgelist(neg_g, "sample_negative_" +str(neg_num)+ ".txt", data=["positive"])
    neg_sample = neg_g.edges()
    nx.write_edgelist(neg_g, "sample_combine_" +str(pos_num + neg_num)+ ".txt", data=["positive"])
    # compute common neighbors for all samples
    combined(g,pos_sample,list(neg_sample))
    
    # get list of available nodes in the graph
    available_nodes = list(g.nodes())[:100]
    
    # print available nodes
    print("Available nodes:", available_nodes)
    print("Number of nodes in the graph:", len(available_nodes))
    # Prompt user to select a node
    selected_node = input("Enter a node from the graph: ")

    # Check if the selected node is in the graph
    if selected_node not in g:
        print("Node not found in the graph.")
    else:
        # Get neighbors of the selected node
        neighbors = list(g.neighbors(selected_node))
        num_neighbors = len(neighbors)
        print("Number of neighbors of Node", selected_node, ":", num_neighbors)
        print("Neighbors of Node", selected_node, ":", neighbors)

        # Remove some neighbors of the selected node
        num_to_remove = int(input("Enter the number of neighbors to remove: "))
        if num_to_remove > num_neighbors:
            print("Cannot remove more neighbors than exist. Removing all neighbors instead.")
            num_to_remove = num_neighbors
        
        random.shuffle(neighbors)
        removed_neighbors = neighbors[:num_to_remove]
        remaining_neighbors = [n for n in neighbors if n not in removed_neighbors]
        num_remaining_neighbors = len(remaining_neighbors)
        
        # Print the filtered neighbors
        print("Removed neighbors:")
        print(removed_neighbors)
        print("Remaining neighbors of node", selected_node, ":")
        print(remaining_neighbors)
        print("Number of neighbors before removal:", num_neighbors)
        print("Number of neighbors after removal:", num_remaining_neighbors)
        

    r=np.loadtxt(open("features.csv", "rb"), delimiter=",", skiprows=1);
    l,b=r.shape;
    np.random.shuffle(r);
    train_l=int(0.7*l)
    X_train=r[0:train_l,0:b-1]
    Y_train=r[0:train_l,b-1]
    X_train = normalize(X_train, axis=0, norm='max')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)  
    X_train = scaler.transform(X_train)  
    
    dt_clf = DecisionTreeClassifier(random_state=1)
    dt_clf.fit(X_train,Y_train)
    
    # Get the neighbors of selected_node
    neighbors_selected = set(g.neighbors(selected_node))

    # Get the neighbors of neighbors of selected_node
    neighbors_of_neighbors_selected = set()
    for neighbor in neighbors_selected:
        neighbors_of_neighbor = set(g.neighbors(neighbor))
    neighbors_of_neighbors_selected = neighbors_of_neighbors_selected.union(neighbors_of_neighbor)
    # Combine the two sets using the union() method
    potentials = neighbors_selected.union(neighbors_of_neighbors_selected)

    # Extract features for potential nodes and normalize the data
    X_potential = np.array([g.degree(node) for node in potentials]).reshape(-1, 1)
    X_potential = normalize(X_potential, axis=0, norm='max')
    X_potential = scaler.transform(X_potential)

    # Use the DT classifier to predict the labels and probabilities of potential nodes
    y_prob_potential = dt_clf.predict_proba(X_potential)
    y_label_potential = dt_clf.predict(X_potential)

    # Find the potential nodes with a positive prediction and probability greater than 0.7, and recommend links to them
    for node, label, prob in zip(potentials, y_label_potential, y_prob_potential[:,1]):
        if label == 1 and prob > 0.7:
            g.add_edge(selected_node, node)
            print(f"Link recommended between nodes {selected_node} and {node}")
            
    return pos_sample,neg_sample
         
def main(filename="facebook_combined.txt",feature_name=feature_set):
    g = create_graph_from_file(filename)
    num_edges = g.number_of_edges()
    pos_num = int(num_edges * 0.009)
    neg_num = int(num_edges * 0.009)
    pos_sample, neg_sample = sample_extraction(g, pos_num, neg_num)
#__________________Entry Point________________________
fn="facebook_combined.txt"
#Run this line to genrate feature Set
main(filename=fn, feature_name=feature_set)

