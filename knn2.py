import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn(train_data, test_instance, k):
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i][:-1], test_instance)
        distances.append((train_data[i], dist))
    distances.sort(key=lambda x: x[1])
    
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
        
    classes = [neighbor[-1] for neighbor in neighbors]
    vote = Counter(classes).most_common(1)[0][0]
    return vote

def train_and_evaluate(train_file, test_file, k):
    train_data = pd.read_csv(train_file, header=None).values
    test_data = pd.read_csv(test_file, header=None).values
    
    predictions = []
    for i in range(len(test_data)):
        vote = knn(train_data, test_data[i][:-1], k)
        predictions.append(vote)
        
    accuracy = accuracy_score(test_data[:,-1], predictions)
    confusion = confusion_matrix(test_data[:,-1], predictions)
    
    print(f"Accuracy: {accuracy * 100} %")
    print(f"Confusion Matrix:\n {confusion}")

k_values = [1,3]
for k in k_values:
    print(f"Evaluating with k={k}")
    train_and_evaluate("data\\fp\\fp.trn","data\\fp\\fp.tst", k)