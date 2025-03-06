from sklearn.neural_network import MLPClassifier #ANN
import numpy as np
from sklearn.metrics import accuracy_score

# Importing the parent: DataPreprocessing class from data_preprocess.py
from src.data_preprocess import DataPreprocessing 


class ModelBuilder(DataPreprocessing):
    def __init__(self, *args, **kwargs):
        super(ModelBuilder, self).__init__(*args, **kwargs)

    def dt(self, X_train, X_test, y_train, y_test):
        #Create ANN model
        ann_classifier = MLPClassifier(max_iter=300,hidden_layer_sizes=100,learning_rate_init=0.1,random_state=42)

        #Train the model
        ann_classifier.fit(X_train, y_train)

        #Test the model
        ann_predicted = ann_classifier.predict(X_test)

        error = 0
        for i in range(len(y_test)):
            error += np.sum(ann_predicted != y_test)

        total_accuracy = 1 - error / len(y_test)

        #get performance
        self.accuracy = accuracy_score(y_test, ann_predicted)

        return ann_classifier