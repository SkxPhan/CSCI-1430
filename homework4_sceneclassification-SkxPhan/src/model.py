import numpy as np
from sklearn import svm


class InvalidLabelException(Exception):
    "Raised when the expected number of labels is incorrect"
    pass


class SVM(object):

    def __init__(self):
        self.model = svm.LinearSVC()

    def train(self, inputs, labels):
        """
        DO NOT MODIFY THIS FUNCTION
        This function trains the binary SVM classifier based on the inputs and corresponding labels

        Inputs:
        inputs: a nxd numpy matrix where n is the number of samples and d is the number of features
        labels: a 1xn numpy array where n is the number of samples

        Outputs:
        weights, bias: parameters of the classifier that define the hyperplane.
        """
        try:
            if np.size(np.unique(labels)) > 2:
                raise InvalidLabelException
            self.model.fit(inputs, labels)
            weights = self.model.coef_
            bias = self.model.intercept_
            return weights, bias
        except InvalidLabelException:
            print(
                "Exception occurred: Inputted more than 2 unique labels, only accepts 2 unique labels"
            )
            raise
