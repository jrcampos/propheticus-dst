"""
Contains the code required to extend the existing LabelBinarizer for the binary class
Source code:
https://stackoverflow.com/questions/31947140/sklearn-labelbinarizer-returns-vector-when-there-are-2-classes
"""
import numpy
import sklearn.preprocessing

class ExtendedLabelBinarizer:
    def __init__(self):
        self.LabelBinarizer = sklearn.preprocessing.LabelBinarizer()

    def fit(self, X):
        # Convert X to array
        X = numpy.array(X)
        # Fit X using the LabelBinarizer object
        self.LabelBinarizer.fit(X)
        # Save the classes
        self.classes_ = self.LabelBinarizer.classes_

    def fit_transform(self, X):
        # Convert X to array
        X = numpy.array(X)
        # Fit + transform X using the LabelBinarizer object
        Xlb = self.LabelBinarizer.fit_transform(X)
        # Save the classes
        self.classes_ = self.LabelBinarizer.classes_
        if len(self.classes_) == 2:
            Xlb = numpy.hstack((Xlb, 1 - Xlb))

        return Xlb

    def transform(self, X):
        # Convert X to array
        X = numpy.array(X)
        # Transform X using the LabelBinarizer object
        Xlb = self.LabelBinarizer.transform(X)
        if len(self.classes_) == 2:
            '''
            NOTE: 
            - changed from original to be consistent with attribute classes_ order
            - original was: Xlb = numpy.hstack((Xlb, 1 - Xlb))
            '''
            Xlb = numpy.hstack((1 - Xlb, Xlb))

        return Xlb

    def inverse_transform(self, Xlb):
        # Convert Xlb to array
        Xlb = numpy.array(Xlb)
        if len(self.classes_) == 2:
            X = self.LabelBinarizer.inverse_transform(Xlb[:, 0])
        else:
            X = self.LabelBinarizer.inverse_transform(Xlb)
        return X

