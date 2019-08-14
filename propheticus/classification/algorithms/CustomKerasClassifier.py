"""
Wrapper class to extend the keras framekwork to the scikit interface
"""
import keras.wrappers.scikit_learn
from sklearn.base import BaseEstimator, ClassifierMixin


# six.with_metaclass(ABCMeta, BaseEstimator)
class CustomKerasClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, dataset_metadata, **kwargs):
        if 'build_fn' not in kwargs:
            kwargs['build_fn'] = self.create_model

        self.dataset_metadata = dataset_metadata
        self.arguments = kwargs
        self.model = keras.wrappers.scikit_learn.KerasClassifier(**kwargs)

    def create_model(self):
        model = keras.models.Sequential()

        model.add(keras.layers.Dense(20, input_dim=self.dataset_metadata['features_count'], activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(20, activation='relu'))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(1, activation='sigmoid'))  # last dimrnsion defines class? 1 = binay?

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def fit(self, X, Y):
        return self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

