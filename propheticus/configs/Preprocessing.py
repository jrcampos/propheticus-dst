import numpy

class Preprocessing(object):
    PreprocessingCallDetails = {
        'normalize': {
            'package': 'sklearn.preprocessing',
            'callable': 'StandardScaler',
            'parameters': {
                'copy': {'type': ''},
                'with_mean': {'type': ''},
                'with_std': {'type': ''}
            }
        },
    }
