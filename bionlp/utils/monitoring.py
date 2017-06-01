"""
I wanted to create a Singleton for data storage, but several discussions in forums suggest that the
'module-approach' is more pythonic:
https://stackoverflow.com/questions/31875/is-there-a-simple-elegant-way-to-define-singletons/31887#31887
https://stackoverflow.com/questions/6841853/python-accessing-module-scope-vars/6842257#6842257
"""

TYPE_VALIDATION = 'validation'
TYPE_TRAINING = 'training'


class MonitoringDataObject(object):
    KEY_ACC = 'accuracy'
    KEY_LOSS_TOT = 'loss_total'
    KEY_LOSS_CRF = 'loss_crf'
    KEY_LOSS_NET_CRF = 'loss_net_crf'

    @staticmethod
    def get_data_keys():
        return [
            MonitoringDataObject.KEY_ACC,
            MonitoringDataObject.KEY_LOSS_CRF,
            MonitoringDataObject.KEY_LOSS_NET_CRF,
            MonitoringDataObject.KEY_LOSS_TOT
        ]

    def __init__(self):
        super(MonitoringDataObject, self).__init__()
        self.__data = dict((key, []) for key in self.get_data_keys())

    def add_iteration_data(self, value_dict):
        assert set(value_dict.keys()) == set(self.get_data_keys()), \
            "input data does not contain the necessary keys {}".format(self.get_data_keys())
        [self.__data[key].append(value) for key, value in value_dict.items()]

    def get_data(self):
        return self.__data


data = {
    TYPE_VALIDATION: MonitoringDataObject(),
    TYPE_TRAINING: MonitoringDataObject()
}


def add_iteration_data(data_type, data_dict):
    assert data_type in data, "provided data type {0) not available".format(data_type)
    data[data_type].add_iteration_data(data_dict)


def get_data():
    return {data_type: data_object.get_data() for data_type, data_object in data.items()}
