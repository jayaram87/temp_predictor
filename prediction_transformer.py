import numpy as np
import pickle
from logger import Logger

class Predictor_Data_Transformer:

    def __init__(self, Type, process_temp, rpm, torque, wear, failure, twf, hdf, pwf, osf, rnf):
        try:
            self.type = Type
            self.temp = process_temp
            self.rpm = rpm
            self.torque = torque
            self.wear = wear
            self.failure = failure
            self.twf = twf
            self.hdf = hdf
            self.pwf = pwf
            self.osf = osf
            self.rnf = rnf
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt initialize the transformer \n {str(e)}')

    def data(self):
        try:
            data = np.zeros((1, 13))
            scaler = pickle.load(open('std_scaler.sav', 'rb'))
            data[:, 0:4] = scaler.transform([[self.temp, self.rpm, self.torque, self.wear]])
            data[:, 4] = self.failure
            data[:, 5] = self.twf
            data[:, 6] = self.hdf
            data[:, 7] = self.pwf
            data[:, 8] = self.osf
            data[:, 9] = self.rnf
            if self.type == 'H':
                data[:, 10] = int(1)
            elif self.type == 'L':
                data[:, 11] = int(1)
            elif self.type == 'M':
                data[:, 12] = int(1)
            return data
        except Exception as e:
            Logger('test.log').logger('ERROR', f'Couldnt transform the prediction data \n {str(e)}')
