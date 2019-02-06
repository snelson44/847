import math
import numpy as np
from scipy import signal

#Layer may not be both the first layer and an output layer
class FirstLayer: 
    def __init__ (self, layer_id, training_raw_data, training_target):
        self.layer_id = layer_id
        self.raw_data = training_raw_data
        self.target = training_target

    def preprocess (self, my_filter, num_bits=3 ):
        return np.zeros(self.raw_data.shape) 


    def generate_spikes(self, threshold):
        return


    def write_spiketimes(self, path, receptive_field): 
        return