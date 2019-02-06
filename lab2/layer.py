import math
import numpy as np
import matplotlib.pyplot as plt

class Layer():
    def __init__(self, layer_id, prev_layer, threshold, receptive_field):  
        self.layer_id = layer_id
        self.prev_layer = prev_layer
        self.threshold = threshold
        self.rf = receptive_field
        self.N,_,_ = self.prev_layer.raw_data.shape

    def reset(self): 
        # Reset the network, clearing out any accumulator variables, etc
        return
    
    def process_image(self):
        """
        This function will control the different processing steps for a 
        single image

        Notice that you can get to values in the previous layer through 
        self.prev_layer
        """

        return


    def write_spiketimes(self, path, receptive_field): 
        # create a file with: image_number, spike_position, spike_time
        return  
