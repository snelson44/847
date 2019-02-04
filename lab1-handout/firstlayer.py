"""
FirstLayer Class

Neuromorphic Lab 1

Sam Nelson
samueljn@andrew.cmu.edu
"""

import math
import csv
import numpy as np
from scipy import signal

import warnings

from sklearn.preprocessing import MinMaxScaler

from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#Layer may not be both the first layer and an output layer
class FirstLayer: 
    def __init__ (self, layer_id, training_raw_data, training_target):
        self.layer_id = layer_id
        self.raw_data = training_raw_data
        self.target = training_target

    def scale_data(self, data, lower_bound, upper_bound):
        scaler = MinMaxScaler(feature_range=(lower_bound, upper_bound))
        scaled_data = scaler.fit_transform(data)
        return scaled_data

    def preprocess (self, my_filter, num_bits=3):
        num_images, height, width = self.raw_data.shape

        bound = 2**num_bits

        scaled_images = []
        for image in self.raw_data:
            scaled_image = self.scale_data(image, 0, bound-1)
            scaled_image_int = scaled_image.astype(int)
            scaled_images.append(scaled_image_int)
        
        return(scaled_images)
        # return np.zeros(d2_raw_data.shape) 


    def generate_spikes(self, pixel_val, threshold, max_val):
        if pixel_val <= 0:
            return 0
        elif pixel_val >= threshold:
            return 0

        # smaller pixel vals correlate to more intense inputs
        pixel_2_spike = max_val - pixel_val 

        return pixel_2_spike


    def write_spiketimes(self, path, spike_volleys, pixel_volleys):

        with open(path + 'spiketimes.csv', mode='w') as spiketimes:
            csv_writer = csv.writer(spiketimes, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(['image_number', 'spike_position', 'spike_time'])
            print("writing spiketimes")
            for image_index, spike_volley in enumerate(spike_volleys):
                for row_index, row in enumerate(spike_volley):
                    for col_index, col in enumerate(spike_volley):
                        csv_writer.writerow([image_index+1, 
                            int(pixel_volleys[image_index][row_index][col_index]),
                            int(spike_volley[row_index][col_index])]) 
        
        spiketimes.close()

        return