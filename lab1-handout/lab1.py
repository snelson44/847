"""
Main

Neuromorphic Lab 1

Sam Nelson
samueljn@andrew.cmu.edu
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from PIL import Image
import firstlayer as firstlayer
import filters as filters

# define width and height of image in pixels
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# if pixel val is >= 4, there is no spike
SPIKE_THRESHOLD = 4

MAX_PIXEL_VALUE = 8

PATH = 'C:/Users/Sam/college/847/lab1-handout/'

def generate_receptive_field(middle_row, middle_col):
	middle_row = int(middle_row)
	middle_col = int(middle_col)
	receptive_field_rows = [middle_row-1, middle_row, middle_row+1]
	receptive_field_cols = [middle_col-1, middle_col, middle_col+1]
	receptive_field = [receptive_field_rows, receptive_field_cols]

	return receptive_field

def generate_volley(image, middle_row, middle_col):
	receptive_field = generate_receptive_field(middle_row, middle_col)

	volley = np.zeros((3, 3))
	row = receptive_field[0]
	row_index = 0
	for col_index, col in enumerate(row):
		volley_x = receptive_field[row_index][col_index]
		for col_index, col in enumerate(row):
			volley_y = receptive_field[row_index][col_index]

			volley[row_index][col_index] = image[volley_x][volley_y]

	return volley

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="on/off center filter images")
    parser.add_argument('-f', '--filter',
                        help="Type of filter",
                        choices=['oncenter', 'offcenter', 'left_edge'],
                        default='oncenter')

    return parser.parse_args(args)


if __name__ == '__main__':
	cell_dict = {
		'left_up': None,
		'left_down': None,
		'left': None,
		'right_up': None,
		'right_down': None,
		'right': None,
		'top': None,
		'bottom': None,
		'center': None,
		'avg_list': None,
		'center_list': None,
		'neighbors': None
	}

	args = parse_args()

	# get filter typ fromuser input
	filter_type = args.filter

	# fetch ML data
	mnist = fetch_mldata('MNIST original')
	N, _ = mnist.data.shape

	# Reshape the data to be square
	mnist.square_data = mnist.data.reshape(N, IMAGE_WIDTH, IMAGE_HEIGHT)

	layer1 = firstlayer.FirstLayer(1, mnist.square_data, mnist.target)

	# create filter
	my_filter = filters.Filter(filter_type)

	# normalize data to 3 bits
	normalized_images = layer1.preprocess(my_filter, 3)

# 3. On Center Off Center Filtering


	middle_row = (IMAGE_HEIGHT / 2) - 1
	middle_col = (IMAGE_WIDTH / 2) - 1
	for normalized_image in normalized_images:
		# create pixel values
		pixel_vals = my_filter.run(cell_dict, normalized_image, IMAGE_WIDTH, IMAGE_HEIGHT)

		# 4. Visualize Your Outputs

		spike_vals = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
		for row_index, row in enumerate(pixel_vals):
			for col_index, pixel in enumerate(row):
				spike_vals[row_index][col_index] = layer1.generate_spikes(pixel, SPIKE_THRESHOLD, MAX_PIXEL_VALUE)
				# print("Row: {}, Col: {}, Pixel Value: {}, Spike Value: {}".format(row_index, col_index, pixel_vals[row_index][col_index], spike_vals[row_index][col_index]))

		# generate spike volley and pixel volley
		spike_volley = generate_volley(spike_vals, middle_row, middle_col)
		pixel_volley = generate_volley(pixel_vals, middle_row, middle_col)

		layer1.write_spiketimes(normalized_image, PATH, pixel_volley, spike_volley)

	# pixel = plt.figure()
	# plt.title('Pixel values')
	# pixel_vals_vis = np.ma.masked_where(pixel_vals <= 0, pixel_vals)
	# pixel_vals_plot = plt.pcolor(pixel_vals_vis, cmap = "gray")

	# spike = plt.figure()
	# plt.title('Spiketimes')
	# spike_vals_vis = np.ma.masked_where(spike_vals <= 0, spike_vals)
	# spike_vals_plot = plt.pcolor(spike_vals_vis, cmap = "gray")

	# plt.show()


# 5. Spiketimes 
