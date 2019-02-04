"""
Main

Neuromorphic Lab 1

Sam Nelson
samueljn@andrew.cmu.edu
"""

import numpy as np
import argparse
import yaml
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from PIL import Image
import firstlayer as firstlayer
import filters as filters


def generate_receptive_field(middle_row, middle_col):
	"""Generate a 3x3 receptive field
	Args:
		middle_row(int): middle row index of 3x3 receptive field w.r.t larger image
		middle_col(int): middle column index of 3x3 receptive field w.r.t larger image
	Returns:
		(receptive_field(np.arr: int)): 3x3 matrix of the receptive field indexes
	"""
	middle_row = int(middle_row)
	middle_col = int(middle_col)
	receptive_field_rows = [middle_row-1, middle_row, middle_row+1]
	receptive_field_cols = [middle_col-1, middle_col, middle_col+1]
	receptive_field = [receptive_field_rows, receptive_field_cols]

	return receptive_field


def generate_volley(image, middle_row, middle_col):
	"""Get values from a receptive field of a matrix
	Args:
		image(np.arr): image
		middle_row(int): middle row index of 3x3 receptive field w.r.t larger image
		middle_col(int): middle column index of 3x3 receptive field w.r.t larger image
	Returns:
		(volley(np.arr: int)): 3x3 matrix of the receptive field from larger image
	"""
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


def get_spike_vals(pixel_vals, spike_threshold, max_pixel_value):
	"""Convert spike positions into spike times
	Args:
		pixel_vals(np.arr): spike positions
		spike_threshold(int): if spike position exceeds threshold, consider it no spike
		max_pixel_value(int): the max value a spike position can have
	Returns:
		(spike_vals(np.arr: int)): spiketimes for image 
	"""
	spike_vals = np.zeros((image_height, image_width))
	for row_index, row in enumerate(pixel_vals):
		for col_index, pixel in enumerate(row):
			spike_vals[row_index][col_index] = layer1.generate_spikes(pixel, spike_threshold, max_pixel_value)

	return(spike_vals)


def apply_filter(image,
				 filter_type,
				 cell_dict,
				 image_width,
				 image_height,
				 max_pixel_value):
	"""Apply filter on a normalized image
	Args:
		image(np.arr): image to apply filter to
		cell_dict (dict): contains keys for each surounding pixel
		image_width(int): number of pixels wide
		image_height(int): number of pixels high
		max_pixel_value(int): the max value a spike position can have
	Returns:
		(pixel_vals(np.arr: int)): spike positions for image 
	"""
	# create pixel values
	pixel_vals = my_filter.run(cell_dict, image, image_width, image_height)

	if filter_type =='sobel':
		sobel_data = firstlayer.FirstLayer(1, pixel_vals, 0)
		pixel_vals = sobel_data.scale_data(pixel_vals, 0, max_pixel_value+1)

	return(pixel_vals)

def visualize_results(pixel_vals, spike_vals, filter_type):
	"""visualize spike positions and times
	Args:
		pixel_vals(np.arr): spike positions for given image
		spike_vals(np.arr): spiketimes for given image
		filter_type(str): filter used on given image
	Returns:
		plots of spiketimes and positions
	"""
	pixel = plt.figure()
	plt.title(filter_type + ' spike positions')
	pixel_vals_vis = np.ma.masked_where(pixel_vals <= 0, pixel_vals)
	pixel_vals_plot = plt.pcolor(pixel_vals_vis, cmap = "gray")

	spike = plt.figure()
	plt.title(filter_type + ' spiketimes')
	spike_vals_vis = np.ma.masked_where(spike_vals <= 0, spike_vals)
	spike_vals_plot = plt.pcolor(spike_vals_vis, cmap = "gray")

	plt.show()

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="on/off center filter images")
    parser.add_argument('-f', '--filter',
                        help="Type of filter",
                        choices=['oncenter', 'offcenter', 'sobel'],
                        default='oncenter')

    return parser.parse_args(args)


def run(cell_dict, config, layer1, my_filter, normalized_images):
	"""Get pixel_vals and spike vals, generate plots and csv file
	Args:
		cell_dict (dict): contains keys for each surounding pixel
		config (yaml): contains values of user inputs
		layer1: instance of FirstLayer class
		my_filter: instance of Filter class
		normalized_images(list): list of every image - normalized
	Returns:
		visual representation of spiketimes and positions
	"""
	image_width = config['image_width']
	image_height = config['image_height']
	max_pixel_value = config['max_pixel_value']
	spike_threshold = config['spike_threshold']
	path = config['path']

	spike_volleys = []
	pixel_volleys = []

	middle_row = (image_height / 2) - 1
	middle_col = (image_width / 2) - 1

	normalized_images = [normalized_images[69997], normalized_images[49997], normalized_images[69897], normalized_images[324]]

	for normalized_image in normalized_images:
		# create pixel values
		pixel_vals = apply_filter(normalized_image,
									 filter_type,
									 cell_dict,
									 image_width,
									 image_height,
									 max_pixel_value)

		spike_vals = get_spike_vals(pixel_vals, spike_threshold, max_pixel_value)

		# generate spike volley and pixel volley
		spike_volley = generate_volley(spike_vals, middle_row, middle_col)
		pixel_volley = generate_volley(pixel_vals, middle_row, middle_col)
		spike_volleys.append(spike_volley)
		pixel_volleys.append(pixel_volley)

		# # write spiketimes into csv
		# layer1.write_spiketimes(path, spike_volleys, pixel_volleys)

		# plot results
		visualize_results(pixel_vals, spike_vals, filter_type)


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

	with open('config.yaml', 'r') as confile:
		config = yaml.load(confile.read())

	# fetch ML data
	mnist = fetch_mldata('MNIST original')
	N, _ = mnist.data.shape

	image_width = config['image_width']
	image_height = config['image_height']

	# Reshape the data to be square
	mnist.square_data = mnist.data.reshape(N, image_width, image_height)

	layer1 = firstlayer.FirstLayer(1, mnist.square_data, mnist.target)

	# create filter
	my_filter = filters.Filter(filter_type)

	# normalize data to 3 bits
	normalized_images = layer1.preprocess(3)

	run(cell_dict, config, layer1, my_filter, normalized_images)

