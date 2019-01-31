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
	x = layer1.preprocess(my_filter, 3)

	# create pixel values
	pixel_vals = my_filter.run(cell_dict, x[1963], IMAGE_WIDTH, IMAGE_HEIGHT)

	# PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
	# img = Image.new( 'RGB', (28,250), "black") # create a new black image
	# pixels = img.load() # create the pixel map

	# for i in range(img.size[0]):    # for every col:
	#     for j in range(img.size[1]):    # For every row
	#         pixels[i,j] = (i, j, 100) # set the colour accordingly

	# img.show()

	spike_vals = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))
	for row_index, row in enumerate(pixel_vals):
		for col_index, pixel in enumerate(row):
			spike_vals[row_index][col_index] = layer1.generate_spikes(pixel, SPIKE_THRESHOLD, MAX_PIXEL_VALUE)
			print("Row: {}, Col: {}, Pixel Value: {}, Spike Value: {}".format(row_index, col_index, pixel_vals[row_index][col_index], spike_vals[row_index][col_index]))


	# normalized_spike_vals = layer1.scale_data(spike_vals, 0, 1)

	# print(normalized_spike_vals)
	# print('\n')

	pixel = plt.figure()
	plt.title('Pixel values')
	pixel_vals_vis = np.ma.masked_where(pixel_vals <= 0, pixel_vals)
	pixel_vals_plot = plt.pcolor(pixel_vals_vis, cmap = "gray")

	spike = plt.figure()
	plt.title('Spiketimes')
	spike_vals_vis = np.ma.masked_where(spike_vals <= 0, spike_vals)
	spike_vals_plot = plt.pcolor(spike_vals_vis, cmap = "gray")

	plt.show()

# 3. On Center Off Center Filtering


# 4. Visualize Your Outputs


# 5. Spiketimes 
