"""
Filter class

Neuromorphic Lab 1

Sam Nelson
samueljn@andrew.cmu.edu
"""

import os
import numpy as np
from numpy import array

# define width and height of image in pixels
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

class Filter():
	def __init__(self, filter_type):
		"""
		Args:
			filter_type (str): on center or off center
		"""
		self.filter_type = filter_type


	def invalid_index(self, x, y, x_max, y_max):
		"""Checks if an index of a matrix is out of bounds
		Args:
			x (int): x index
			y (int): y index
			x_max (int): maximum x index
			y_max (int): maximum y index
		Returns:
			bool : returns 1 if invalid index, 0 if valid index
		"""
		if x < 0 or y < 0:
			return 1
		elif x > x_max or y > y_max:
			return 1
		else:
			return 0


	def count_neighbors(self, image, x, y):
		"""Counts the number of surrounding pixels a specific pixel has
		Args:
			image (np.arr): the image, 28 x 28
			x (int): x index
			y (int): y index
		Returns:
			neighbors (int): returns the number of neighboring pixels
		"""
		neighbors = 8

		if self.invalid_index(x-1, y-1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x-1, y, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x, y-1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x+1, y+1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x+1, y, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x, y+1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x-1, y+1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1
		if self.invalid_index(x+1, y-1, IMAGE_WIDTH-1, IMAGE_HEIGHT-1): neighbors -= 1

		return neighbors


	def get_neighbor(self, image, x, y):
		"""searches matrix for a certain index
		Args:
			image (np.arr): the image, 28 x 28
			x (int): x index
			y (int): y index
		Returns:
			(int): returns the value stored in index, or 0 if it does not exist
		"""
		if self.invalid_index(x, y, IMAGE_WIDTH-1, IMAGE_HEIGHT-1):
			return 0
		else:
			return image[x][y]


	def build_cell(self, image, row_index, col_index, cell_dict):
		"""Store cell position values inside dictionary
		Args:
			image (np.arr): the image, 28 x 28
			row_index (int): the current row of interest in the image
			col_index (int): the current column of interest in the image
			cell_dict (dict): contains keys for each surounding pixel
		Returns:
			(dict(str: int)): returns dictionary with pixel value of each key
		"""
		cell_dict['left_up'] = self.get_neighbor(image, row_index-1, col_index-1)
		cell_dict['left_down'] = self.get_neighbor(image, row_index+1, col_index-1)
		cell_dict['left'] = self.get_neighbor(image, row_index, col_index-1)
		cell_dict['right_up'] = self.get_neighbor(image, row_index-1, col_index+1)
		cell_dict['right_down'] = self.get_neighbor(image, row_index+1, col_index+1)
		cell_dict['right'] = self.get_neighbor(image, row_index, col_index+1)
		cell_dict['top'] = self.get_neighbor(image, row_index-1, col_index)
		cell_dict['bottom'] = self.get_neighbor(image, row_index+1, col_index)
		cell_dict['center'] = self.get_neighbor(image, row_index, col_index)

		cell_dict['neighbors'] = self.count_neighbors(image, row_index, col_index)

		return(cell_dict)


	def calculate_avgs(self, image, width, height, cell_dict):
		"""Calculate and store the average value of each cell and store
			its corresponding center
		Args:
			image (np.arr): the image, width x height
			width (int): the width of the image
			height (int): the height of the image
			cell_dict (dict): contains keys for each surounding pixel
		Returns:
			(dict(str: int)): returns dictionary with that includes list
								off averages and corresponding centers
		"""
		avgs = np.zeros((height, width))

		for row_index, row in enumerate(image):
			for col_index, pixel in enumerate(row):
				
				cell = self.build_cell(image, row_index, col_index, cell_dict)
				
				total_sum = cell['left_up'] + cell['left_down'] + cell['left'] + \
					  cell['right_up'] + cell['right_down'] + cell['right'] + \
					  cell['top'] + cell['bottom']

				num_neighbors = cell['neighbors']

				avg = total_sum / num_neighbors

				round_avg = int(round(avg))
				
				avgs[row_index][col_index] = round_avg

		cell_dict['avg_list'] = avgs

		return(cell_dict)


	def oncenter(self, average, center):
		"""apply oncenter filter to pixel
		Args:
			average (int): average value of surrounding pixels
			center (int): value of center pixel
		Returns:
			(spike value(int)): returns average - center
		"""
		return(center - average)


	def offcenter(self, average, center):
		"""apply offcenter filter to pixel
		Args:
            average (int): average value of surrounding pixels
            center (int): value of center pixel
		Returns:
			(spike value(int)): returns center - average
		"""
		return(average - center)


	def sobel(self, image, x, y):
		"""apply sobel edge detection filter to pixel

		   I thought this filter would be interesting because it is nearly an
		   overly of an oncenter filtered image on top of an offcenter filtered image.
		   This can show oncenter and offcenter spiketimes on the same image.
		   
		Args:
            image (np.arr): image to filter
            x (int): x position in image
            y (int): y position in image
		Returns:
			(mag(int)): filtered pixel magnitude
		"""
		sobel_x = np.c_[
		    [-1,0,1],
		    [-2,0,2],
		    [-1,0,1]
		]
		sobel_y = np.c_[
		    [1,2,1],
		    [0,0,0],
		    [-1,-2,-1]
		]

		xmag = 0
		ymag = 0
		try:
			for a in range(3):
				for b in range(3):
					xn = x + a - 1
					yn = y + b - 1

					xmag += image[xn][yn] * sobel_x[a][b]
					ymag += image[xn][yn] * sobel_y[a][b]

			mag = (xmag**2 + ymag**2)**.5
		except IndexError:
			mag = 0

		return(mag)


	def run(self, cell_dict, image, width, height):
		"""Generate list of pixel_vals
		Args:
			cell_dict (dict): contains keys for each surounding pixel
			image (np.arr): the image, width x height
			width (int): the width of the image
			height (int): the height of the image
		Returns:
			(pixel_vals(list: int)): returns list of pixel_vals for every pixel in the image
		"""

		pixel_vals = np.zeros((height, width))
		cell_dict = self.calculate_avgs(image, width, height, cell_dict)

		# treat data different for sobel filter
		if self.filter_type == 'sobel':
			for row_index, row in enumerate(image):
				for col_index, pixel in enumerate(row):
					pixel_vals[row_index][col_index] = self.sobel(image, row_index, col_index)
			return(pixel_vals)

		for row_index, row in enumerate(image):
			for col_index, pixel in enumerate(row):
				if self.filter_type == 'oncenter':
					spike = self.oncenter(cell_dict['avg_list'][row_index][col_index], image[row_index][col_index])
				elif self.filter_type == 'offcenter':
					spike = self.offcenter(cell_dict['avg_list'][row_index][col_index], image[row_index][col_index])

				pixel_vals[row_index][col_index] = spike

		return(pixel_vals)