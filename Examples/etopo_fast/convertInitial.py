
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to 
"""

from pyBadlands.model import Model as badlandsModel
import numpy as np
import cmocean as cmo
from pylab import rcParams
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import time
import itertools


def convertInitial(fname, res_fact, reduce_factor):
	
	arr = np.loadtxt(fname)
	with open('data/convertedInitial_low.csv', 'a') as the_file:
		for i in range(0, arr.shape[0]-2, reduce_factor):
			for j in range(0, arr.shape[1]-2, reduce_factor):
				x_c = i*res_fact
				y_c = j*res_fact

				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.2f}".format(arr[i,j]))) +  '\n'
				the_file.write(line)


	#np.savetxt('data/convert_res_crater.csv', , fmt='%.5f')

def reduceAmplitude(fname, res_fact):
	arr = np.loadtxt(fname)
	amp_percentage = 1
	# new_arr = np.zeros((arr.shape[0],arr.shape[1]))
	with open('data/res_crater_reduced_ampl.csv', 'a') as the_file:
		for i in range(arr.shape[0]-2):
			for j in range(arr.shape[1]-2):
				x_c = i*res_fact
				y_c = j*res_fact

				# if arr[i,j]<=100:
				# 	arr[i-j] = arr[i-j]
				# elif arr[i,j]>=100 and arr[i,j]<=200:
				# 	arr[i,j] = arr[i,j]*0.8
				# elif arr[i,j]>=200 and arr[i,j]<=300:
				# 	arr[i,j] = arr[i,j]*0.7
				# elif arr[i,j]>=300 and arr[i,j]<=400:
				# 	arr[i,j] = arr[i,j]*0.6
				# elif arr[i,j]>=400 and arr[i,j]<=500:
				# 	arr[i,j] = arr[i,j]*0.5
				# elif arr[i,j]>=500:
				# 	arr[i,j] = arr[i,j]*0.4
				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.3f}".format(arr[i,j]*amp_percentage))) +  '\n'
				the_file.write(line)


def main():

	# convertInitial('data/initial_elev.txt', res_fact = 20, reduce_factor = 1)
	reduceAmplitude('data/final_elev.txt', res_fact=20)
	print 'Finished'

if __name__ == "__main__": main()
