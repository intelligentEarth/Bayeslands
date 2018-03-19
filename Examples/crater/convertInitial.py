
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
	new_arr = np.zeros((arr.shape[0],arr.shape[1]))
	
	with open('data/convertedInitial_low.csv', 'a') as the_file:
		for i in range(0, arr.shape[0]-2, reduce_factor):
			for j in range(0, arr.shape[1]-2, reduce_factor):
				x_c = i*res_fact
				y_c = j*res_fact

				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.6f}".format(arr[i,j]))) +  '\n'

				
				the_file.write(line)


	#np.savetxt('data/convert_res_crater.csv', , fmt='%.5f')

def main():

	convertInitial('data/final.txt', res_fact = 10, reduce_factor = 1 )
	
	print 'Finished'

if __name__ == "__main__": main()
