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


def res_generator(fname, x, y, size, res_fact= None, max_coord= None):

	arr = np.loadtxt(fname)

	split_num = max_coord/res_fact
	
	split_arr = np.vsplit(arr,split_num+1)

	newarr = []

	y_ind = np.arange(y, y+size+1)

	print y_ind
	print len(split_arr)

	for i in y_ind:
		p = split_arr[i]
		newarr.append(p[x:size+1])

	# for num, p in enumerate(split_arr):
	# 	newarr.append(p[x:size+1])
	# 	if num == y:
	# 		break

	X = list(itertools.chain.from_iterable(newarr))

	Z = np.vstack(X) 

	print Z

	np.savetxt('test.csv', Z, fmt='%.5f')

def main():

	res_generator('data/res_crater.csv', 100, 100, 100, 10, 2400)
	
	print 'Finished'

if __name__ == "__main__": main()