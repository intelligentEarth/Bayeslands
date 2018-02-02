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


def inputGenerator(fname, x, y, size, res_fact= None, max_coord= None):

	x = x/res_fact
	y = y/res_fact
	arr = np.loadtxt(fname)

	split_num = max_coord/res_fact
	
	split_arr = np.vsplit(arr,split_num+1)

	newarr = []

	y_ind = np.arange(y, y+size+1)

	print y_ind
	print len(split_arr)

	for i in y_ind:
		p = split_arr[i]
		print 'p ', p
		newarr.append(p[x:size+1])

	X = list(itertools.chain.from_iterable(newarr))

	Z = np.vstack(X) 

	print Z

	np.savetxt('data/res_crater.csv', Z, fmt='%.5f')

def main():

	inputGenerator('data/res_crater_original.csv', 400, 250, 50, 10, 2400)
	
	print 'Finished'

if __name__ == "__main__": main()
