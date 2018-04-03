
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
import time
import itertools
import numpy as np
import cmocean as cmo
import matplotlib.pyplot as plt
# import pybadlands_companion.resizeInput as resize
from pylab import rcParams
from pyBadlands.model import Model as badlandsModel
from scipy.spatial import cKDTree
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

def convertInitialTXT_CSV(directory,fname, res_fact, reduce_factor):
	
	arr = np.loadtxt(fname)
	with open('%s/data/convertedInitial_low.csv' %(directory), 'a') as the_file:
		for i in xrange(0, arr.shape[0]-2, reduce_factor):
			for j in xrange(0, arr.shape[1]-2, reduce_factor):
				x_c = i*res_fact
				y_c = j*res_fact

				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.2f}".format(arr[i,j]))) +  '\n'
				the_file.write(line)

	#np.savetxt('data/convert_res_crater.csv', , fmt='%.5f')

def cropTopoCSV(directory,fname, x, y, size, res_fact, max_coord= None):

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
	np.savetxt('%s/data/res_crater.csv' (directory), Z, fmt='%.5f')

def reduceAmplitude(directory,fname, res_fact, amp_percentage):
	arr = np.loadtxt(fname)
	# new_arr = np.zeros((arr.shape[0],arr.shape[1]))
	with open('%s/data/res_crater_reduced_ampl.csv' %(directory), 'a') as the_file:
		for i in range(arr.shape[0]-2):
			for j in range(arr.shape[1]-2):
				x_c = i*res_fact
				y_c = j*res_fact
				line = str(float(y_c)) + ' ' + str(float(x_c))+ ' ' + str(float("{0:.3f}".format(arr[i,j]*amp_percentage))) +  '\n'
				the_file.write(line)

def upScale(directory,fname, res_fact):
	# Load python class and set required resolution
	newRes = resize.resizeInput(requestedSpacing = res_fact)

	# Regrid DEM file
	newRes.regridDEM(inDEM=fname,outDEM='newnodes.csv')

	# Regrid Rain file
	# newRes.regridRain(inRain='data/rain.csv',outRain='newrain.csv')
	
	# Regrid Tectonic files (Vertical only file)
	# newRes.regridTecto(inTec='data/disp.csv', outTec='newdisp.csv')
	
	# Regrid Tectonic files (3D displacement file)
	# newRes.regridDisp(inDisp='data/disp.csv', outDisp='newdisp.csv')

def main():

	functionality = input("Would you like to: \n 1) Resuce Amplitude of Initial/Final topo\n 2) Convert topo TXT to CSV w/ reduction\n 3) Crop Topo CSV file\n 4) Upscale Topo")
	choice = input("Please choose a Badlands example to apply it to:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n 5) delta\n")
	directory = ""

	if choice == 1:
		directory = 'Examples/crater_fast_3030'
	elif choice ==2:
		directory = 'Examples/crater'
	elif choice ==3:
		directory = 'Examples/etopo_fast'
	elif choice ==4:
		directory = 'Examples/etopo'
	# elif choice ==5:
	# 	directory = 'Examples/mountain'
	
	if functionality == 1:
			tstart = time.clock()
			amp_percentage = input("What percentage would you like to decrease the amplitude by?")
			res_fact = input("Resolution Factor")
			topo = input("Would you like to apply it to the \n 1) Final topo \n 2) Initial topo")
			if topo == 1:
				reduceAmplitude(directory,'%s/data/initial_elev.txt'%(directory), res_fact=res_fact, amp_percentage= amp_percentage)
			else:
				reduceAmplitude(directory,'%s/data/final_elev.txt'%(directory), res_fact=res_fact, amp_percentage= amp_percentage)				
			print 'Task completed in (s):',time.clock()-tstart	

	elif functionality == 2:
			tstart = time.clock()
			reduce_factor = input("What should the reduction factor be?")
			res_fact = input("Resolution Factor")
			convertInitialTXT_CSV(directory,'%s/data/initial_elev.txt' %(directory), res_fact = res_fact, reduce_factor = reduce_factor)
			print 'Task completed in (s):',time.clock()-tstart

	elif functionality == 3:
			tstart = time.clock()
			x_coord = input("X coord")
			y_coord = input("Y coord ")
			grid_size = input("Size of Grid")
			res_fact = input("Resolution Factor")
			max_size = 2400
			cropTopoCSV(directory,'%s/data/res_crater.csv'%(directory), x_coord, y_coord, grid_size, 10, max_size)
			print 'Task completed in (s):',time.clock()-tstart

	# elif functionality == 4:
	# 		tstart = time.clock()
	# 		res_fact = input("Resolution Factor")
	# 		upScale('%s/data/res_crater.csv'%(directory), res_fact)
	# 		print 'Task completed in (s):',time.clock()-tstart

if __name__ == "__main__": main()