##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to implement 
"""
import os
import numpy as np
import random
import time
import math
import copy
import fnmatch
import shutil
import plotly
import collections
import plotly.plotly as py
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import cmocean as cmo
import plotly.graph_objs as go
from copy import deepcopy
from pylab import rcParams
from PIL import Image
from io import StringIO
from cycler import cycler
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats
from scipy import special
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class BayesLands():
	def __init__(self, muted, simtime, samples, real_elev , real_erdp, real_erdp_pts, erdp_coords, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb, likl_sed):
		self.filename = filename
		self.input = xmlinput
		self.real_elev = real_elev
		self.real_erdp = real_erdp
		
		self.real_erdp_pts = real_erdp_pts
		self.erdp_coords = erdp_coords
		self.likl_sed = likl_sed

		self.simtime = simtime
		self.samples = samples
		self.run_nb = run_nb
		self.muted = muted
		self.erodlimits = erodlimits
		self.rainlimits = rainlimits
		self.mlimit = mlimit
		self.nlimit = nlimit

		self.initial_erod = []
		self.initial_rain = []
		self.initial_m = []
		self.initial_n = []

		self.step_rain = (rainlimits[1]- rainlimits[0])*0.01
		self.step_erod = (erodlimits[1] - erodlimits[0])*0.01
		self.step_m = (mlimit[1] - mlimit[0])*0.01
		self.step_n = (nlimit[1] - nlimit[0])*0.01

		self.sim_interval = np.arange(0, self.simtime+1, 5000)
		self.burn_in = 0.0

	def blackBox(self, rain, erodibility, m , n):
		"""
		Main entry point for running badlands model with different forcing conditions.
		The following forcing conditions can be used:
			- different uniform rain (uniform meaning same precipitation value on the entire region)
			- different uniform erodibility (uniform meaning same erodibility value on the entire region)
		Parameters
		----------
		variable : inputname
			XML file defining the parameters used to run Badlands simulation.
		variable: rain
			Requested uniform precipitation value.
		variable: erodibility
			Requested uniform erodibility value.
		variable: etime
			Duration of the experiment.
		Return
		------
		The function returns 2D numpy arrays containing the following information:
		variable: elev
			Elevation as a 2D numpy array (regularly spaced dataset with resolution equivalent to simulation one)
		variable: erdp
			Cumulative erosion/deposition accumulation as a 2D numpy array (regularly spaced as well)
		"""
		tstart = time.clock()
		# Re-initialise badlands model
		model = badlandsModel()

		# Load the XmL input file
		model.load_xml(str(self.run_nb), self.input, muted = self.muted)

		# Adjust erodibility based on given parameter
		model.input.SPLero = erodibility
		model.flow.erodibility.fill(erodibility)

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = rain

		#Adjust m and n values
		model.input.SPLm = m
		model.input.SPLn = n

		elev_vec = collections.OrderedDict()
		erdp_vec = collections.OrderedDict()
		erdp_pts_vec = collections.OrderedDict()
		
		for x in range(len(self.sim_interval)):
			
			self.simtime = self.sim_interval[x]

			model.run_to_time(self.simtime, muted = self.muted)
			
			elev, erdp = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
			
			erdp_pts = np.zeros((self.erdp_coords.shape[0]))

			for count, val in enumerate(self.erdp_coords):
				erdp_pts[count] = erdp[val[0], val[1]]

			elev_vec[self.simtime] = elev
			erdp_vec[self.simtime] = erdp
			erdp_pts_vec[self.simtime] = erdp_pts
			
			# print 'Badlands black box model took (s):',time.clock()-tstart

		return elev_vec, erdp_vec, erdp_pts_vec

	def interpolateArray(self, coords=None, z=None, dz=None):
		"""
		Interpolate the irregular spaced dataset from badlands on a regular grid.
		"""
		x, y = np.hsplit(coords, 2)
		dx = (x[1]-x[0])[0]

		nx = int((x.max() - x.min())/dx+1)
		ny = int((y.max() - y.min())/dx+1)
		xi = np.linspace(x.min(), x.max(), nx)
		yi = np.linspace(y.min(), y.max(), ny)

		xi, yi = np.meshgrid(xi, yi)
		xyi = np.dstack([xi.flatten(), yi.flatten()])[0]
		XY = np.column_stack((x,y))

		tree = cKDTree(XY)
		distances, indices = tree.query(xyi, k=3)
		if len(z[indices].shape) == 3:
			z_vals = z[indices][:,:,0]
			dz_vals = dz[indices][:,:,0]
		else:
			z_vals = z[indices]
			dz_vals = dz[indices]

		zi = np.average(z_vals,weights=(1./distances), axis=1)
		dzi = np.average(dz_vals,weights=(1./distances), axis=1)
		onIDs = np.where(distances[:,0] == 0)[0]
		if len(onIDs) > 0:
			zi[onIDs] = z[indices[onIDs,0]]
			dzi[onIDs] = dz[indices[onIDs,0]]
		zreg = np.reshape(zi,(ny,nx))
		dzreg = np.reshape(dzi,(ny,nx))
		return zreg,dzreg

	def plotFunctions(self, fname, pos_likl, pos_rain, pos_erod):
		nb_bins=30
		font = 9
		width = 1

		fig = plt.figure(figsize=(15,15))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		# ax.set_title(' Likelihood', fontsize=  font+2)#, y=1.02)
		
		ax1 = fig.add_subplot(211)
		ax1.set_facecolor('#f2f2f3')
		X = pos_rain
		Y = pos_likl
		print 'Y', Y

		print 'X shape ', X.shape, 'Y shape ', Y.shape

		ax1.plot(X,Y)
		ax1.set_ylim(Y.min(), Y.max()*1.10) #(Y.max()+(Y.max()*0.2)))
		ax1.yaxis.set_major_locator(LinearLocator(10))
		ax1.yaxis.set_major_formatter(FormatStrFormatter('%.05f'))
		ax1.set_title(' Log Likelihood', fontsize=  font+2)#, y=1.02)
		# Add a color bar which maps values to colors.

		ax2 = fig.add_subplot(212)
		ax2.set_facecolor('#f2f2f3')
		
		X = pos_rain

		max_pos_likl = pos_likl.max()
		print 'max_pos_likl', max_pos_likl
		pos_likl = pos_likl - (max_pos_likl)
		print 'pos_likl', pos_likl
		EY = np.exp(pos_likl)
		print 'EY', EY
		ax2.plot(X,EY)
		ax2.set_ylim(EY.min(), EY.max()*1.10) #(Y.max()+(Y.max()*0.2)))
		ax2.yaxis.set_major_locator(LinearLocator(10))
		ax2.yaxis.set_major_formatter(FormatStrFormatter('%.05f'))
		ax2.set_title('Likelihood' , fontsize= font +2)

		plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
		plt.show()

	def storeParams(self, naccept, pos_rain, pos_erod, pos_likl):
		"""
		
		"""
		pos_rain = str(pos_rain)
		if not os.path.isfile(('%s/accept_rain.txt' % (self.filename))):
			with file(('%s/accept_rain.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_rain)
		else:
			with file(('%s/accept_rain.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rain)

		pos_erod = str(pos_erod)
		if not os.path.isfile(('%s/accept_erod.txt' % (self.filename))):
			with file(('%s/accept_erod.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_erod)
		else:
			with file(('%s/accept_erod.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_erod)

		pos_likl = str(pos_likl)
		if not os.path.isfile(('%s/accept_likl.txt' % (self.filename))):
			with file(('%s/accept_likl.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)
		else:
			with file(('%s/accept_likl.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)

	def likelihoodFunc(self,input_vector, real_elev, real_erdp, real_erdp_pts, tausq_elev, tausq_erdp, tausq_erdp_pts):
		"""
		
		"""
		pred_elev_vec, pred_erdp_vec, pred_erdp_pts_vec = self.blackBox(input_vector[0], input_vector[1], input_vector[2], input_vector[3])
		
		print 'real_elev size',  real_elev.size


		tausq_elev = (np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))/real_elev.size

		print 'tausq_elev', tausq_elev

		
		######### DEGREE OF FREEDOM = 
		# likelihood_elev = -0.5 * np.log(2* math.pi * tausq_elev) - 0.5 * np.square(pred_elev_vec[self.simtime] - real_elev) / tausq_elev
		
		######### DEGREE OF FREEEDOM = , Essentially just a squared error loss
		likelihood_elev = - np.square(pred_elev_vec[self.simtime] - real_elev)

		print 'Likelihood elev' , likelihood_elev
		# size_by_two = real_elev.size/2

		# print 'real_elev.size()', size_by_two
		# print 'first component', special.gammaln(size_by_two)
		# # print 'second component', size_by_two*np.log(0.5*np.square(pred_elev_vec[self.simtime] - real_elev))
		# print 'sum', 0.5*np.sum(np.square(pred_elev_vec[self.simtime] - real_elev))
		# print 'log of sum ', np.log(0.5*np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))
		# print 'n/2 * sum', size_by_two* np.log(0.5*np.sum(np.square(pred_elev_vec[self.simtime] - real_elev))), '\n'

		# likelihood_elev = special.gammaln(size_by_two) - size_by_two*np.log(0.5*np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))

		# likelihood = likelihood + 34390 #RAIN variant
		# likelihood = likelihood + 34390

		# print 'Likelihood without adjustment', likelihood_elev
		likelihood = np.sum(likelihood_elev)

		print 'Likelihood elev' , likelihood
		
		# print 'Likelihood with adjustment', likelihood

		return likelihood

	def likelihoodSurface(self):
		
		# Initializing variables
		samples = self.samples

		real_elev = self.real_elev
		real_erdp = self.real_erdp
		real_erdp_pts = self.real_erdp_pts

		# Creating storage for data
		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)
		
		# List of accepted samples
		count_list = []

		# print 'rain dimension', int(math.sqrt(samples))

		rain = np.linspace(self.rainlimits[0], self.rainlimits[1], num = int(samples))
		erod = np.linspace(self.erodlimits[0], self.erodlimits[1], num = int(samples))

		# erod = erod.fill(5.e-5)
		# print 'erod', erod

		dimx = rain.shape[0]
		# dimy = erod.shape[0]

		pos_likl = np.zeros(dimx)
		# print 'pos_likl', pos_likl.shape, 'pos_rain', pos_rain, 'pos_erod', pos_erod

		m = 0.5
		n = 1.0

		tau_elev = 1500 
		tau_erdp = 1500
		tau_erdp_pts = 1000

		# Creating storage for parameters to be passed to blackBox model 
		v_proposal = []
		v_proposal.append(rain[0])
		v_proposal.append(erod[0])
		v_proposal.append(m)
		v_proposal.append(n)

		# Storing RMSE, tau values and adding initial run to accepted list
		start = time.time()

		i = 0
		counter = 0
		for r in range(len(rain)):
			print '\nr,', r,'\n'
			
			# Updating rain parameter and checking limits
			p_rain = rain[r]
			
			# Updating edodibility parameter and checking limits
			p_erod = 5.e-5

			p_m = m
			p_n = n

			# Creating storage for parameters to be passed to blackBox model
			v_proposal = []
			v_proposal.append(p_rain)
			v_proposal.append(p_erod)
			v_proposal.append(p_m)
			v_proposal.append(p_n)


			# Passing paramters to calculate likelihood and rmse with new tau
			likelihood = self.likelihoodFunc(v_proposal,real_elev, real_erdp, real_erdp_pts, tau_elev, tau_erdp, tau_erdp_pts)
			
			
			pos_erod[i] = p_erod
			pos_rain[i] = p_rain
			# print 'likelihood',likelihood
			# print 'exp(likl)', np.exp(likelihood)
			pos_likl[r] = likelihood
			# pos_likl[e] = likelihood
			
			self.storeParams(i, pos_rain[i], pos_erod[i], pos_likl[r]) 

			i += 1
			counter +1

		self.plotFunctions(self.filename, pos_likl, rain, erod)
		# print 'max of Likelihood', pos_likl.max()
		# print 'min of Likelihood', pos_likl.min()
		end = time.time()
		total_time = end - start
		print 'counter', counter, '\nTime elapsed:', total_time, '\npos_likl.shape', pos_likl.shape
		
		return (pos_rain, pos_erod, pos_likl)

def main():

	random.seed(time.time())
	muted = True
	run_nb = 0
	directory = ""
	likl_sed = False
	
	erdp_coords_crater = np.array([ [2,3], [3,2], [5,5] ])
	erdp_coords_etopo = np.array([ [10,60], [30,30], [60,10], [80,75] ])
	
	choice = input("Please choose a Badlands example to run the likelihood surface generator on:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")
	samples = input("Please enter number of samples (Make sure it is a perfect square): ")

	if choice == 1:
		directory = 'Examples/crater_fast_3030'
		xmlinput = '%s/crater.xml' %(directory)
		simtime = 15000
		rainlimits = [1.0, 2.0]
		# rainlimits = [1.495, 1.505]
		erodlimits = [4.e-5, 6.e-5]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		true_rain = 1.5
		true_erod = 5.e-5
		likl_sed = False
		erdp_coords = erdp_coords_crater

	elif choice == 2:
		directory = 'Examples/crater'
		xmlinput = '%s/crater.xml' %(directory)
		simtime = 50000
		rainlimits = [0.5, 3.0]
		erodlimits = [1.e-5, 9.e-5]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		true_rain = 1.5
		true_erod = 5.e-5
		likl_sed = True
		erdp_coords = erdp_coords_crater

	elif choice == 3:
		directory = 'Examples/etopo_fast'
		xmlinput = '%s/etopo.xml' %(directory)
		simtime = 500000
		rainlimits = [0.5, 3.0]
		erodlimits = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		true_rain = 1.5
		true_erod = 5.e-6
		likl_sed = True
		erdp_coords = erdp_coords_etopo

	elif choice == 4:
		directory = 'Examples/etopo'
		xmlinput = '%s/etopo.xml' %(directory)
		simtime = 500000
		rainlimits = [0.5, 3.0]
		erodlimits = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		true_rain = 1.5
		true_erod = 5.e-6
		likl_sed = True
		erdp_coords = erdp_coords_etopo

	else:
		print('Invalid selection, please choose a problem from the list ')

	final_elev = np.loadtxt('%s/data/final_elev.txt' %(directory))
	final_erdp = np.loadtxt('%s/data/final_erdp.txt' %(directory))
	final_erdp_pts = np.loadtxt('%s/data/final_erdp_pts.txt' %(directory))	

	while os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
		run_nb+=1
	if not os.path.exists('%s/liklSurface_%s' % (directory,run_nb)):
		os.makedirs('%s/liklSurface_%s' % (directory,run_nb))
		# os.makedirs('%s/liklSurface_%s/plots' % (directory,run_nb))
		# os.makedirs('%s/liklSurface_%s/prediction_data' % (directory,run_nb))
		filename = ('%s/liklSurface_%s' % (directory,run_nb))

	print '\nInput file shape', final_elev.shape, '\n'
	run_nb_str = 'liklSurface_' + str(run_nb)

	bLands = BayesLands(muted, simtime, samples, final_elev, final_erdp, final_erdp_pts, erdp_coords, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb_str, likl_sed)
	[pos_rain, pos_erod, pos_likl] = bLands.likelihoodSurface()

	print 'Finished producing Likelihood Surface'

if __name__ == "__main__": main()