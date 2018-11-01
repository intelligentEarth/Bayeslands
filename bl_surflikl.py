##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##

#Main Contributer: Danial Azam  Email: dazam92@gmail.com

"""
This script is intended to implement functionality to  generate the likelihood surface of the free parameters.

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
from sklearn.preprocessing import normalize
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
	def __init__(self, muted, simtime, samples, real_elev , real_erdp, real_erdp_pts, erdp_coords, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, marinelimit, aeriallimit, run_nb, likl_sed):
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
		self.marinelimit = marinelimit
		self.aeriallimit = aeriallimit

		self.initial_erod = []
		self.initial_rain = []
		self.initial_m = []
		self.initial_n = []

		self.step_rain = (rainlimits[1]- rainlimits[0])*0.01
		self.step_erod = (erodlimits[1] - erodlimits[0])*0.01
		self.step_m = (mlimit[1] - mlimit[0])*0.01
		self.step_n = (nlimit[1] - nlimit[0])*0.01

		self.sim_interval = np.arange(0, self.simtime+1, self.simtime/4)
		self.burn_in = 0.0

	def blackBox(self, rain, erodibility, m , n, marinediff, aerialdiff):
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

		model.input.CDm = marinediff
		model.input.CDa = aerialdiff

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

	def viewGrid(self, plot_name ,fname, Z, rain, erod, width = 1000, height = 1000, zmin = None, zmax = None, zData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the grid in 3D.

		Parameters
		----------
		variable : resolution
		    Required resolution for the model grid (in metres).
		variable: width
		    Figure width.
		variable: height
		    Figure height.
		variable: zmin
		    Minimal elevation.
		variable: zmax
		    Maximal elevation.
		variable: height
		    Figure height.
		variable: zData
		    Elevation data to plot.
		variable: title
		    Title of the graph.
		"""

		zData = Z

		if zmin == None:
			zmin = zData.min()

		if zmax == None:
			zmax = zData.max()

		data = Data([ Surface( x=rain, y=erod, z=zData ) ])

		layout = Layout(
			title=plot_name,
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				zaxis=ZAxis(title = 'Log Likelihood',range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(title = 'Rain (m/a)',nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(title = 'Erodibility',nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)

		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/elev_grid_%s.html' %(fname, plot_name), validate=False)
		return

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
		ax.set_title(' Likelihood', fontsize=  font+2)#, y=1.02)
		
		ax1 = fig.add_subplot(211, projection = '3d')
		ax1.set_facecolor('#f2f2f3')
		X = pos_rain
		Y = pos_erod

		R = X/Y

		X, Y = np.meshgrid(X, Y)
		Z = pos_likl

		print 'X shape ', X.shape, 'Y shape ', Y.shape, 'Z shape ', Z.shape

		surf = ax1.plot_surface(X,Y,Z, cmap = cm.coolwarm, linewidth= 0, antialiased = False)
		ax1.set_zlim(Z.min(), Z.max())
		ax1.zaxis.set_major_locator(LinearLocator(10))
		ax1.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
		# Add a color bar which maps values to colors.

		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
		plt.show()

	def storeParams(self, naccept, pos_rain, pos_erod, pos_m, pos_n, pos_marinediff, pos_aerialdiff, pos_likl):
		"""
		
		"""
		pos_likl = str(pos_likl)
		pos_rain = str(pos_rain)
		pos_erod = str(pos_erod)
		pos_m = str(pos_m)
		pos_n = str(pos_n)
		pos_marinediff = str(pos_marinediff) 
		pos_aerialdiff = str(pos_aerialdiff) 

		if not os.path.isfile(('%s/exp_data.txt' % (self.filename))):
			with file(('%s/exp_data.txt' % (self.filename)),'w') as outfile:
				# outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rain)
				outfile.write('\t')
				outfile.write(pos_erod)
				outfile.write('\t')
				outfile.write(pos_m)
				outfile.write('\t')
				outfile.write(pos_n)
				outfile.write('\t')
				outfile.write(pos_marinediff)
				outfile.write('\t')
				outfile.write(pos_aerialdiff)
				outfile.write('\t')
				
				outfile.write(pos_likl)
				# outfile.write('\t')
				# outfile.write(sq_error)
				# outfile.write('\t')
				# outfile.write(tausq_elev)
				# outfile.write('\t')
				# outfile.write(tausq_erdp_pts)
				outfile.write('\n')
		else:
			with file(('%s/exp_data.txt' % (self.filename)),'a') as outfile:
				# outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rain)
				outfile.write('\t')
				outfile.write(pos_erod)
				outfile.write('\t')
				outfile.write(pos_m)
				outfile.write('\t')
				outfile.write(pos_n)
				outfile.write('\t')
				outfile.write(pos_marinediff)
				outfile.write('\t')
				outfile.write(pos_aerialdiff)
				outfile.write('\t')
	
				outfile.write(pos_likl)				
				# outfile.write('\t')
				# outfile.write(sq_error)
				# outfile.write('\t')
				# outfile.write(tausq_elev)
				# outfile.write('\t')
				# outfile.write(tausq_erdp_pts)
				outfile.write('\n')
				  
	def likelihoodFunc(self,input_vector, real_elev, real_erdp, real_erdp_pts):
		"""
		
		"""
		pred_elev_vec, pred_erdp_vec, pred_erdp_pts_vec = self.blackBox(input_vector[0], input_vector[1], input_vector[2], input_vector[3], input_vector[4], input_vector[5])

		tausq_elev = (np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))/real_elev.size
		sq_error_elev = (np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))/real_elev.size

		tausq_erdp_pts = np.zeros(self.sim_interval.size)
		for i in range(self.sim_interval.size):
			tausq_erdp_pts[i] = np.sum(np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]))/real_erdp_pts.shape[1]
		
		# print 'tausq_erdp_pts' , tausq_erdp_pts

		likelihood_elev = -0.5 * np.log(2* math.pi * tausq_elev) - 0.5 * np.square(pred_elev_vec[self.simtime] - real_elev) / tausq_elev
		likelihood_erdp_pts = 0

		if self.likl_sed:
			#likelihood_erdp  = -0.5 * np.log(2* math.pi * tausq_erdp) - 0.5 * np.square(pred_erdp_vec[self.simtime] - real_erdp) / tausq_erdp		
			for i in range(1,self.sim_interval.size):
				likelihood_erdp_pts += np.sum(-0.5 * np.log(2* math.pi * tausq_erdp_pts[i]) - 0.5 * np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]) / tausq_erdp_pts[i])
			
			likelihood = np.sum(likelihood_elev) + (likelihood_erdp_pts)

			sq_error_erdp_pts = np.sum(np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]))/real_erdp_pts.shape[1]
			sq_error = sq_error_elev+ sq_error_erdp_pts
			print 'Using sediment pts in the likelihood'
		
		else:
			likelihood = np.sum(likelihood_elev)
			sq_error = sq_error_elev

		return likelihood, sq_error, tausq_elev, tausq_erdp_pts

	def likelihoodSurface(self):
		
		# Initializing variables
		samples = self.samples

		real_elev = self.real_elev
		real_erdp = self.real_erdp
		real_erdp_pts = self.real_erdp_pts

		# Creating storage for data
		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)
		pos_m = np.zeros(samples)
		pos_n = np.zeros(samples)
		pos_marinediff = np.zeros(samples)
		pos_aerialdiff = np.zeros(samples)
		# List of accepted samples
		count_list = []

		rain = np.linspace(self.rainlimits[0], self.rainlimits[1], num = int(math.sqrt(samples)), endpoint = False)
		erod = np.linspace(self.erodlimits[0], self.erodlimits[1], num = int(math.sqrt(samples)), endpoint = False)

		dimx = rain.shape[0]
		dimy = erod.shape[0]

		pos_likl = np.zeros((dimx, dimy))
		pos_sq_error = np.zeros((dimx, dimy))
		# print 'pos_likl', pos_likl.shape, 'pos_rain', pos_rain, 'pos_erod', pos_erod

		# Storing RMSE, tau values and adding initial run to accepted list
		start = time.time()

		i = 0
		
		for r in range(len(rain)):
			for e in range(len(erod)):
				print '\n'
				print 'Rain : ', rain[r], '  Erod : ', erod[e]
				print 'Simtime', self.simtime
				
				# Updating rain parameter and checking limits
				p_rain = rain[r]
				
				# Updating edodibility parameter and checking limits
				p_erod = erod[e]

				p_m = np.random.normal(0.5, 0.05)
				p_n = np.random.normal(1.0, 0.05)
				p_marinediff = np.random.normal(np.mean(self.marinelimit), np.std(self.marinelimit)/2)
				p_aerialdiff = np.random.normal(np.mean(self.aeriallimit), np.std(self.aeriallimit)/2)

				# Creating storage for parameters to be passed to blackBox model
				v_proposal = []
				v_proposal.append(p_rain)
				v_proposal.append(p_erod)
				v_proposal.append(p_m)
				v_proposal.append(p_n)
				v_proposal.append(p_marinediff)
				v_proposal.append(p_aerialdiff)

				# Passing paramters to calculate likelihood and rmse with new tau
				likelihood, sq_error, tau_elev, tau_erdp_pts = self.likelihoodFunc(v_proposal,real_elev, real_erdp, real_erdp_pts)
				# print 'sq_error', sq_error
				pos_erod[i] = p_erod
				pos_rain[i] = p_rain
				pos_m[i] = p_m
				pos_n[i] = p_n
				pos_marinediff[i] = p_marinediff
				pos_aerialdiff[i] = p_aerialdiff

				pos_likl[r,e] = likelihood
				pos_sq_error[r,e] = sq_error
				self.storeParams(i, pos_rain[i], pos_erod[i], pos_m[i], pos_n[i], pos_marinediff[i], pos_aerialdiff[i], pos_likl[r,e])

				i += 1

		# self.plotFunctions(self.filename, pos_likl, rain, erod)
		self.viewGrid('Log_likelihood ',self.filename, pos_likl, rain, erod)
		self.viewGrid('Sum Squared Error',self.filename, pos_sq_error, rain, erod)
		end = time.time()
		total_time = end - start
		print 'counter', i, '\nTime elapsed:', total_time, '\npos_likl.shape', pos_likl.shape
		
		return (pos_rain, pos_erod, pos_likl)

def main():

	random.seed(time.time())
	muted = True
	run_nb = 0
	directory = ""
	likl_sed = False
	
	erdp_coords_crater = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
	erdp_coords_crater_fast = np.array([[60,60],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69],[79,91],[96,77],[42,49]])
	erdp_coords_etopo = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])
	erdp_coords_etopo_fast = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[68,40],[72,44]])

	choice = input("Please choose a Badlands example to run the likelihood surface generator on:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")
	samples = input("Please enter number of samples (Make sure it is a perfect square): \n")

	if choice == 1:
		directory = 'Examples/crater_fast'
		xmlinput = '%s/crater.xml' %(directory)
		simtime = 15000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-5, 7.e-5]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		marinelimit = [5.e-3,4.e-2]
		aeriallimit = [3.e-2,7.e-2]
		true_rain = 1.5
		true_erod = 5.e-5
		likl_sed = True
		erdp_coords = erdp_coords_crater_fast

	elif choice == 2:
		directory = 'Examples/crater'
		xmlinput = '%s/crater.xml' %(directory)
		simtime = 50000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-5, 7.e-5]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		marinelimit = [5.e-3,4.e-2]
		aeriallimit = [3.e-2,7.e-2]
		true_rain = 1.5
		true_erod = 5.e-5
		likl_sed = True
		erdp_coords = erdp_coords_crater

	elif choice == 3:
		directory = 'Examples/etopo_fast'
		xmlinput = '%s/etopo.xml' %(directory)
		simtime = 500000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		marinelimit = [0.3,0.7]
		aeriallimit = [0.6,1.0]
		true_rain = 1.5
		true_erod = 5.e-6
		likl_sed = True
		erdp_coords = erdp_coords_etopo_fast

	elif choice == 4:
		directory = 'Examples/etopo'
		xmlinput = '%s/etopo.xml' %(directory)
		simtime = 1000000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		marinelimit = [0.3,0.7]
		aeriallimit = [0.6,1.0]
		true_rain = 1.5
		true_erod = 5.e-6
		likl_sed = True
		erdp_coords = erdp_coords_etopo

	elif choice == 5:
		directory = 'Examples/tasmania'
		xmlinput = '%s/tasmania.xml' %(directory)
		simtime = 1000000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-6, 7.e-6]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
		marinelimit = [0.3,0.7]
		aeriallimit = [0.6,1.0]
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
		os.makedirs('%s/liklSurface_%s/plots' % (directory,run_nb))
		os.makedirs('%s/liklSurface_%s/prediction_data' % (directory,run_nb))
		filename = ('%s/liklSurface_%s' % (directory,run_nb))


	with file(('%s/liklSurface_%s/description.txt' % (directory,run_nb)),'a') as outfile:
			outfile.write('\n\samples: {0}'.format(samples))
			outfile.write('\n\terod_limits: {0}'.format(erodlimits))
			outfile.write('\n\train_limits: {0}'.format(rainlimits))
			outfile.write('\n\terdp coords: {0}'.format(erdp_coords))
			outfile.write('\n\tlikl_sed: {0}'.format(likl_sed))
			outfile.write('\n\tfilename: {0}'.format(filename))

	print '\nInput file shape', final_elev.shape, '\n'
	run_nb_str = 'liklSurface_' + str(run_nb)

	bLands = BayesLands(muted, simtime, samples, final_elev, final_erdp, final_erdp_pts, erdp_coords, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, marinelimit, aeriallimit, run_nb_str, likl_sed)
	[pos_rain, pos_erod, pos_likl] = bLands.likelihoodSurface()

	print 'Results are stored in ', filename

	print 'Finished producing Likelihood Surface'

if __name__ == "__main__": main()