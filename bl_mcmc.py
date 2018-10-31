##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.    ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##

#Main Contributer: Danial Azam  Email: dazam92@gmail.com

"""
This script is intended to implement an MCMC (Markov Chain Monte Carlo) Metropolis Hastings methodology to pyBadlands. 
Badlands is used as a "black box" model for bayesian methods.
"""
import os
import sys
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
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()

class bayeslands_mcmc():
	"""
		
	"""
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

		self.step_rain = (rainlimits[1]- rainlimits[0])*0.03
		self.step_erod = (erodlimits[1] - erodlimits[0])*0.03
		self.step_m = (mlimit[1] - mlimit[0])*0.01
		self.step_n = (nlimit[1] - nlimit[0])*0.01

		self.sim_interval = np.arange(0, self.simtime+1, self.simtime/4)
		self.burn_in = 0.05

	def blackBox(self, rain, erodibility, m , n):
		"""
		Main entry point for running badlands model with different forcing conditions.
		The following forcing conditions can be used:
			- different uniform rain (uniform meaning same precipitation value on the entire region)
			- different uniform erodibility (uniform meaning same erodibility value on the entire region)
		
		Parameters
		----------
		variable: rain
			Requested uniform precipitation value.
		variable: erodibility
			Requested uniform erodibility value.
		variable: m, n
			Values of m and n indicate how the incision rate scales
            with bed shear stress for constant value of sediment flux
            and sediment transport capacity.
		
		Returns
		------
		variable: elev_vec
			Elevation as a 2D numpy array (regularly spaced dataset with resolution equivalent to simulation one)
		variable: erdp_vec
			Cumulative erosion/deposition accumulation as a 2D numpy array (regularly spaced as well)
		variable: erdp_pts_vec
			Cumulative erosion/deposition at particular co-ordinates on the grid stored in erdp_coords
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
		Parameters
		----------
		variable : coords
			model grid coordinates
		variable: z
			elevation
		variable: dz
			cummulative difference in sediment
		Return
		------
		The function returns 2D numpy arrays containing the following information:
		variable: zreg
			elevation on a regular grid
		variable: dzreg
			erodep on a regular grid
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

	def viewMap(self, sample_num, likl, rain, erod, width = 600, height = 600, zmin = None, zmax = None, zData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the Erosion Deposition Heatmap.
		
		Parameters
		----------
		variable : likl, rain, erod
			values of rain, erodibility and likelihood to display on the map
		variable: width
			Figure width.
		variable: height
			Figure height.
		variable: zmin
			Minimal elevation.
		variable: zmax
			Maximal elevation.
		variable: zData
			Elevation data to plot.
		variable: title
			Title of the graph.
		"""
		if zmin == None:
			zmin = zData.min()
		if zmax == None:
			zmax = zData.max()
		
		trace = go.Heatmap(z=zData)
		data=[trace]
		layout = Layout(
			title='Crater Erosiondeposition     rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				zaxis=ZAxis(range=[zmin,zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)
		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/erdp_heatmap_%s.html' %(self.filename, sample_num), validate=False)
		
		return

	def viewBar(self,sample_num, likl, rain, erod, width = 500, height = 500, xData = None, yData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the BarPlot of Erosion Deposition at certain coordinates.

		Parameters
		----------
		variable : likl, rain, erod
			values of rain, erodibility and likelihood to display on the map
		variable: width
			Figure width.
		variable: height
			Figure height.
		variable: xData, yData
			X, Y data to plot.
		variable: title
			Title of the graph.
		"""
		xData = np.array_str(xData)
		trace = go.Bar(x=xData, y = yData)
		data=[trace]
		layout = Layout(
			title='Crater Erosion deposition pts    rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)

		return

	def viewGrid(self, sample_num, likl, rain, erod, width = 1600, height = 1600, zmin = None, zmax = None, zData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the elevation grid in 3D.

		Parameters
		----------
		variable : likl, rain, erod
			values of rain, erodibility and likelihood to display on the map
		variable: width
		    Figure width.
		variable: height
		    Figure height.
		variable: zmin
		    Minimal elevation.
		variable: zmax
		    Maximal elevation.
		variable: zData
		    Elevation data to plot.
		variable: title
		    Title of the graph.
		"""

		if zmin == None:
			zmin = zData.min()

		if zmax == None:
			zmax = zData.max()

		data = Data([ Surface( x=zData.shape[0], y=zData.shape[1], z=zData, colorscale='YIGnBu' ) ])

		layout = Layout(
			title='Crater Elevation  	rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				zaxis=ZAxis(title = 'Elevation (m)',range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(title = 'X Distance (km)', nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(title = 'Y Distance (km)', nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)

		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/elev_grid_%s.html' %(self.filename, sample_num), validate=False)
		return

	def viewCrossSection(self, list_xslice, list_yslice):
		"""
		Function to visualise the prediction alongside the cross section of the topography/grid
		
		Parameters
		----------
		variable : list_xslice, list_yslice
			cross section of elevation grid at x,y co-ordinate of the grid
		"""

		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)

		x_ymid_real = self.real_elev[xmid, :] 
		x_ymid_mean = list_xslice.mean(axis=1)
		x_ymid_5th = np.percentile(list_xslice, 5, axis=1)
		x_ymid_95th= np.percentile(list_xslice, 95, axis=1)
		
		y_xmid_real = self.real_elev[:, ymid ] 
		y_xmid_mean = list_yslice.mean(axis=1)
		y_xmid_5th = np.percentile(list_yslice, 5, axis=1)
		y_xmid_95th= np.percentile(list_yslice, 95, axis=1)

		x = np.linspace(0, x_ymid_mean.size , num=x_ymid_mean.size) 
		x_ = np.linspace(0, y_xmid_mean.size , num=y_xmid_mean.size)

		plt.plot(x, x_ymid_real, label='ground truth') 
		plt.plot(x, x_ymid_5th, label='pred.(5th percen.)')
		plt.plot(x, x_ymid_95th, label='pred.(95th percen.)')
		plt.plot(x, x_ymid_mean, label='pred. (mean)')
		plt.fill_between(x, x_ymid_5th , x_ymid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Uncertainty in topography prediction (cross section)  ")
		plt.xlabel(' Distance in kilometers  ')
		plt.ylabel(' Height in meters')
		plt.savefig(self.filename+'/x_ymid_opt.png') 
		plt.clf()

		plt.plot(x_, y_xmid_real, label='ground truth') 
		plt.plot(x_, y_xmid_5th, label='pred.(5th percen.)')
		plt.plot(x_, y_xmid_95th, label='pred.(95th percen.)')
		plt.plot(x_, y_xmid_mean, label='pred. (mean)') 
		plt.xlabel(' Distance in kilometers ')
		plt.ylabel(' Height in meters')		
		plt.fill_between(x_, y_xmid_5th , y_xmid_95th, facecolor='g', alpha=0.4)
		plt.legend(loc='upper right')

		plt.title("Uncertainty in topography prediction  (cross section)  ")
		plt.savefig(self.filename+'/y_xmid_opt.png') 
		plt.clf()

	def storeParams(self, naccept, pos_rain, pos_erod, pos_m, pos_n, pos_tau_elev, pos_tau_erdp, pos_tau_erdp_pts, pos_likl): 
		"""
		storing the posterior distributions of parameters in a txt/csv file
		"""
		pos_rain = str(pos_rain)
		pos_erod = str(pos_erod)
		pos_m = str(pos_m)
		pos_n = str(pos_n)
		pos_tau_elev = str(pos_tau_elev)
		pos_tau_erdp = str(pos_tau_erdp)
		pos_tau_erdp_pts = str(pos_tau_erdp_pts)
		pos_likl = str(pos_likl)
		if not os.path.isfile(('%s/exp_data.txt' % (self.filename))):
			with file(('%s/exp_data.txt' % (self.filename)),'w') as outfile:
				outfile.write('{0} '.format(pos_rain))
				outfile.write('{0} '.format(pos_erod))
				outfile.write('{0} \n'.format(pos_likl))
				outfile.write('')

		else:
			with file(('%s/exp_data.txt' % (self.filename)),'a') as outfile:
				outfile.write('{0} '.format(pos_rain))
				outfile.write('{0} '.format(pos_erod))
				outfile.write('{0} \n'.format(pos_likl))

	def likelihoodFunc(self,input_vector, real_elev, real_erdp, real_erdp_pts, tausq_elev, tausq_erdp, tausq_erdp_pts):
		"""
		Likelihood function implementation to be used for the MCMC chain in the metropolis-Hastings acceptance ratio
		"""
		pred_elev_vec, pred_erdp_vec, pred_erdp_pts_vec = self.blackBox(input_vector[0], input_vector[1], input_vector[2], input_vector[3])

		tausq_elev = (np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))/real_elev.size
		sq_error_elev = (np.sum(np.square(pred_elev_vec[self.simtime] - real_elev)))/real_elev.size

		tausq_erdp_pts = np.zeros(self.sim_interval.size)
		for i in range(self.sim_interval.size):
			tausq_erdp_pts[i] = np.sum(np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]))/real_erdp_pts.shape[1]
		
		likelihood_elev = -0.5 * np.log(2* math.pi * tausq_elev) - 0.5 * np.square(pred_elev_vec[self.simtime] - real_elev) / tausq_elev
		likelihood_erdp_pts = 0

		if self.likl_sed:
			#likelihood_erdp  = -0.5 * np.log(2* math.pi * tausq_erdp) - 0.5 * np.square(pred_erdp_vec[self.simtime] - real_erdp) / tausq_erdp		
			for i in range(1,self.sim_interval.size):
				likelihood_erdp_pts += np.sum(-0.5 * np.log(2* math.pi * tausq_erdp_pts[i]) - 0.5 * np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]) / tausq_erdp_pts[i])
			
			likelihood = np.sum(likelihood_elev) + (likelihood_erdp_pts*50)

			sq_error_erdp_pts = np.sum(np.square(pred_erdp_pts_vec[self.sim_interval[i]] - self.real_erdp_pts[i]))/real_erdp_pts.shape[1]
			sq_error = sq_error_elev+ sq_error_erdp_pts
			print 'Using sediment pts in the likelihood'
		
		else:
			likelihood = np.sum(likelihood_elev)
			sq_error = sq_error_elev

		return [likelihood, pred_elev_vec, pred_erdp_vec, pred_erdp_pts_vec]

	def sampler(self):
		"""
		Implementation of the MCMC sampler
		"""
		start = time.time()

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
		

		list_yslicepred = np.zeros((samples,self.real_elev.shape[0]))  # slice taken at mid of topography along y axis  
		list_xslicepred = np.zeros((samples,self.real_elev.shape[1])) # slice taken at mid of topography along x axis  
		ymid = int(self.real_elev.shape[1]/2 ) #   cut the slice in the middle 
		xmid = int(self.real_elev.shape[0]/2)

		# List of accepted samples
		count_list = []

		num_div = 0

		print 'Initial Values of parameters: '
		# UPDATE PARAMS AS PER EXPERIMENT
		rain = np.random.uniform(self.rainlimits[0],self.rainlimits[1])
		erod = np.random.uniform(self.erodlimits[0],self.erodlimits[1])
		# rain = 1.50
		# erod = 5.e-5

		m = 0.5
		n = 1.0

		print 'rain :', rain		
		print 'erodibility :', erod		
		print 'm :', m
		print 'n :', n

		# Creating storage for parameters to be passed to blockBox model 
		v_proposal = []
		v_proposal.append(rain)
		v_proposal.append(erod)
		v_proposal.append(m)
		v_proposal.append(n)

		# Output predictions from blockBox model
		init_pred_elev_vec, init_pred_erdp_vec, init_pred_erdp_pts_vec = self.blackBox(v_proposal[0], v_proposal[1], v_proposal[2], v_proposal[3])

		eta_elev = np.log(np.var(init_pred_elev_vec[self.simtime] - real_elev))
		eta_erdp = np.log(np.var(init_pred_erdp_vec[self.simtime] - real_erdp))
		eta_erdp_pts = np.log(np.var(init_pred_erdp_pts_vec[self.simtime] - real_erdp_pts))
		
		tau_elev = np.exp(eta_elev)
		tau_erdp = np.exp(eta_erdp)
		tau_erdp_pts = np.exp(eta_erdp_pts)
		
		step_eta_elev = np.abs(eta_elev*0.02)
		step_eta_erdp = np.abs(eta_erdp*0.02)
		step_eta_erdp_pts = np.abs(eta_erdp_pts*0.02)

		print 'eta_elev = ', eta_elev, 'step_eta_elev', step_eta_elev
		print 'eta_erdp = ', eta_erdp, 'step_eta_erdp', step_eta_erdp
		print 'eta_erdp_pts = ', eta_erdp_pts, 'step_eta_erdp_pts', step_eta_erdp_pts
		# prior_likelihood = 1

		# Recording experimental conditions
		with file(('%s/description.txt' % (self.filename)),'a') as outfile:
			outfile.write('\n\tsamples: {0}'.format(self.samples))
			outfile.write('\n\tstep_rain: {0}'.format(self.step_rain))
			outfile.write('\n\tstep_erod: {0}'.format(self.step_erod))
			outfile.write('\n\tstep_m: {0}'.format(self.step_m))
			outfile.write('\n\tstep_n: {0}'.format(self.step_n))
			outfile.write('\n\tstep_eta_elev: {0}'.format(step_eta_elev))
			outfile.write('\n\tstep_eta_erdp: {0}'.format(step_eta_erdp))
			outfile.write('\n\tstep_eta_erdp_pts: {0}'.format(step_eta_erdp_pts))
			outfile.write('\n\tInitial_proposed_rain: {0}'.format(rain))
			outfile.write('\n\tInitial_proposed_erod: {0}'.format(erod))
			outfile.write('\n\tInitial_proposed_m: {0}'.format(m))
			outfile.write('\n\tInitial_proposed_n: {0}'.format(n))
			outfile.write('\n\terod_limits: {0}'.format(self.erodlimits))
			outfile.write('\n\train_limits: {0}'.format(self.rainlimits))
			outfile.write('\n\tm_limit: {0}'.format(self.mlimit))
			outfile.write('\n\tn_limit: {0}'.format(self.nlimit))
			#outfile.write('\n\tInitial_tausq_elev_n: {0}'.format(np.exp(np.log(np.var(init_pred_elev - real_elev)))))


		# Passing initial variables along with tau to calculate likelihood and rmse
		[likelihood, pred_elev, pred_erdp, pred_erdp_pts] = self.likelihoodFunc(v_proposal, real_elev, real_erdp, real_erdp_pts, tau_elev, tau_erdp, tau_erdp_pts)
		print '\tinitial likelihood:', likelihood #, 'and initial rmse:', rmse

		# Storing RMSE, tau values and adding initial run to accepted list
		pos_tau_elev = np.full(samples, tau_elev)
		pos_tau_erdp = np.full(samples,tau_erdp)
		pos_tau_erdp_pts = np.full(samples, tau_erdp_pts)

		pos_likl = np.zeros(samples, likelihood)

		prev_acpt_elev = deepcopy(pred_elev)
		prev_acpt_erdp = deepcopy(pred_erdp)
		prev_acpt_erdp_pts = deepcopy(pred_erdp_pts)
		
		# Saving parameters for Initial run
		self.storeParams(0, pos_rain[0], pos_erod[0],pos_m[0], pos_n[0], pos_tau_elev[0], pos_tau_erdp[0] , pos_tau_erdp_pts[0], pos_likl[0]) #, pos_rmse[0])

		sum_elev = deepcopy(pred_elev)
		sum_erdp = deepcopy(pred_erdp)
		sum_erdp_pts = deepcopy(pred_erdp_pts)
		burnsamples = int(samples*0.05)
		count_list.append(0)

		for i in range(samples-1):
			print '\nSample : ', i

			# Updating rain parameter and checking limits
			p_rain = rain + np.random.normal(0,self.step_rain)
			if p_rain < self.rainlimits[0]:
				p_rain = rain
			elif p_rain > self.rainlimits[1]:
				p_rain = rain

			# p_rain = rain

			# Updating edodibility parameter and checking limits
			p_erod = erod + np.random.normal(0, self.step_erod)
			if p_erod < self.erodlimits[0]:
				p_erod = erod
			elif p_erod > self.erodlimits[1]:
				p_erod = erod

			# p_erod = erod

			p_m = m
			p_n = n

			# Creating storage for parameters to be passed to blockBox model
			v_proposal = []
			v_proposal.append(p_rain)
			v_proposal.append(p_erod)
			v_proposal.append(p_m)
			v_proposal.append(p_n)

			#++++++++++++++++++++++++++++++ 
			# IMPT: With the current implementation of the likelihood function
			# random walk not being used on tau or eta. It is instead integrated
			# out and analytically approximated.

			# Updating eta_elev and and recalculating tau for proposal (pro)
			eta_elev_pro = eta_elev + np.random.normal(0, step_eta_elev, 1)
			tau_elev_pro = math.exp(eta_elev_pro)
			
			eta_erdp_pro = eta_erdp + np.random.normal(0, step_eta_erdp, 1)
			tau_erdp_pro = math.exp(eta_erdp_pro)

			eta_erdp_pts_pro = eta_erdp_pts + np.random.normal(0, step_eta_erdp_pts, 1)
			tau_erdp_pts_pro = math.exp(eta_erdp_pts_pro)
			print 'eta_el', eta_elev_pro, 'eta_ero', eta_erdp_pro, 'eta_ero_pts', eta_erdp_pts_pro, 'tau_el', tau_elev_pro, 'tau_ero', tau_erdp_pro, 'tau_ero_pts', tau_erdp_pts_pro

			# ++++++++++++++++++++++++++++++


			# Passing paramters to calculate likelihood and rmse with new tau
			[likelihood_proposal, pred_elev, pred_erdp, pred_erdp_pts] = self.likelihoodFunc(v_proposal, real_elev, real_erdp, real_erdp_pts, tau_elev_pro, tau_erdp_pro, tau_erdp_pts_pro)

			final_predtopo = pred_elev[self.simtime]

			# Difference in likelihood from previous accepted proposal
			diff_likelihood = likelihood_proposal - likelihood
			
			print '(Sampler) likelihood_proposal:', likelihood_proposal, 'diff_likelihood: ',diff_likelihood, '\n'

			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0,1)
			#print 'u', u, 'and mh_probability', mh_prob

			if u < mh_prob: # Accept sample
				print i, 'ACCEPTED\n with likelihood:',likelihood
				count_list.append(i)			# Append sample number to accepted list
				likelihood = likelihood_proposal
				eta_elev = eta_elev_pro
				eta_erdp = eta_erdp
				eta_erdp_pts = eta_erdp_pts
				erod = p_erod
				rain = p_rain
				m = p_m
				n = p_n

				pos_erod[i+1] = erod
				pos_rain[i+1] = rain
				pos_m[i+1] = m
				pos_n[i+1] = n
				pos_tau_elev[i + 1,] = tau_elev_pro
				pos_tau_erdp[i + 1,] = tau_erdp_pro
				pos_tau_erdp_pts[i + 1,] = tau_erdp_pts_pro

				list_yslicepred[i+1,:] =  final_predtopo[:, ymid] # slice taken at mid of topography along y axis  
				list_xslicepred[i+1,:]=   final_predtopo[xmid, :]  # slice taken at mid of topography along x axis 

				pos_likl[i + 1,] = likelihood
				
				self.storeParams(i, pos_rain[i + 1], pos_erod[i + 1], pos_m[i+1], pos_n[i+1], pos_tau_elev[i+1,], pos_tau_erdp[i+1,] , pos_tau_erdp_pts[i+1,], pos_likl[i+1,])
				
				#Save the previous accepted to current in case next is rejected
				prev_acpt_elev.update(pred_elev)
				prev_acpt_erdp.update(pred_erdp)
				prev_acpt_erdp_pts.update(pred_erdp_pts)
				
				if i>burnsamples:
					for k, v in pred_elev.items():
						sum_elev[k] += v

					for k, v in pred_erdp.items():
						sum_erdp[k] += v

					for k, v in pred_erdp_pts.items():
						sum_erdp_pts[k] += v
					
					num_div += 1

			else: # Reject sample
				pos_erod[i+1] = pos_erod[i]
				pos_rain[i+1] = pos_rain[i]
				pos_m[i+1] = pos_m[i]
				pos_n[i+1] = pos_n[i]
				pos_tau_elev[i + 1,] = pos_tau_elev[i,]
				pos_tau_erdp[i + 1,] = pos_tau_erdp[i,]
				pos_tau_erdp_pts[i + 1,] = pos_tau_erdp_pts[i,]
				pos_likl[i + 1,] = pos_likl[i,]
				
				list_yslicepred[i+1,:] =  list_yslicepred[i,:] 
				list_xslicepred[i+1,:]=   list_xslicepred[i,:]
				
				self.storeParams(i, pos_rain[i + 1], pos_erod[i + 1], pos_m[i+1], pos_n[i+1], pos_tau_elev[i+1,], pos_tau_erdp[i+1,] , pos_tau_erdp_pts[i+1,], pos_likl[i+1,]) #Save last accepted parameters in accept file # pos_rmse[i+1,],
				
				if i>burnsamples:
					for k, v in prev_acpt_elev.items():
						sum_elev[k] += v
					
					for k, v in prev_acpt_erdp.items():
						sum_erdp[k] += v
					
					for k, v in prev_acpt_erdp_pts.items():
						sum_erdp_pts[k] += v
					
					num_div += 1

				print 'REJECTED\n with likelihood: ',likelihood
		
		for k, v in sum_elev.items():
			sum_elev[k] = np.divide(sum_elev[k], num_div)
			mean_pred_elevation = sum_elev[k]
			np.savetxt(self.filename+'/prediction_data/mean_pred_elev_%s.txt' %(k), mean_pred_elevation, fmt='%.5f')
			self.viewGrid('mean_pred_elevation%s' %(k), 'Mean Elevation_%s' %(k), '-', '-', zData=mean_pred_elevation, title='Export Slope Grid ')

		mse_elev = (np.sum(np.square(sum_elev[self.simtime] - self.real_elev)))/real_elev.size

		for k, v in sum_erdp.items():
			sum_erdp[k] = np.divide(sum_erdp[k], num_div)
			mean_pred_erdp = sum_erdp[k]
			np.savetxt(self.filename+'/prediction_data/mean_pred_erdp_%s.txt' %(k), mean_pred_erdp, fmt='%.5f')
			self.viewMap('mean_pred_erdp_%s' %(k), 'Mean erdp_%s' %(k), '-', '-', zData=mean_pred_erdp, title='Export Slope Grid ')

		mse_erdp = (np.sum(np.square(sum_erdp[self.simtime] - self.real_erdp)))/real_erdp.size

		for k, v in sum_erdp_pts.items():
			sum_erdp_pts[k] = np.divide(sum_erdp_pts[k], num_div)
			mean_pred_erdp_pts = sum_erdp_pts[k]
			np.savetxt(self.filename+'/prediction_data/mean_pred_erdp_pts_%s.txt' %(k), mean_pred_erdp_pts, fmt='%.5f')
			self.viewBar('mean_pred_erdp_pts_%s' %(k), 'Mean erdp pts_%s' %(k), '-', '-',xData = self.erdp_coords , yData=mean_pred_erdp_pts, title='Export Slope Grid ')

		mse_erdp_pts = (np.sum(np.square(sum_erdp_pts[self.simtime] - self.real_erdp_pts)))/real_erdp_pts.size

		end = time.time()
		total_time = end - start
		total_time_mins = total_time/60
		accepted_count =  len(count_list)
		print (count_list)
		accept_ratio = accepted_count / (samples * 1.0) * 100
		
		print 'Time elapsed: (s)', total_time
		print accepted_count, ' number accepted'
		print len(count_list) / (samples * 0.01), '% was accepted'
		print 'Results are stored in ', self.filename
		
		with file(('%s/experiment_stats.txt' % (self.filename)),'w') as outres:
			outres.write('MSEelev: {0}\nMSEerdp: {1}\nMSEerdp_pts: {2}\nTime:(s) {3}\nTime:(mins) {4}\n'.format(mse_elev,mse_erdp,mse_erdp_pts,total_time,total_time_mins))
			outres.write('Accept ratio: {0} %\nSamples accepted : {1} out of {2}\n Count List : {3} '.format(accept_ratio, accepted_count, self.samples, count_list))
			outres.write('Time Elapsed: (s) {0} , (mins): {1}'.format(total_time, total_time_mins))
		np.savetxt('%s/prediction_data/pred_xslc.txt' % (self.filename), list_xslicepred )
		np.savetxt('%s/prediction_data/pred_yslc.txt' % (self.filename), list_yslicepred )

		self.viewCrossSection(list_xslicepred.T, list_yslicepred.T)
		return

def main():
	"""
		
	"""
	random.seed(time.time())
	muted = True
	run_nb = 0
	directory = ""
	likl_sed = False
	erdp_coords_crater = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
	erdp_coords_crater_fast = np.array([[60,60],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69],[79,91],[96,77],[42,49]])
	erdp_coords_etopo = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])
	erdp_coords_etopo_fast = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[68,40],[72,44]])

	#choice = input("Please choose a Badlands example to run the MCMC algorithm on:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")
	choice = int(sys.argv[1])
	samples = int(sys.argv[2])
	#samples = input("Please enter number of samples : \n")

	if choice == 1:
		directory = 'Examples/crater_fast'
		xmlinput = '%s/crater.xml' %(directory)
		simtime = 15000
		rainlimits = [0.0, 3.0]
		erodlimits = [3.e-5, 7.e-5]
		mlimit = [0.4, 0.6]
		nlimit = [0.9, 1.1]
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
		true_rain = 1.5
		true_erod = 5.e-6
		likl_sed = False
		erdp_coords = erdp_coords_etopo
	else:
		print('Invalid selection, please choose a problem from the list ')

	final_elev = np.loadtxt('%s/data/final_elev.txt' %(directory))
	final_erdp = np.loadtxt('%s/data/final_erdp.txt' %(directory))
	final_erdp_pts = np.loadtxt('%s/data/final_erdp_pts.txt' %(directory))	

	while os.path.exists('%s/mcmcresults_%s' % (directory,run_nb)):
		run_nb+=1
	if not os.path.exists('%s/mcmcresults_%s' % (directory,run_nb)):
		os.makedirs('%s/mcmcresults_%s' % (directory,run_nb))
		os.makedirs('%s/mcmcresults_%s/plots' % (directory,run_nb))
		os.makedirs('%s/mcmcresults_%s/prediction_data' % (directory,run_nb))
		filename = ('%s/mcmcresults_%s' % (directory,run_nb))

	print '\nInput file shape', final_elev.shape, '\n'
	run_nb_str = 'mcmcresults_' + str(run_nb)

	bl_mcmc = bayeslands_mcmc(muted, simtime, samples, final_elev, final_erdp, final_erdp_pts, erdp_coords, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb_str, likl_sed)
	bl_mcmc.sampler()

	print '\nsuccessfully sampled\nFinished simulations'

if __name__ == "__main__": main()