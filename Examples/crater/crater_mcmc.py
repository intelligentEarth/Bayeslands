##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the BayesLands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to implement an MCMC (Markov Chain Monte Carlo) Metropolis Hastings methodology to pyBadlands. 
Badlands is used as a "black box" model for bayesian methods.
"""

import numpy as np
import random
import time
import math
import copy
from copy import deepcopy
import cmocean as cmo
from pylab import rcParams
import fnmatch
import shutil
import collections
from PIL import Image
from io import StringIO
from cycler import cycler
import os

import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html

class Crater_MCMC():
	def __init__(self, muted, simtime, samples, real_elev , filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb):
		self.filename = filename
		self.input = xmlinput
		self.real_elev = real_elev
		self.simtime = simtime
		self.samples = samples
		self.run_nb = run_nb
		self.muted = muted
		self.erodlimits = erodlimits
		self.rainlimits = rainlimits
		self.mlimit = mlimit
		self.nlimit = nlimit
		self.font = 9
		self.width = 1

		self.initial_erod = []
		self.initial_rain = []
		self.initial_m = []
		self.initial_n = []

		self.step_rain = (rainlimits[1]-erodlimits[0])*0.25
		self.step_erod = (erodlimits[1] - erodlimits[0])*0.25
		self.step_m = 0.05
		self.step_n = 0.05
		self.step_eta = 0

		self.sim_interval = np.arange(0, self.simtime+1, 1000)

	def blackbox(self, rain, erodibility, m , n):
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

		variable: erodep
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
		erodep_vec = collections.OrderedDict()
		
		for x in range(len(self.sim_interval)):
			
			self.simtime = self.sim_interval[x]
			
			model.run_to_time(self.simtime, muted = self.muted)
			
			elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
			
			#print 'elev shape  ', elev.shape
			#print 'erodep shape  ', erodep.shape
			#print 'SIMTIME IS   ' , self.simtime

			elev_vec[self.simtime] = elev
			erodep_vec[self.simtime] = erodep
			
			print 'Badlands black box model took (s):',time.clock()-tstart

		return elev_vec, erodep_vec	## Considering elev as pred variable to be compared	

	def plotElev(self,size=(8,8),elev=None,erodep=None, name = None):
		rcParams['figure.figsize']=size
		if elev is not None:
			fig = plt.figure()
			ax=plt.gca()
			im = ax.imshow(np.flipud(elev),interpolation='nearest',cmap=cmo.cm.delta,
							vmin=elev.min(), vmax=elev.max())
			plt.title('Elevation [m]', fontsize=10)
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="2%", pad=0.2)
			plt.colorbar(im,cax=cax)
			plt.show()
			plt.savefig(name)
			plt.close(fig)
		if erodep is not None:
			fig = plt.figure()
			ax=plt.gca()
			im = ax.imshow(np.flipud(erodep),interpolation='nearest',cmap=cmo.cm.balance,
							vmin=erodep.min(), vmax=-erodep.min())
			plt.title('Erosion/deposition [m]', fontsize=10)
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="2%", pad=0.2)
			plt.colorbar(im,cax=cax)
			plt.show()
			plt.close(fig)
		return

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

	def viewGrid(self, sample_num, likl, rain, erod, width = 1600, height = 1600, zmin = None, zmax = None, zData = None, title='Export Grid'):
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

		if zmin == None:
			zmin = self.zData.min()

		if zmax == None:
			zmax = self.zData.max()

		data = Data([ Surface( x=zData.shape[0], y=zData.shape[1], z=zData, colorscale='YIGnBu' ) ])

		layout = Layout(
			title='Crater Elevation  	rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
			autosize=True,
			width=width,
			height=height,
			scene=Scene(
				zaxis=ZAxis(range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			)
		)

		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/plot_image_%s.html' %(self.filename, sample_num), validate=False)
		return

	def save_accepted_params(self, naccept, pos_rain, pos_erod, pos_m, pos_n, pos_rmse, pos_tau, pos_likl):
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

		pos_m = str(pos_m)
		if not os.path.isfile(('%s/accept_m.txt' % (self.filename))):
			with file(('%s/accept_m.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_m)
		else:
			with file(('%s/accept_m.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_m)

		pos_n = str(pos_n)
		if not os.path.isfile(('%s/accept_n.txt' % (self.filename))):
			with file(('%s/accept_n.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_n)
		else:
			with file(('%s/accept_n.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_n)

		pos_rmse = str(pos_rmse)
		if not os.path.isfile(('%s/accept_rmse.txt' % (self.filename))):
			with file(('%s/accept_rmse.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rmse)
		else:
			with file(('%s/accept_rmse.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rmse)

		pos_tau = str(pos_tau)
		if not os.path.isfile(('%s/accept_taus.txt' % (self.filename))):
			with file(('%s/accept_taus.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_tau)
		else:
			with file(('%s/accept_taus.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_tau)

		pos_likl = str(pos_likl)
		if not os.path.isfile(('%s/accept_likl.txt' % (self.filename))):
			with file(('%s/accept_likl.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)
		else:
			with file(('%s/accept_likl.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)

	def rmse(self, pred_elev, real_elev):
		rmse =np.sqrt(((pred_elev - real_elev) ** 2).mean())
		return rmse

	def likelihood_func(self,input_vector, real_elev, tausq):
		pred_elev_vec, pred_erodep_vec = self.blackbox(input_vector[0], input_vector[1], input_vector[2], input_vector[3])

		rmse = 0 #self.rmse(pred_elev, real_elev)
		
		likelihood = - 0.5 * np.log(2* math.pi * tausq) - 0.5 * np.square(pred_elev_vec[self.simtime] - real_elev) / tausq

		return [np.sum(likelihood), pred_elev_vec, rmse]

	def sampler(self):
		
		# Initializing variables
		samples = self.samples
		real_elev = self.real_elev

		self.viewGrid('real', 0 , 2, 5.e-4, width=1000, height=1000, zmin=-10, zmax=600, zData=real_elev, title='Real Elevation')

		# Creating storage for data
		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)
		pos_m = np.zeros(samples)
		pos_n = np.zeros(samples)
		
		# List of accepted samples
		count_list = []

		#
		num_div = 0

		# Generating close to real values
		rain = np.random.uniform(3.3,3.4)
		print 'rain initial value', rain		
		erod = np.random.uniform(3.e-5,4.e-5)
		print 'erod initial value', erod		
		# m = np.random.uniform(0.49,0.51)
		m = 0.5
		print 'm initial value', m
		# n = np.random.uniform(0.99,1.01)
		n = 1.0
		print 'n initial value', n

		# Recording experimental conditions
		with file(('%s/description.txt' % (self.filename)),'a') as outfile:
			outfile.write('\n\samples: {0}'.format(self.samples))
			outfile.write('\n\tstep_rain: {0}'.format(self.step_rain))
			outfile.write('\n\tstep_erod: {0}'.format(self.step_erod))
			outfile.write('\n\tstep_m: {0}'.format(self.step_m))
			outfile.write('\n\tstep_n: {0}'.format(self.step_n))
			outfile.write('\n\tstep_eta: {0}'.format(self.step_eta))
			outfile.write('\n\tInitial_proposed_rain: {0}'.format(rain))
			outfile.write('\n\tInitial_proposed_erod: {0}'.format(erod))
			outfile.write('\n\tInitial_proposed_m: {0}'.format(m))
			outfile.write('\n\tInitial_proposed_n: {0}'.format(n))
			outfile.write('\n\terod_limits: {0}'.format(self.erodlimits))
			outfile.write('\n\train_limits: {0}'.format(self.rainlimits))
			outfile.write('\n\tm_limit: {0}'.format(self.mlimit))
			outfile.write('\n\tn_limit: {0}'.format(self.nlimit))
			#outfile.write('\n\tInitial_tausq_n: {0}'.format(np.exp(np.log(np.var(init_pred_elev - real_elev)))))

		# Creating storage for parameters to be passed to Blackbox model 
		v_proposal = []
		v_proposal.append(rain)
		v_proposal.append(erod)
		v_proposal.append(m)
		v_proposal.append(n)

		# Output predictions from Blackbox model
		init_pred_elev_vec, init_pred_erodep_vec = self.blackbox(v_proposal[0], v_proposal[1], v_proposal[2], v_proposal[3])

		# Calculating eta and tau / Geofffrey's Prior for Tausq
		eta = np.log(np.var(init_pred_elev_vec[self.simtime] - real_elev))
		print 'eta = ', eta
		self.step_eta = eta * 0.05
		tau_pro = np.exp(eta)
		prior_likelihood = 1

		#  Passing initial variables along with tau to calculate likelihood and rmse
		[likelihood, pred_elev, rmse] = self.likelihood_func(v_proposal, real_elev, tau_pro)
		print '\tinitial likelihood:', likelihood, 'and initial rmse:', rmse

		# Storing RMSE, tau values and adding initial run to accepted list
		pos_rmse = np.full(samples, rmse)
		pos_tau = np.full(samples, tau_pro)
		pos_likl = np.zeros(samples, likelihood)
		count_list.append(0)

		prev_acpt_elev = deepcopy(pred_elev)
		
		# Saving parameters for Initial run
		self.save_accepted_params(0, pos_rain[0], pos_erod[0],pos_m[0], pos_n[0], pos_rmse[0], pos_tau[0], pos_likl[0])

		self.viewGrid(0, likelihood, rain, erod, width=1000, height=1000, zmin=-10, zmax=600, zData=pred_elev[self.simtime], title='Export Slope Grid')

		start = time.time()

		sum_elevation = np.zeros((init_pred_elev_vec[self.simtime].shape[0], init_pred_elev_vec[self.simtime].shape[1]))
		
		sum_elev = deepcopy(pred_elev)

		burnsamples = int(samples*0.05)

		#print 'MY PRED KEYS SHOULD BE = SIMTIMES ,  ', pred_elev.keys()
		#print 'sum elevation shape ', sum_elevation.shape

		for i in range(samples-1):

			print 'Sample : ', i

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

			# # Updating m parameter and checking limits
			# p_m = m + np.random.normal(0,self.step_m)
			# if p_m < self.rainlimits[0]:
			# 	p_m = m
			# elif p_m > self.rainlimits[1]:
			# 	p_m = m

			p_m = m

			# # Updating n parameter and checking limits
			# p_n = n + np.random.normal(0,self.step_n)
			# if p_n < self.rainlimits[0]:
			# 	p_n = n
			# elif p_n > self.rainlimits[1]:
			# 	p_n = n

			p_n = n

			# Creating storage for parameters to be passed to Blackbox model
			v_proposal = []
			v_proposal.append(p_rain)
			v_proposal.append(p_erod)
			v_proposal.append(p_m)
			v_proposal.append(p_n)

			# Updating eta and and recalculating tau for proposal (pro)
			eta_pro = eta + np.random.normal(0, self.step_eta, 1)
			print 'eta_pro', eta_pro
			tau_pro = math.exp(eta_pro)
			#print 'tau_pro ', tau_pro

			# Passing paramters to calculate likelihood and rmse with new tau
			[likelihood_proposal, pred_elev, rmse] = self.likelihood_func(v_proposal, real_elev, tau_pro)

			# Difference in likelihood from previous accepted proposal
			diff_likelihood = likelihood_proposal - likelihood
			print '(Sampler) likelihood_proposal:', likelihood_proposal, 'diff_likelihood: ',diff_likelihood

			try:
				mh_prob = min(1, math.exp(diff_likelihood))
			except OverflowError as e:
				mh_prob = 1

			u = random.uniform(0,1)
			#print 'u', u, 'and mh_probability', mh_prob

			# Save sample parameters 
			# self.viewGrid(i, likelihood_proposal, p_rain, p_erod, width=1000, height=1000, zmin=-10, zmax=600, zData=pred_elev, title='Export Slope Grid '+ str(i))

			if u < mh_prob: # Accept sample
				print i, ' is accepted sample'
				count_list.append(i)			# Append sample number to accepted list
				likelihood = likelihood_proposal
				eta = eta_pro
				erod = p_erod
				rain = p_rain
				m = p_m
				n = p_n

				print  '(Sampler) likelihood:',likelihood, ' and rmse:', rmse, 'accepted'
				pos_erod[i+1] = erod
				pos_rain[i+1] = rain
				pos_m[i+1] = m
				pos_n[i+1] = n
				pos_tau[i + 1,] = tau_pro
				pos_rmse[i + 1,] = rmse
				pos_likl[i + 1,] = likelihood
				
				self.save_accepted_params(i, pos_rain[i + 1], pos_erod[i + 1], pos_m[i+1], pos_n[i+1], pos_rmse[i+1,],pos_tau[i+1,], pos_likl[i+1,]) #Save accepted parameters in accept file
				
				#Save the previous accepted to current in case next is rejected
				prev_acpt_elev.update(pred_elev)
				
				if i>burnsamples:
					for k, v in pred_elev.items():
						sum_elev[k] += v

					# sum_elevation = sum_elevation + pred_elev[self.simtime]

					# if sum_elev[self.simtime].all() == sum_elevation.all():
					# 	print 'TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE'

					num_div += 1

			else: # Reject sample
				pos_erod[i+1] = pos_erod[i]
				pos_rain[i+1] = pos_rain[i]
				pos_m[i+1] = pos_m[i]
				pos_n[i+1] = pos_n[i]
				pos_tau[i + 1,] = pos_tau[i,]
				pos_rmse[i + 1,] = pos_rmse[i,]
				pos_likl[i + 1,] = pos_likl[i,]
				
				self.save_accepted_params(i, pos_rain[i + 1], pos_erod[i + 1], pos_m[i+1], pos_n[i+1], pos_rmse[i+1,],pos_tau[i+1,], pos_likl[i+1,]) #Save last accepted parameters in accept file
				
				if i>burnsamples:
					for k, v in prev_acpt_elev.items():
						sum_elev[k] += v

					# sum_elevation = sum_elevation + prev_acpt_elev[self.simtime]

					# if sum_elev[self.simtime].all() == sum_elevation.all():
					# 	print 'THIS IS ALSO TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE TRUE'

					num_div += 1

				print '\n REJECTED\nlikelihood: ',likelihood
				print 'Sample ', i, ' rejected and retained'
		
		print 'num_div', num_div
		for k, v in sum_elev.items():
			sum_elev[k] = np.divide(sum_elev[k], num_div)
			mean_pred_elevation = sum_elev[k]
			np.savetxt(self.filename+'/mean_pred_elev_%s.txt' %(k), mean_pred_elevation, fmt='%.5f')
			self.viewGrid('mean_pred_elevation%s' %(k), 'Mean Elevation_%s' %(k), '-', '-', width=1000, height=1000, zmin=-10, zmax=600, zData=mean_pred_elevation, title='Export Slope Grid ')

		# print 'divisor', samples - burnsamples -2
		# mean_pred_elevation = np.divide(sum_elevation, samples-burnsamples-2)
		# np.savetxt(self.filename+'/mean_pred_elevation.txt', mean_pred_elevation, fmt='%.5f')
		# self.viewGrid('mean_pred_elevation', 'Mean Elevation', '-', '-', width=1000, height=1000, zmin=-10, zmax=600, zData=mean_pred_elevation, title='Export Slope Grid ')


		burnin = 0.05 * samples  # use post burn in samples
		pos_rmse = pos_rmse[int(burnin):,]
		pos_tau = pos_tau[int(burnin):, ]
		pos_likl = pos_likl[int(burnin):,]
		pos_erod = pos_erod[int(burnin):]
		pos_rain = pos_rain[int(burnin):]
		pos_m = pos_m[int(burnin):]
		pos_n = pos_n[int(burnin):]

		end = time.time()
		total_time = end - start
		print 'Time elapsed:', total_time

		accepted_count =  len(count_list)
		print accepted_count, ' number accepted'
		print len(count_list) / (samples * 0.01), '% was accepted'
		accept_ratio = accepted_count / (samples * 1.0) * 100

		return (pos_rain, pos_erod, pos_m, pos_n, pos_tau, pos_rmse, pos_likl, accept_ratio, accepted_count)
	
def main():

	random.seed(time.time())
	muted = True
	xmlinput = 'crater.xml'
	simtime = 75000
	samples = 30000
	run_nb = 0
	rainlimits = [0,5]
	erodlimits = [1.e-5,9.e-5]
	mlimit = [0 , 2]
	nlimit = [0 , 4]

	while os.path.exists('mcmcresults_%s' % (run_nb)):
		run_nb+=1
	if not os.path.exists('mcmcresults_%s' % (run_nb)):
		os.makedirs('mcmcresults_%s' % (run_nb))
		os.makedirs('mcmcresults_%s/plots' % (run_nb))
		filename = ('mcmcresults_%s' % (run_nb))

	final_elev = np.loadtxt('data/final.txt')

	print 'Input file shape', final_elev.shape
	run_nb_str = 'mcmcresults_' + str(run_nb)

	crater_mcmc = Crater_MCMC(muted, simtime, samples, final_elev, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb_str)
	[pos_rain, pos_erod, pos_m, pos_n, pos_tau, pos_rmse, pos_likl, accept_ratio, accepted_count] = crater_mcmc.sampler()

	print 'successfully sampled'

	with file(('%s/out_results.txt' % (filename)),'w') as outres:
		outres.write('Accept ratio: {0} %\nSamples accepted : {1} out of {2}\n'.format(accept_ratio, accepted_count, samples))

	print '\nFinished simulations'

if __name__ == "__main__": main()