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
import cmocean as cmo
from pylab import rcParams
import fnmatch
import shutil
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
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

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

		self.step_rain = 0.01
		self.step_erod = 9.e-6
		self.step_m = 0.05
		self.step_n = 0.05
		self.step_eta = 0.007

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
		telev = []

		# Run badlands simulation for input simulation time
		model.run_to_time(self.simtime, muted = self.muted)
		# Extract elevation and erosion-deposition grid
		elev,erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)
		telev.append(elev)
		# HeatMap visualisation for elevation and erosion-deposition
		self.plotElev(elev = None, erodep = None, name = str(self.run_nb))

		# simulation_times = np.arange(0, self.simtime+1, 1000)
		# elevation_vec = np.zeros(simulation_times.size)

		# for x in simulation_times:
		# 	self.simtime = simulation_times[x]
		# 	model.run_to_time(self.simtime, muted = self.muted)
		# 	elev, erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2], model.elevation, model.cumdiff)
		# 	self.plotElev(elev = None, erodep = None, name = str(self.run_nb))
		# 	elevation_vec.append(elev)

		print 'Badlands black box model took (s):',time.clock()-tstart

		return elev, erodep	## Considering elev as predicted variable to be compared	

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
			title='Crater Elevation  rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
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

		ax2 = fig.add_subplot(212)
		

		fig.colorbar(surf, shrink=0.5, aspect=5)
		plt.savefig('%s/plot.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
		plt.show()

	def save_accepted_params(self, naccept, pos_rain, pos_erod, pos_m, pos_n, pos_likl):
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

		pos_likl = str(pos_likl)
		if not os.path.isfile(('%s/accept_likl.txt' % (self.filename))):
			with file(('%s/accept_likl.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)
		else:
			with file(('%s/accept_likl.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_likl)

	def rmse(self, predicted_elev, real_elev):
		rmse =np.sqrt(((predicted_elev - real_elev) ** 2).mean())
		return rmse

	def likelihood_func(self,input_vector, real_elev, tausq):
		predicted_elev, predicted_erodep = self.blackbox(input_vector[0], input_vector[1], input_vector[2], input_vector[3])

		rmse = 0 #self.rmse(predicted_elev, real_elev)
		
		likelihood = - 0.5 * np.log(2* math.pi * tausq) - 0.5 * np.square(predicted_elev - real_elev) / tausq

		return [np.sum(likelihood), predicted_elev, rmse]

	def sampler(self):
		
		# Initializing variables
		samples = self.samples
		real_elev = self.real_elev

		self.viewGrid('real', 0 , 1.5, 5.e-5, width=1000, height=1000, zmin=-10, zmax=600, zData=real_elev, title='Real Elevation')

		# Creating storage for data
		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)
		pos_m = np.zeros(samples)
		pos_n = np.zeros(samples)
		
		# List of accepted samples
		count_list = []

		print 'rain dimension', int(math.sqrt(samples))

		rain = np.linspace(0.0, 3.0, num = int(math.sqrt(samples)))
		erod = np.linspace(1.e-5,9.e-5, num = int(math.sqrt(samples)))

		dimx = rain.shape[0]
		dimy = erod.shape[0]

		print rain 
		print erod

		m = 0.5
		n = 1.0

		# Creating storage for parameters to be passed to Blackbox model 
		v_proposal = []
		v_proposal.append(rain[0])
		v_proposal.append(erod[0])
		v_proposal.append(m)
		v_proposal.append(n)

		# Storing RMSE, tau values and adding initial run to accepted list
		pos_likl = np.zeros((dimx, dimy))
		print 'pos_likl', pos_likl.shape
		start = time.time()

		i = 0
		counter = 0
		for r in range(len(rain)):
			for e in range(len(erod)):
				print '\n'
				print 'Rain : ', rain[r], '  Erod : ', erod[e]
				
				# Updating rain parameter and checking limits
				p_rain = rain[r]
				
				# Updating edodibility parameter and checking limits
				p_erod = erod[e]

				p_m = m
				p_n = n

				# Creating storage for parameters to be passed to Blackbox model
				v_proposal = []
				v_proposal.append(p_rain)
				v_proposal.append(p_erod)
				v_proposal.append(p_m)
				v_proposal.append(p_n)

				tausq = 5

				# Passing paramters to calculate likelihood and rmse with new tau
				[likelihood, predicted_elev, rmse] = self.likelihood_func(v_proposal, real_elev, tausq)
				
				m = p_m
				n = p_n
				
				pos_erod[i] = p_erod
				pos_rain[i] = p_rain
				pos_m[i] = m
				pos_n[i] = n
				pos_likl[r,e] = likelihood
				
				self.save_accepted_params(i, pos_rain[i], pos_erod[i], pos_m[i], pos_n[i], pos_likl[r,e]) 

				i += 1
				counter +1

		end = time.time()
		total_time = end - start
		print 'counter', counter
		print 'Time elapsed:', total_time

		accepted_count =  0
		print len(count_list) / (samples * 0.01), '% was accepted'
		accept_ratio = accepted_count / (samples * 1.0) * 100

		print 'pos_likl.shape', pos_likl.shape

#		self.viewGrid(100, likelihood, pos_rain, pos_erod, width=1000, height=1000, zmin=-10, zmax=600, zData=pos_likl, title='Export Slope Grid')

		self.plotFunctions(self.filename, pos_likl, rain, erod)

		return (rain, erod, pos_m, pos_n, pos_likl, accept_ratio, accepted_count)

def main():

	random.seed(time.time())
	muted = True
	xmlinput = 'crater.xml'
	simtime = 50000
	samples = 22500
	run_nb = 0
	rainlimits = [0,3]
	erodlimits = [1.e-5,9.e-5]
	mlimit = [0 , 2]
	nlimit = [0 , 4]

	while os.path.exists('res_bruteForce%s' % (run_nb)):
		run_nb+=1
	if not os.path.exists('res_bruteForce%s' % (run_nb)):
		os.makedirs('res_bruteForce%s' % (run_nb))
		os.makedirs('res_bruteForce%s/plots' % (run_nb))
		filename = ('res_bruteForce%s' % (run_nb))

	final_elev = np.loadtxt('data/final_elev.txt')

	print 'Input file shape', final_elev.shape
	run_nb_str = 'res_bruteForce' + str(run_nb)

	crater_mcmc = Crater_MCMC(muted, simtime, samples, final_elev, filename, xmlinput, erodlimits, rainlimits, mlimit, nlimit, run_nb_str)
	[pos_rain, pos_erod, pos_m, pos_n,  pos_likl, accept_ratio, accepted_count] = crater_mcmc.sampler()

	print 'successfully sampled'

	with file(('%s/out_results.txt' % (filename)),'w') as outres:
		outres.write('Accept ratio: {0} %\nSamples accepted : {1} out of {2}\n'.format(accept_ratio, accepted_count, samples))

	print '\nFinished simulations'

if __name__ == "__main__": main()