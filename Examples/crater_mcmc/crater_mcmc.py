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

class MCMC():
	def __init__(self, simtime, samples, real_data , filename, xmlinput, erodlimits, rainlimits):
		self.filename = filename
		self.input = xmlinput
		self.samples = samples       
		self.real_data = real_data
		self.simtime = simtime

		self.erodlimits = erodlimits
		self.rainlimits = rainlimits
		self.font = 9
		self.width = 1

		self.initial_erod = []
		self.initial_rain = []

		self.step_erod = 1.e-5
		self.step_rain = 0.02
		self.step_eta = 0.05

	def plotElev(self,size=(8,8),elev=None,erodep=None):
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

	def blackbox(self, rain, erodibility):
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
		model.load_xml(self.input)

		# Adjust erodibility based on given parameter
		model.input.SPLero = erodibility
		model.flow.erodibility.fill(erodibility)

		# Adjust precipitation values based on given parameter
		model.force.rainVal[:] = rain

		# Run badlands simulation
		model.run_to_time(self.simtime)

		# Extract
		elev,erodep = self.interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)
		#elev = self.interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)

		#self.plotElev(elev = elev, erodep = erodep)

		print 'Badlands black box model took (s):',time.clock()-tstart

		return elev,erodep	## Considering elev as ystar
		#return elev
	
	def likelihood_func(self, input_vector, y, tausq):
		ystar, erodep = self.blackbox(input_vector[0],input_vector[1])
		rmse = self.rmse(ystar, y)
		loss = -0.5 * np.log(2* math.pi * tausq) - 0.5 * np.square(ystar - y) / tausq
		return [np.sum(loss), ystar, rmse]

	def rmse(self, prediction, target):
		rmse =np.sqrt(((prediction - target) ** 2).mean(axis = None))
		return rmse

	def prior_likelihood(self):
		
		return 1
	
	def sampler(self):
		data_size = self.real_data.shape[0]
		samples = self.samples
		y_data = self.real_data

		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)
		#pos_samples = np.zeros((samples, data_size))
		rmse = np.zeros(samples)

		#Initial prediction
		max_erod = 1.0
		max_rain = 1.0

		#erod = pos_erod[0] = np.random.uniform(0., max_erod)
		#rain = pos_rain[0] = np.random.uniform(0., max_rain)
		
		erod = pos_erod[0] = 9.e-5
		rain = pos_rain[0] = 1.0
	
		print 'erod ' , erod , ' rain ', rain
		v_proposal=[]
		v_proposal = np.append(v_proposal,(erod,rain))
		pos_v = np.zeros((samples, v_proposal.size))

		initial_pred, erodep = self.blackbox(v_proposal[0], v_proposal[1])

		eta = np.log(np.var(initial_pred - y_data))
		tau_pro = np.exp(eta)
		prior_likelihood = 1

		[likelihood, pred_data, rmse] = self.likelihood_func(v_proposal, y_data, tau_pro)
		pos_rmse = np.full(samples, rmse)
		pos_tau = np.full(samples, tau_pro)
		#pos_samples[0,:] = pred_data

		print '\tinitial likelihood:', likelihood, 'and rmse:', rmse
		naccept = 0
		count_list = []
		count_list.append(0)

		# function to save params and figures. -> TBW
		for i in range(samples - 1):
			print '\nSample: ', i
			start = time.time()
			p_erod = erod + np.random.normal(0,self.step_erod,1)
			if p_erod < 0:
			    p_erod = erod
			elif p_erod > max_erod:
			    p_erod = erod

			p_rain = rain + np.random.normal(0,self.step_rain,1)
			if p_rain < 0:
			    p_rain = p_rain
			elif p_rain > max_rain:
			    p_rain = rain

			v_proposal = []
			v_proposal = np.append(v_proposal,(erod,rain))

			eta_pro = eta + np.random.normal(0, self.step_eta, 1)
			tau_pro = math.exp(eta_pro)
			[likelihood_proposal, pred_data, rmse] = self.likelihood_func(v_proposal, y_data, tau_pro)
			diff_likelihood = likelihood_proposal - likelihood # to divide probability, must subtract
			print '(Sampler) likelihood_proposal:', likelihood_proposal, 'diff_likelihood',diff_likelihood
			
			mh_prob = min(1, math.exp(diff_likelihood))
			u = random.uniform(0,1)
			print 'u', u, 'and mh_probability', mh_prob

			if u < mh_prob:
				#accept
				print i, ' is accepted sample'
				naccept += 1
				count_list.append(i)
				likelihood = likelihood_proposal
				eta = eta_pro
				erod = p_erod
				rain = p_rain

				print  '(Sampler) likelihood:',likelihood, ' and rmse:', rmse, 'accepted'
				pos_erod[i+1] = erod
				pos_rain[i+1] = rain
				pos_v[i+1,] = v_proposal
				pos_tau[i + 1,] = tau_pro
				#pos_samples[i + 1,] = pred_data
				pos_rmse[i + 1,] = rmse
				self.save_params(i, pos_erod[i + 1,], pos_rain[i + 1], pos_rmse[i + 1,])
			else:
				pos_v[i + 1,] = pos_v[i,]
				pos_tau[i + 1,] = pos_tau[i,]
				#pos_samples[i + 1,] = pos_samples[i,]
				pos_rmse[i + 1,] = pos_rmse[i,]
				print 'REJECTED\nLikelihood:',likelihood,'and RMSE rejected:', pos_rmse[i,]
				pos_erod[i+1] = pos_erod[i]
				pos_rain[i+1] = pos_rain[i]
				print i, 'rejected and retained'
			end = time.time()
			total_time = end - start
			print 'Time elapsed:', total_time
		accepted_count =  len(count_list)
		print accepted_count, ' number accepted'
		print len(count_list) / (samples * 0.01), '% was accepted'
		accept_ratio = accepted_count / (samples * 1.0) * 100

		return (pos_v, pos_tau, pos_erod, pos_rain, pos_rmse, accept_ratio, accepted_count)
	

	def save_params(self,naccept, pos_erod, pos_rain, pos_rmse):
		erod__ = str(pos_erod)
		if not os.path.isfile('%s/accept_erod.txt' % (self.filename)):
			with file(('%s/accept_erod.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(erod__)
		else:
			with file(('%s/accept_erod.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(erod__)

		rain__ = str(pos_rain)
		if not os.path.isfile('%s/accept_rain.txt' % (self.filename)):
			with file(('%s/accept_rain.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(rain__)
		else:
			with file(('%s/accept_rain.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(rain__)


		rmse__ = str(pos_rmse)
		if not os.path.isfile('%s/accept_rmse.txt' % (self.filename)):
			with file(('%s/accept_rmse.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(rmse__)
		else:
			with file(('%s/accept_rmse.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\n'.format(naccept))
				outfile.write(rmse__)

def main():
	random.seed(time.time())
	samples = 350
	simtime = 100000
	run_nb = 0
	xmlinput = 'crater.xml'

	erodlimts = [0.,1.0]
	rainlimits = [0.,1.0]
	data_mcmc = np.loadtxt('data/badlands.txt') #z2DReal

	while os.path.exists('mcmcresults_%s' % (run_nb)):
		run_nb+=1
	if not os.path.exists('mcmcresults_%s' % (run_nb)):
		os.makedirs('mcmcresults_%s' % (run_nb))
		filename = ('mcmcresults_%s' % (run_nb))

	mcmc = MCMC(simtime, samples, data_mcmc, filename, xmlinput, erodlimts, rainlimits)

	[pos_v, pos_tau, pos_erod, pos_rain, pos_rmse, accept_ratio, accepted_count] = mcmc.sampler()

	print 'successfully sampled'

	burnin = 0.05 * samples  # use post burn in samples
	pos_v = pos_v[int(burnin):, ]
	pos_tau = pos_tau[int(burnin):, ]
	pos_erod = pos_erod[int(burnin):]
	pos_rain = pos_rain[int(burnin):]
	
	rmse_mu = np.mean(pos_rmse[int(burnin):])
	rmse_std = np.std(pos_rmse[int(burnin):])

	print 'mean rmse:',rmse_mu, 'standard deviation:', rmse_std

	with file(('%s/out_results.txt' % (filename)),'w') as outres:
		outres.write('Mean RMSE: {0}\nStandard deviation: {1}\nAccept ratio: {2} %\nSamples accepted : {3} out of {4}\n'.format(rmse_mu, rmse_std, accept_ratio, accepted_count, samples))

	if not os.path.isfile(('%s/out_pos.csv' % (filename))):
		np.savetxt("%s/out_pos.csv" % (filename), pos_v, delimiter=',')

	print 'mean rmse:',rmse_mu, 'standard deviation:', rmse_std
	print 'Finished simulations'

if __name__ == "__main__": main()
