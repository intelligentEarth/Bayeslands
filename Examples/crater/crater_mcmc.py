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

class Crater_MCMC():
	def __init__(self, simtime, samples, ydata , filename, xmlinput, erodlimits, rainlimits, counter):
		self.filename = filename
		self.input = xmlinput
		self.ydata = ydata
		self.simtime = simtime
		self.samples = samples
		
		self.erodlimits = erodlimits
		self.rainlimits = rainlimits
		self.font = 9
		self.width = 1
		
		self.initial_erod = []
		self.initial_rain = []

		self.step_rain = 0.05		
		self.step_erod = 5.e-6
		self.step_eta = 0.005

		##DELETE LATER
		self.counter = counter

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
			#plt.show()
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

	def blackbox(self, rain, erodibility):
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

		self.plotElev(elev = None, erodep = None, name = str(self.counter))

		print 'Badlands black box model took (s):',time.clock()-tstart

		return elev,erodep	## Considering elev as ystar

	def save_params(self,naccept, pos_rain, pos_erod, pos_rmse):
		pos_rain = str(pos_rain)
		if not os.path.isfile(('%s/accept_m.txt' % (self.filename))):
			with file(('%s/accept_m.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_rain)
		else:
			with file(('%s/accept_m.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rain)

		pos_erod = str(pos_erod)		
		if not os.path.isfile(('%s/accept_m.txt' % (self.filename))):
			with file(('%s/accept_m.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))    
				outfile.write(pos_rain)
		else:
			with file(('%s/accept_m.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(pos_rain)

		rmse__ = str(pos_rmse)
		if not os.path.isfile(('%s/accept_rmse.txt' % (self.filename))):
			with file(('%s/accept_rmse.txt' % (self.filename)),'w') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(rmse__)
		else:
			with file(('%s/accept_rmse.txt' % (self.filename)),'a') as outfile:
				outfile.write('\n# {0}\t'.format(naccept))
				outfile.write(rmse__)

	def rmse(self, ystar, ydata):
		rmse =np.sqrt(((ystar - ydata) ** 2).mean(axis = None))
		return rmse

	def prior_loss(self):	
		return 1

	def loss_func(self,input_vector, ydata, tausq):
		ystar, erodep = self.blackbox(input_vector[0], input_vector[1])
		rmse = self.rmse(ystar, ydata)
		loss = -0.5 * np.log(2* math.pi * tausq) - 0.5 * np.square(ystar - ydata) / tausq
		#loss = 0.5
		#loss = - 0.5 * np.square(ystar - ydata) / tausq
		return [np.sum(loss), ystar, rmse]

	def sampler(self):
		#initializing variables
		samples = self.samples
		ydata = self.ydata
		naccept = 0
		
		#Creating storage for data
		pos_erod = np.zeros(samples)
		pos_rain = np.zeros(samples)

		count_list = []

		#Initial Prediction
		rain = np.random.normal(0.8, step_rain)
		erod = np.random.normal(8.e-5, step_erod)

		v_proposal = []
		v_proposal.append(rain)
		v_proposal.append(erod)

		ystar_initial, erodep = self.blackbox(v_proposal[0], v_proposal[1])
		eta = np.log(np.var(ystar_initial - ydata))
		tau_pro = np.exp(eta)
		prior_loss = 1

		[loss, ystar, rmse] = self.loss_func(v_proposal, ydata, tau_pro)
		pos_rmse = np.full(samples, rmse)
		pos_tau = np.full(samples, tau_pro)

		count_list.append(0)
		print '\tinitial loss:', loss, 'and initial rmse:', rmse

		self.save_params(naccept, pos_rain[0], pos_erod[0],pos_rmse[0])
		start = time.time()

		for i in range(samples-1):
			print 'Sample : ', i
			self.counter+=1
			p_erod = erod + np.random.normal(0, self.step_erod, 1)
			if p_erod < self.erodlimits[0]:
			    p_erod = erod
			elif p_erod > self.erodlimits[1]:
			    p_erod = erod

			p_rain = rain + np.random.normal(0,self.step_rain,1)
			if p_rain < self.rainlimits[0]:
			    p_rain = p_rain
			elif p_rain > self.rainlimits[1]:
			    p_rain = rain

			v_proposal = []
			v_proposal.append(p_rain)
			v_proposal.append(p_erod)

			eta_pro = eta + np.random.normal(0, self.step_eta, 1)
			tau_pro = math.exp(eta_pro)
			print 'tau_pro ', tau_pro
			[loss_proposal, ystar, rmse] = self.loss_func(v_proposal, ydata, tau_pro)
			diff_loss = loss_proposal - loss
			print '(Sampler) loss_proposal:', loss_proposal, 'diff_likelihood: ',diff_loss
			
			try:
				mh_prob = min(1, math.exp(diff_loss))
			except OverflowError as e:
				mh_prob = 1
			
			u = random.uniform(0,1)
			print 'u', u, 'and mh_probability', mh_prob

			if u < mh_prob: #accept
				print i, ' is accepted sample'
				naccept += 1
				count_list.append(i)
				loss = loss_proposal
				eta = eta_pro
				erod = p_erod
				rain = p_rain
				print  '(Sampler) loss:',loss, ' and rmse:', rmse, 'accepted'
				pos_erod[i+1] = erod
				pos_rain[i+1] = rain
				pos_tau[i + 1,] = tau_pro
				pos_rmse[i + 1,] = rmse
				self.save_params(naccept, pos_rain[i + 1], pos_erod[i + 1], pos_rmse[i+1,])

			else: #reject
				pos_tau[i + 1,] = pos_tau[i,]
				pos_rmse[i + 1,] = pos_rmse[i,]
				pos_erod[i+1] = pos_erod[i]
				pos_rain[i+1] = pos_rain[i]
				print 'REJECTED\nLoss:',loss,'and RMSE rejected:', pos_rmse[i,]
				print i, 'rejected and retained'
		
		end = time.time()
		total_time = end - start		
		print 'Time elapsed:', total_time
		accepted_count =  len(count_list)
		print accepted_count, ' number accepted'
		print len(count_list) / (samples * 0.01), '% was accepted'
		accept_ratio = accepted_count / (samples * 1.0) * 100
		
		return (pos_rain, pos_erod, pos_tau, pos_rmse, accept_ratio, accepted_count)

def main():
	
	random.seed(time.time())
	xmlinput = 'crater.xml'
	simtime = 150000
	samples = 10000
	run_nb = 0
	rainlimits = [0.5,4.0]
	erodlimts = [1.e-6,1.e-4]

	## DELETE LATER. Only to visualise
	counter = 0

	while os.path.exists('mcmcresults_%s' % (run_nb)):
		run_nb+=1
	if not os.path.exists('mcmcresults_%s' % (run_nb)):
		os.makedirs('mcmcresults_%s' % (run_nb))
		filename = ('mcmcresults_%s' % (run_nb))
	
	input_file = np.loadtxt('data/badlands.txt')
	print '\ninput file shape ', input_file.shape, '\n'

	crater_mcmc = Crater_MCMC(simtime, samples, input_file, filename, xmlinput, erodlimts, rainlimits, counter)
	[pos_rain, pos_erod, pos_tau, pos_rmse, accept_ratio, accepted_count] = crater_mcmc.sampler()

	print 'successfully sampled'

	burnin = 0.05 * samples  # use post burn in samples
	pos_tau = pos_tau[int(burnin):, ]
	pos_erod = pos_erod[int(burnin):]
	pos_rain = pos_rain[int(burnin):]
	
	rmse_mu = np.mean(pos_rmse[int(burnin):])
	rmse_std = np.std(pos_rmse[int(burnin):])

	print 'mean rmse:',rmse_mu, 'standard deviation:', rmse_std

	with file(('%s/out_results.txt' % (filename)),'w') as outres:
		outres.write('Mean RMSE: {0}\nStandard deviation: {1}\nAccept ratio: {2} %\nSamples accepted : {3} out of {4}\n'.format(rmse_mu, rmse_std, accept_ratio, accepted_count, samples))

	print '\nFinished simulations'

if __name__ == "__main__": main()

