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
import os
import numpy as np
import random
import time
import math
import cmocean as cmo
import fnmatch
import shutil
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib as mpl
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from PIL import Image
from io import StringIO
from cycler import cycler
from pylab import rcParams
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import cKDTree
from scipy import stats 
from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from plotly.graph_objs import *
from plotly.offline.offline import _plot_html
plotly.offline.init_notebook_mode()

def plotFunctions(fname, pos_likl, pos_rain, pos_erod, pos_tau_elev, pos_tau_erdp, pos_tau_erdppts, bins, rain_true_val, erod_true_val):
	
	burnin = 0.05 * len(pos_likl)  # use post burn in samples
	pos_likl = pos_likl[int(burnin):,]
	pos_erod = pos_erod[int(burnin):]
	pos_rain = pos_rain[int(burnin):]
	pos_tau_elev = pos_tau_elev[int(burnin):, ]
	pos_tau_erdp = pos_tau_erdp[int(burnin):,]
	pos_tau_erdppts = pos_tau_erdppts[int(burnin):,]

	nb_bins= bins
	slen = np.arange(0,len(pos_likl),1)
	font = 9
	width = 1

	######################################     RAIN      ################################

	rainmin, rainmax = min(pos_rain), max(pos_rain)
	#print rainmin, rainmax, len(pos_likl)
	rainspace = np.linspace(rainmin,rainmax,len(pos_rain))
	rainm,rains = stats.norm.fit(pos_rain)
	pdf_rain = stats.norm.pdf(rainspace,rainm,rains)
	
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(' Rain Parameter', fontsize=  font+2)#, y=1.02)
	
	ax1 = fig.add_subplot(211)
	#ax1.set_facecolor('#f2f2f3')
	ax1.set_axis_bgcolor("white")
	n, rainbins, patches = ax1.hist(pos_rain, bins=nb_bins, alpha=0.5, facecolor='sandybrown', normed=False)	
	ax1.axvline(rain_true_val)
	ax1.grid(True)
	ax1.set_ylabel('Frequency',size= font+1)
	ax1.set_xlabel(r'$\varepsilon$', size= font+1)
	
	ax2 = fig.add_subplot(212)
	ax2.set_facecolor('#f2f2f3')
	# ax2.set_axis_bgcolor("white")
	ax2.plot(slen,pos_rain,linestyle='-', linewidth= width, color='k', label=None)
	ax2.set_title(r'Trace of Rain',size= font+2)
	ax2.set_xlabel('Samples',size= font+1)
	ax2.set_ylabel(r'$\varepsilon$', size= font+1)
	ax2.set_xlim([0,np.amax(slen)])

	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig('%srain.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()

	#####################################      EROD    ################################

	erodmin, erodmax = min(pos_erod), max(pos_erod)
	erodspace = np.linspace(erodmin,erodmax,len(pos_erod))
	print 'erodmin', erodmin
	print 'erodmax', erodmax
	erodm,erods = stats.norm.fit(pos_erod)
	pdf_erod = stats.norm.pdf(erodspace,erodm,erods)

	#erod_opt_value = 9.e-05	
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(' Erosion Parameter', fontsize=  font+2)#, y=1.02)

	ax1 = fig.add_subplot(211)
	ax1.set_facecolor('#f2f2f3')
	# ax1.set_axis_bgcolor("white")
	n, erodbins, patches = ax1.hist(pos_erod, bins=nb_bins, alpha=0.5, facecolor='sandybrown', normed=False)
	ax1.axvline(erod_true_val)
	ax1.grid(True)
	ax1.set_ylabel('Frequency',size= font+1)
	ax1.set_xlabel(r'$\varepsilon$', size= font+1)
	
	ax2 = fig.add_subplot(212)
	ax2.set_facecolor('#f2f2f3')
	# ax2.set_axis_bgcolor("white")

	ax2.plot(slen,pos_erod,linestyle='-', linewidth= width, color='k', label=None)
	ax2.set_title(r'Trace of $\varepsilon$',size= font+2)
	ax2.set_xlabel('Samples',size= font+1)
	ax2.set_ylabel(r'$\varepsilon$', size= font+1)
	ax2.set_xlim([0,np.amax(slen)])
	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig('%serod.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()

	#####################################      likl    ################################
	liklmin, liklmax = min(pos_likl), max(pos_likl)
	liklspace = np.linspace(liklmin,liklmax,len(pos_likl))
	liklm,likls = stats.norm.fit(pos_likl)
	pdf_likl = stats.norm.pdf(liklspace,liklm,likls)

	fig = plt.figure(figsize=(8,10))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		
	ax1 = fig.add_subplot(211)
	ax1.set_facecolor('#f2f2f3')
	# ax1.set_axis_bgcolor("white")
	ax1.plot(slen,pos_likl,linestyle='-', linewidth= width, color='k', label=None)
	ax1.set_title(r'Trace of Likelihood',size= font+2)
	ax1.set_xlabel('Samples',size= font+1)
	ax1.set_ylabel(r'Likelihood', size= font+1)
	ax1.set_xlim([0,np.amax(slen)])
	
	tau_elevmin, tau_elevmax = min(pos_tau_elev), max(pos_tau_elev)
	tau_elevspace = np.linspace(tau_elevmin,tau_elevmax,len(pos_tau_elev))
	tau_elevm,tau_elevs = stats.norm.fit(pos_tau_elev)
	pdf_tau_elev = stats.norm.pdf(tau_elevspace,tau_elevm,tau_elevs)
	tau_elevmean=np.mean(pos_tau_elev)
	tau_elevmedian=np.median(pos_tau_elev)

	ax2 = fig.add_subplot(212)
	ax2.set_facecolor('#f2f2f3')
	# ax2.set_axis_bgcolor("white")
	ax2.plot(slen,pos_tau_elev,linestyle='-', linewidth= width, color='k', label=None)
	ax2.set_title(r'Trace of Tau sq',size= font+2)
	ax2.set_xlabel('Samples',size= font+1)
	ax2.set_ylabel(r'tau_elevq', size= font+1)
	ax2.set_xlim([0,np.amax(slen)])
	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig('%slik_tau.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()

	####################################      Joint    ################################
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	ax.set_title(' Rain Parameter', fontsize=  font+2)#, y=1.02)
	
	ax1 = fig.add_subplot(211, projection = '3d')
	ax1.set_facecolor('#f2f2f3')
	# ax1.set_axis_bgcolor("white")
	hist, rainedges, erodedges = np.histogram2d(pos_rain, pos_erod, bins = 15 , range = [[rainmin, rainmax],[erodmin, erodmax]])
	rainpos, erodpos = np.meshgrid(rainedges[:-1], erodedges[:-1])
	rainpos = rainpos.flatten('F')
	erodpos = erodpos.flatten('F')
	zpos = np.zeros_like(rainpos)
	drain = 0.5* np.ones_like(zpos)
	derod = drain.copy()
	dz = hist.flatten()
	ax1.bar3d(rainpos, erodpos, zpos, drain, derod, dz, color = 'g', zsort = 'average')

	trace1 = go.Scatter3d(x=rainpos, y=erodpos, z=dz, mode='markers', marker=dict(size=12, color = dz, colorscale='Portland', showscale=True))
	data = [trace1]
	layout = go.Layout(title='Joint distribution rain, erod',autosize=True,width=1000,height=1000, scene=Scene(
				xaxis=XAxis(title = 'Rain', nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				yaxis=YAxis(title = 'Erodibility', nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
				bgcolor="rgb(244, 244, 248)"
			))
	fig = go.Figure(data=data, layout=layout)
	graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/plots/plot_image_joint3d.html' %(fname), validate=False)

	trace = go.Scattergl(x= pos_rain, y = pos_erod, mode = 'markers', marker = dict(color = '#FFBAD2',line = dict(width = 1)))
	data = [trace]
	layout = go.Layout(margin=dict(l=0,r=0,b=0,t=0))
	fig = go.Figure(data=data, layout=layout)
	graph = plotly.offline.plot(fig, auto_open = False, output_type = 'file', filename='%s/plots/plot_image_jointscatter.html' %(fname), validate=False)

	####################################      Ratio      ####################################
	
	fig = plt.figure(figsize=(8,10))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

	ax1 = fig.add_subplot(211)
	ratio = pos_rain/pos_erod
	ax1.plot(ratio, pos_likl)
	plt.savefig('%sratio.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	ax1.set_title(r'Ratio',size= font+2)
	ax1.set_xlabel('Ratio',size= font+1)
	ax1.set_ylabel(r'Likelihood', size= font+1)
	ax1.set_xlim([0,np.amax(ratio)])
	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig('%sratio.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()

def covarMat (fname, pos_rain, pos_erod):
	c = np.column_stack((rain_,erod_))
	c = c.T
	print 'c_shape', c.shape, c.size
	cov = np.cov(c)
	print cov
	print 'covar shape', cov.shape
	
	if os.path.isfile('%s/covar_matrix.csv' % directory):
		os.remove('%s/covar_matrix.csv' % directory)
	with file(('%s/covar_matrix.csv' % directory),'a') as outfile:
		for i in range(c.shape[0]):
			for j in range(c.shape[0]):
				outfile.write('{0}, '.format(cov[i,j]))
			outfile.write('\n')

	trace = go.Heatmap(z=cov, colorscale='Viridis')
	data=[trace]
	graph  = plotly.offline.plot(data, auto_open=False, output_type='file',  
		filename='%s/heatmap_cov.html' % directory, validate=False)

def main():
	functionality = input("Would you like to: \n 1) Plot Posterior Histogram for Params\n 2) Calculate Covariance mat for Params")
	choice = input("Please choose a Badlands example to apply it to:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n")
	directory = ""

	if choice == 1:
		directory = 'Examples/crater_fast'
		rain_true_val = 1.5
		erod_true_val = 5.e-5
	elif choice == 2:
		directory = 'Examples/crater'
		rain_true_val = 1.5
		erod_true_val = 5.e-5
	elif choice == 3:
		directory = 'Examples/etopo_fast'
		rain_true_val = 1.5
		erod_true_val = 5.e-6
	elif choice == 4:
		directory = 'Examples/etopo'
		rain_true_val = 1.5
		erod_true_val = 5.e-6

	run_nb = input('Please enter the folder number for Experiment i.e. mcmcresults_% ')

	fname = '%s/mcmcresults_%s/' % (directory,run_nb)
	
	filename_list = []
	filename_list.append('%s/mcmcresults_%s/accept_likl.txt' % (directory,run_nb))
	filename_list.append('%s/mcmcresults_%s/accept_rain.txt' % (directory,run_nb))
	filename_list.append('%s/mcmcresults_%s/accept_erod.txt' % (directory,run_nb))
	filename_list.append('%s/mcmcresults_%s/accept_tau_elev.txt' % (directory,run_nb))
	filename_list.append('%s/mcmcresults_%s/accept_tau_erdp.txt' % (directory,run_nb))
	filename_list.append('%s/mcmcresults_%s/accept_tau_erdppts.txt' % (directory,run_nb))

	likl = rain = erod = tau_elev = tau_erdp = tau_erdppts = []

	for list_name in filename_list:
		with open(list_name) as f:
			next(f)
			next(f)
			for line in f:
				words = line.split()
				error = words[2]
				lname =  list_name[-8:-4]
				if lname == 'likl':
					likl.append(error)
				elif lname == 'rain':
					rain.append(error)
				elif lname == 'erod':
					erod.append(error)
				elif lname == 'elev':
					tau_elev.append(error)
				elif lname == 'erdp':
					tau_erdp.append(error)	
				elif lname == 'ppts':
					tau_erdppts.append(error)	

	print 'length of likl', len(likl), ' rain', len(rain), ' erod', len(erod)

	likl_ = np.asarray(likl, dtype = float)
	rain_ = np.asarray(rain, dtype = float)
	erod_ = np.asarray(erod, dtype = float)
	tau_elev_ = np.asarray(tau_elev, dtype = float)
	tau_erdp_ = np.asarray(tau_erdp, dtype = float)
	tau_erdppts_ = np.asarray(tau_erdppts, dtype = float)


	if functionality == 1:
		bins= input("Please enter Bin Size for Histogram (e.g. 10, 20 ,30 ...")
		plotFunctions(fname, likl_, rain_, erod_, tau_elev_, tau_erdp_, tau_erdppts_, bins, rain_true_val, erod_true_val)
		print '\nFinished plotting'

	elif functionality ==2:
		covarMat(fname, rain_, erod_)
		print '\n Covariance Matrix has been created'
	
if __name__ == "__main__": main()

