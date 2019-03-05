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
This script is intended to perform post processing functionality on the results obtained from MCMC.
This includes plotting target distribution, displaying timevariantErodep and evaluating covariance matrix.
"""
import os
import re
import numpy as np
import random
import time
import math
import cmocean as cmo
import fnmatch
import shutil
import plotly
import argparse
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

parser=argparse.ArgumentParser(description='PTBayeslands modelling')

parser.add_argument('-p','--problem', help='Problem Number 1-crater-fast,2-crater,3-etopo-fast,4-etopo,5-null,6-mountain', required=True, dest="problem",type=int)
parser.add_argument('-f','--functionality', help="Would you like to: \n 1) Plot Posterior Histogram for Params\n 2) Calculate Covariance mat for Params\n 3) Sediment variation with time\n", required=True, dest="functionality",type=int)
parser.add_argument('-b','--bins', help="number of bins in Histogram", required=True, dest="bins",type=int)

args = parser.parse_args()
problem = args.problem
functionality = args.functionality
bins = args.bins

def plotFunctions(fname, pos_likl, pos_rain, pos_erod, bins, t_val):
	
	burnin = 0.05 * len(pos_likl)  # use post burn in samples
	pos_likl = pos_likl[int(burnin):,]
	pos_erod = pos_erod[int(burnin):]
	pos_rain = pos_rain[int(burnin):]

	nb_bins= bins
	slen = np.arange(0,len(pos_likl),1)
	font = 9
	width = 1

	rain_true_val = np.asarray(t_val[0], dtype = float)
	erod_true_val = np.asarray(t_val[1], dtype = float)
	print 'r_t ', rain_true_val
	print 'e_t ', erod_true_val 
	color = ['brown','blue','magenta','red','purple','darkorange','deepskyblue','mediumslateblue','limegreen','black']
	######################################     RAIN      ################################

	rainmin, rainmax = min(pos_rain), max(pos_rain)
	#print rainmin, rainmax, len(pos_likl)
	rainspace = np.linspace(rainmin,rainmax,len(pos_rain))
	rainm,rains = stats.norm.fit(pos_rain)
	pdf_rain = stats.norm.pdf(rainspace,rainm,rains)
	
	# fig = plt.figure(figsize=(10,12))
	# ax = fig.add_subplot(111)
	# ax.spines['top'].set_color('none')
	# ax.spines['bottom'].set_color('none')
	# ax.spines['left'].set_color('none')
	# ax.spines['right'].set_color('none')
	# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	# ax.set_title(' Rain Parameter', fontsize=  font+2)#, y=1.02)
	
	# ax1 = fig.add_subplot(211)
	# ax1.set_facecolor('#f2f2f3')
	# # ax1.set_axis_bgcolor("white")
	# n, rainbins, patches = ax1.hist(pos_rain, bins=nb_bins, alpha=0.5, facecolor='sandybrown', density=False)	
	
	# for count, v in enumerate(rain_true_val):
	# 		ax1.axvline(x=v, color='%s' %(color[count]), linestyle='dashed', linewidth=1) # comment when go real value is 

	# ax1.grid(True)
	# ax1.set_ylabel('Frequency',size= font+1)
	# ax1.set_xlabel('Rain', size= font+1)
	
	# ax2 = fig.add_subplot(212)
	# ax2.set_facecolor('#f2f2f3')
	# ax2.plot(slen,pos_rain,linestyle='-', linewidth= width, color='k', label=None)
	# ax2.set_title('Trace of Rain',size= font+2)
	# ax2.set_xlabel('Samples',size= font+1)
	# ax2.set_ylabel('Rain', size= font+1)
	# ax2.set_xlim([0,np.amax(slen)])

	# fig.tight_layout()
	# fig.subplots_adjust(top=0.88)
	# plt.savefig('%srain.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	# plt.clf()

	size = 15

	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)
	plt.grid(alpha=0.75)

	plt.hist(pos_rain, bins=nb_bins, alpha=0.5, facecolor='sandybrown', density=False)	
	plt.title("Posterior distribution ", fontsize = size)
	plt.xlabel(' Parameter value  ', fontsize = size)
	plt.ylabel(' Frequency ', fontsize = size)
	plt.tight_layout()  
	plt.savefig(fname + 'plots/' +'rain_posterior.pdf')
	plt.clf()
	
	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)

	plt.plot(slen,pos_rain,linestyle='-', linewidth= width, color='k', label=None)
	plt.title("Parameter trace plot", fontsize = size)
	plt.xlabel(' Number of Samples  ', fontsize = size)
	plt.ylabel(' Parameter value ', fontsize = size)
	plt.tight_layout()  
	plt.savefig(fname + 'plots/' +'rain_trace.pdf')
	plt.clf()


	#####################################      EROD    ################################

	erodmin, erodmax = min(pos_erod), max(pos_erod)
	erodspace = np.linspace(erodmin,erodmax,len(pos_erod))
	print 'erodmin', erodmin
	print 'erodmax', erodmax
	erodm,erods = stats.norm.fit(pos_erod)
	pdf_erod = stats.norm.pdf(erodspace,erodm,erods)

	# #erod_opt_value = 9.e-05	
	# fig = plt.figure(figsize=(10,12))
	# ax = fig.add_subplot(111)
	# ax.spines['top'].set_color('none')
	# ax.spines['bottom'].set_color('none')
	# ax.spines['left'].set_color('none')
	# ax.spines['right'].set_color('none')
	# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
	# ax.set_title(' Erosion Parameter', fontsize=  font+2)#, y=1.02)

	# ax1 = fig.add_subplot(211)
	# ax1.set_facecolor('#f2f2f3')
	# # ax1.set_axis_bgcolor("white")
	# n, erodbins, patches = ax1.hist(pos_erod, bins=nb_bins, alpha=0.5, facecolor='sandybrown', density=False)

	# for count, v in enumerate(erod_true_val):
	# 		ax1.axvline(x=v, color='%s' %(color[count]), linestyle='dashed', linewidth=1) # comment when go real value is 

	# ax1.grid(True)
	# ax1.set_ylabel('Frequency',size= font+1)
	# ax1.set_xlabel('Erodibility', size= font+1)
	
	# ax2 = fig.add_subplot(212)
	# ax2.set_facecolor('#f2f2f3')
	# ax2.plot(slen,pos_erod,linestyle='-', linewidth= width, color='k', label=None)
	# ax2.set_title('Trace of Erodibility',size= font+2)
	# ax2.set_xlabel('Samples',size= font+1)
	# ax2.set_ylabel('Erodibility', size= font+1)
	# ax2.set_xlim([0,np.amax(slen)])
	# fig.tight_layout()
	# fig.subplots_adjust(top=0.88)
	# plt.savefig('%serod.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	# plt.clf()
	size = 15

	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)
	plt.grid(alpha=0.75)

	plt.hist(pos_erod, bins=nb_bins, alpha=0.5, facecolor='sandybrown', density=False)	
	plt.title("Posterior distribution ", fontsize = size)
	plt.xlabel(' Parameter value  ', fontsize = size)
	plt.ylabel(' Frequency ', fontsize = size)
	plt.tight_layout()  
	plt.savefig(fname + 'plots/'  + 'erod_posterior.pdf')
	plt.clf()
	
	plt.tick_params(labelsize=size)
	params = {'legend.fontsize': size, 'legend.handlelength': 2}
	plt.rcParams.update(params)

	plt.plot(slen,pos_erod,linestyle='-', linewidth= width, color='k', label=None)
	plt.title("Parameter trace plot", fontsize = size)
	plt.xlabel(' Number of Samples  ', fontsize = size)
	plt.ylabel(' Parameter value ', fontsize = size)
	plt.tight_layout()  
	plt.savefig(fname + 'plots/' +'erod_trace.pdf')
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
	ax.tick_params(labelcolor='w', top=False, bottom=False, left = False, right=False)
	
	ax1 = fig.add_subplot(211)
	ax1.set_facecolor('#f2f2f3')
	ax1.plot(slen,pos_likl,linestyle='-', linewidth= width, color='k', label=None)
	ax1.set_title('Trace of Likelihood',size= font+2)
	ax1.set_xlabel('Samples',size= font+1)
	ax1.set_ylabel('Log likelihood', size= font+1, labelpad = 10)
	ax1.set_xlim([0,np.amax(slen)])
	ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.88)
	plt.savefig('%slikl.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	plt.clf()

	####################################      Joint    ################################
	fig = plt.figure(figsize=(10,12))
	ax = fig.add_subplot(111)
	ax.spines['top'].set_color('none')
	ax.spines['bottom'].set_color('none')
	ax.spines['left'].set_color('none')
	ax.spines['right'].set_color('none')
	ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
	ax.set_title(' Rain Parameter', fontsize=  font+2)#, y=1.02)
	
	ax1 = fig.add_subplot(212, projection = '3d')
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

	# ####################################      Ratio      ####################################
	
	# fig = plt.figure(figsize=(8,10))
	# ax = fig.add_subplot(111)
	# ax.spines['top'].set_color('none')
	# ax.spines['bottom'].set_color('none')
	# ax.spines['left'].set_color('none')
	# ax.spines['right'].set_color('none')
	# ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

	# ax1 = fig.add_subplot(211)
	# ratio = pos_rain/pos_erod
	# ax1.plot(ratio, pos_likl)
	# plt.savefig('%sratio.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	# ax1.set_title(r'Ratio',size= font+2)
	# ax1.set_xlabel('Ratio',size= font+1)
	# ax1.set_ylabel(r'Likelihood', size= font+1)
	# # ax1.set_xlim([0,np.amax(ratio)])
	# fig.tight_layout()
	# fig.subplots_adjust(top=0.88)
	# plt.savefig('%sratio.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
	# plt.clf()

def plotLiklSurf(fname, pos_likl, pos_rain, pos_erod):
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

def atoi(text):

	return int(text) if text.isdigit() else text

def natural_keys(text):
	'''
	alist.sort(key=natural_keys) sorts in human order
	http://nedbatchelder.com/blog/200712/human_sorting.html
	(See Toothy's implementation in the comments)
	'''
	return [ atoi(c) for c in re.split('(\d+)', text) ]

def timevariantErodep(directory, fname, real_erdp_pts, filenames, run_nb):
	
	fig, ax = plt.subplots()
	index = np.arange(real_erdp_pts.shape[1])
	width = 0.30
	opacity = 0.8 

	if directory == "Examples/crater_fast/":
		erdp_coords = np.array([[60,60],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69],[79,91],[96,77],[42,49]])
	elif directory == "Examples/crater/":
		erdp_coords = np.array([[60,60],[52,67],[74,76],[62,45],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69]])
	elif directory == "Examples/etopo_fast/":
		erdp_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[68,40],[72,44]])
	elif directory == "Examples/etopo/":
		erdp_coords = np.array([[42,10],[39,8],[75,51],[59,13],[40,5],[6,20],[14,66],[4,40],[72,73],[46,64]])
	else:
		erdp_coords = np.array([[60,60],[72,66],[85,73],[90,75],[44,86],[100,80],[88,69],[79,91],[96,77],[42,49]])

	filenames.sort(key = natural_keys)

	for count, list_name in enumerate(filenames):
		var = np.loadtxt(fname+list_name)
		print 'count ', count
		# print 'real_erdp_pts[count] ', real_erdp_pts[count]
		# print 'var ', var
		rects1 = plt.bar(index, real_erdp_pts[count], width,alpha=opacity,color='b',label='Real')
		rects2 = plt.bar(index + width, var, width,alpha=opacity,color='g',label='Predicted')
		plt.xlabel('Selected Coordinates')
		tick_space = np.arange(0.0 , 10.0, step = 1.0)
		tick_space += 0.25 
		plt.xticks(tick_space, erdp_coords, fontsize = 8)
		plt.ylabel('Height in meters')
		plt.title('Erosion Deposition') 
		plt.legend() 
		plt.tight_layout() 
		plt.savefig('%s/p_erdp_%s_m%s.png' %(fname, count, run_nb))
		plt.clf()

	return

def main():
	directory = ""

	if problem == 1:
		directory = 'Examples/crater_fast'
		rain_true_val = 1.5
		erod_true_val = 5.e-5
	elif problem == 2:
		directory = 'Examples/crater'
		rain_true_val = 1.5
		erod_true_val = 5.e-5
	elif problem == 3:
		directory = 'Examples/etopo_fast'
		rain_true_val = 1.5
		erod_true_val = 5.e-6
	elif problem == 4:
		directory = 'Examples/etopo'
		rain_true_val = 1.5
		erod_true_val = 5.e-6
	elif problem ==5:
		directory = 'Examples/tasmania'
		rain_true_val = 1.5
		erod_true_val = 5.e-6

	# run_nb = input('Please enter the folder number for Experiment i.e. mcmcresults_% ')
	run_nb = np.loadtxt('latest_run.txt')
	run_nb = int(run_nb)
	fname = '%s/mcmcresults_%s/' % (directory,run_nb)
	exp_data = '%s/mcmcresults_%s/exp_data.txt' % (directory,run_nb)
	prediction_data = '%s/mcmcresults_%s/prediction_data/' % (directory,run_nb)
	
	
	filename_list = []
	filename_list.append(exp_data)
	
	rain = []
	erod = []
	likl = []

	for list_name in filename_list:
		with open(list_name) as f:
			for count, line in enumerate(f):
				# print count,' ', line
				term = line.split()
				# print 'term', term
				rain.append(term[0])
				# print 'rain', rain
				erod.append(term[1])
				# print 'erod', erod
				likl.append(term[2])
				# print 'likl', likl

	print 'length of likl', len(likl), ' rain', len(rain), ' erod', len(erod)

	rain_ = np.asarray(rain, dtype = float)
	erod_ = np.asarray(erod, dtype = float)
	likl_ = np.asarray(likl, dtype = float)
	t_val_ = np.loadtxt('%s/data/true_values.txt' % (directory))
	erdp_pts_data = np.loadtxt('%s/data/final_erdp_pts.txt' % (directory))
	prefixed = [filename for filename in os.listdir(prediction_data) if filename.startswith("mean_pred_erdp_pts_")]

	if functionality == 1:
		plotFunctions(fname, likl_, rain_, erod_, bins, t_val_)
		print '\nFinished plotting'

	elif functionality ==2:
		covarMat(fname, rain_, erod_)
		print '\n Covariance Matrix has been created'

	elif functionality ==3:
		timevariantErodep(directory, prediction_data, erdp_pts_data, prefixed, run_nb)
		print 'Finished plotting time variant erodep'

	
if __name__ == "__main__": main()

