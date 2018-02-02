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

def plotFunctions(fname, pos_likl, pos_rain, pos_erod, pos_taus):
		nb_bins=30
		slen = np.arange(0,len(pos_likl),1)
		#print 'slen', slen
		font = 9
		width = 1

		######################################     RAIN      ################################

		rainmin, rainmax = min(pos_rain), max(pos_rain)
		#print rainmin, rainmax, len(pos_likl)
		rainspace = np.linspace(rainmin,rainmax,len(pos_rain))
		rainm,rains = stats.norm.fit(pos_rain)
		pdf_rain = stats.norm.pdf(rainspace,rainm,rains)
		#rain_real_value = 0.5

		fig = plt.figure(figsize=(10,12))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
		ax.set_title(' Rain Parameter', fontsize=  font+2)#, y=1.02)
		
		ax1 = fig.add_subplot(211)
		ax1.set_facecolor('#f2f2f3')

		n, rainbins, patches = ax1.hist(pos_rain, bins=nb_bins, alpha=0.5, facecolor='sandybrown', normed=False)
		
		# rainy = mlab.normpdf(rainbins, rainm, rains)
		# l = ax1.plot(rainbins, rainy, 'r--', linewidth= width)

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel(r'$\varepsilon$', size= font+1)
		
		ax2 = fig.add_subplot(212)
		ax2.set_facecolor('#f2f2f3')
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
		erodmean=np.mean(pos_erod)
		erodmedian=np.median(pos_erod)
		#erodmode, count= stats.mode(pos_erod)

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
		
		n, erodbins, patches = ax1.hist(pos_erod, bins=nb_bins, alpha=0.5, facecolor='sandybrown', normed=False)
		
		# erody = mlab.normpdf(erodbins, erodm, erods)
		# l = ax1.plot(erodbins, erody, 'r--', linewidth= width)

		ax1.grid(True)
		ax1.set_ylabel('Frequency',size= font+1)
		ax1.set_xlabel(r'$\varepsilon$', size= font+1)
		
		ax2 = fig.add_subplot(212)
		ax2.set_facecolor('#f2f2f3')
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
		liklmean=np.mean(pos_likl)
		liklmedian=np.median(pos_likl)

		fig = plt.figure(figsize=(8,10))
		ax = fig.add_subplot(111)
		ax.spines['top'].set_color('none')
		ax.spines['bottom'].set_color('none')
		ax.spines['left'].set_color('none')
		ax.spines['right'].set_color('none')
		ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
			
		ax1 = fig.add_subplot(211)
		ax1.set_facecolor('#f2f2f3')
		ax1.plot(slen,pos_likl,linestyle='-', linewidth= width, color='k', label=None)
		ax1.set_title(r'Trace of Likelihood',size= font+2)
		ax1.set_xlabel('Samples',size= font+1)
		ax1.set_ylabel(r'Likelihood', size= font+1)
		ax1.set_xlim([0,np.amax(slen)])
		
		tausmin, tausmax = min(pos_taus), max(pos_taus)
		tausspace = np.linspace(tausmin,tausmax,len(pos_taus))
		tausm,tauss = stats.norm.fit(pos_taus)
		pdf_taus = stats.norm.pdf(tausspace,tausm,tauss)
		tausmean=np.mean(pos_taus)
		tausmedian=np.median(pos_taus)

		ax2 = fig.add_subplot(212)
		ax2.set_facecolor('#f2f2f3')
		ax2.plot(slen,pos_taus,linestyle='-', linewidth= width, color='k', label=None)
		ax2.set_title(r'Trace of Tau sq',size= font+2)
		ax2.set_xlabel('Samples',size= font+1)
		ax2.set_ylabel(r'Tausq', size= font+1)
		ax2.set_xlim([0,np.amax(slen)])
		fig.tight_layout()
		fig.subplots_adjust(top=0.88)
		plt.savefig('%slik_tau.png'% (fname), bbox_inches='tight', dpi=300, transparent=False)
		plt.clf()

def main():

	run_nb = input('Please enter the folder number i.e. mcmcresults_% ')

	fname = 'mcmcresults_%s/' % (run_nb)
	likl_filename = 'mcmcresults_%s/accept_likl.txt' % (run_nb)
	rain_filename = 'mcmcresults_%s/accept_rain.txt' % (run_nb)
	erod_filename = 'mcmcresults_%s/accept_erod.txt' % (run_nb)
	taus_filename = 'mcmcresults_%s/accept_taus.txt' % (run_nb)
	#m_filename = 'mcmcresults_%s/accept_m.txt' % (run_nb)
	#n_filename = 'mcmcresults_%s/accept_n.txt' % (run_nb)
	
	filename_list = []
	filename_list.append(likl_filename)
	filename_list.append(rain_filename)
	filename_list.append(erod_filename)
	filename_list.append(taus_filename)
	#filename_list.append(m_filename)
	#filename_list.append(n_filename)

	likl = []
	rain = []
	erod = []
	taus = []
	#m = []
	#n = []

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
				elif lname == 'taus':
					taus.append(error)	

	print 'length of likl', len(likl)
	print 'length of rain', len(rain)
	print 'length of erod', len(erod)
	print 'length of taus', len(taus)

	likl_ = np.asarray(likl, dtype = float)
	rain_ = np.asarray(rain, dtype = float)
	erod_ = np.asarray(erod, dtype = float)
	taus_ = np.asarray(taus, dtype = float)

	plotFunctions(fname, likl_, rain_, erod_, taus_)
	
	print '\nFinished plotting'

if __name__ == "__main__": main()

