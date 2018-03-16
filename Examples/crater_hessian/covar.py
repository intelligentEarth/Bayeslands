# !/usr/bin/python
# Description: Calculates covariance matrix between parameters in BayesReef
# Authors: Danial Azam and Jodie Pall
# Date: 9 March 2018

import numpy as np
import math
import os
from io import StringIO
import csv
import sys
# import matplotlib as mpl

import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html
import plotly.graph_objs as go 

def main():
	folder = 'results_multinomial_1'
	curr_file = __file__
	loc= os.path.dirname(curr_file)
	print 'loc:', loc
	# folder = '../Testing/results_multinomial_7'
	directory=os.path.join(loc, folder)
	filename=('%s/accepted_proposals.csv' % directory)
	# filename='%s/pos_proposal.csv' % directory



	c = np.loadtxt(open(filename,'rb'), delimiter=",")
	c = c.T
	print 'c_shape', c.shape[0], c.size
	print c[1,:]

	cov = np.cov(c)
	print 'covar shape', cov.shape
	
	sed_cov = np.cov(c[:12,:12])
	print 'sed_cov shape', sed_cov.shape
	flow_cov = np.cov(c[12:24,12:24])
	print 'flow_cov shape', flow_cov.shape
	glv_cov = np.cov(c[24:27,24:27])
	print 'glv_cov shape', glv_cov.shape

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
	
	trace = go.Heatmap(z=sed_cov)
	data=[trace]
	graph  = plotly.offline.plot(data, auto_open=False, output_type='file',  
		filename='%s/heatmap_sedcov.html' % directory, validate=False)
	
	trace = go.Heatmap(z=flow_cov)
	data=[trace]
	graph  = plotly.offline.plot(data, auto_open=False, output_type='file',  
		filename='%s/heatmap_flowcov.html' % directory, validate=False)
	
	trace = go.Heatmap(z=glv_cov)
	data=[trace]
	graph  = plotly.offline.plot(data, auto_open=False, output_type='file',  
		filename='%s/heatmap_glvcov.html' % directory, validate=False)
if __name__=="__main__": main()
