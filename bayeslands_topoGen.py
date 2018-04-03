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

def interpolateArray(coords=None, z=None, dz=None):
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

def topoGenerator(directory,inputname, outputname, rain, erodibility, m, n, etime, erdp_coords, final_noise):
	"""
		
	"""
	model = badlandsModel()
	# Define the XmL input file
	model.load_xml('outputname',inputname, verbose = False, muted = True)

	rreal = rain
	ereal = erodibility

	model.input.SPLero = ereal
	model.flow.erodibility.fill(ereal)
	model.force.rainVal[:] = rreal
	#Adjust m and n values
	model.input.SPLm = m
	model.input.SPLn = n   
	
	model.run_to_time(etime, muted = True)
	
	elev,erdp = interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)

	###########################
	erdp_pts = np.zeros((erdp_coords.shape[0]))

	for count, val in enumerate(erdp_coords):
		erdp_pts[count] = erdp[val[0], val[1]]
	###########################


	############# Adding Noise
	tausq = elev.max()* 0.01

	# print 'elev.size', elev.size
	elev_noise = np.random.normal(0, np.sqrt(tausq), elev.size)
	elev_noise = np.reshape(elev_noise,(elev.shape[0],elev.shape[1]))	
	erdp_noise = np.random.normal(0, np.sqrt(tausq), erdp.size)
	erdp_noise = np.reshape(erdp_noise,(erdp.shape[0],erdp.shape[1]))	
	erdp_pts_noise = np.random.normal(0, np.sqrt(tausq), erdp_pts.size)
	erdp_pts_noise = np.reshape(erdp_pts_noise,(erdp_pts.shape))
	# print 'noise', noise
	# print 'noise shape now ', noise.shape

	#############
	
	elev_=np.matrix(elev)
	erdp_=np.matrix(erdp)
	erdp_pts_ = np.matrix(erdp_pts)

	if final_noise:
		elev_mat=np.add(elev_, elev_noise)
		erdp_mat=np.add(erdp_, erdp_noise)
		erdp_pts_mat = np.add(erdp_pts_, erdp_pts_noise)
	else:
		elev_mat = elev_
		erdp_mat = erdp_
		erdp_pts_mat = erdp_pts_

	np.savetxt('%s/data/%s_elev.txt' %(directory, outputname),elev_mat,fmt='%.5f')
	np.savetxt('%s/data/%s_erdp.txt' %(directory,outputname),erdp_mat,fmt='%.5f')
	np.savetxt('%s/data/%s_erdp_pts.txt' %(directory,outputname),erdp_pts_mat,fmt='%.5f')

	viewGrid(directory,'%s_elev' %(outputname), 'N/A', rreal, ereal, zData=elev_mat, title='Export Slope Grid')
	viewMap(directory,'%s_erdp' %(outputname), 'N/A', rreal, ereal, zData=erdp_mat, title='Export Slope Grid')
	viewBar(directory,'%s_erdp_pts' %(outputname), 'N/A', rreal, ereal, xData = erdp_coords, yData=erdp_pts, title='Export Slope Grid')

	return

def visualiseInput(directory,inputname, outputname, rain, erodibility, m, n, etime, erdp_coords):
	"""
		
	"""
	model = badlandsModel()
	# Define the XmL input file
	model.load_xml('real',inputname, verbose = False, muted = True)

	rreal = rain
	ereal = erodibility

	final_noise = False

	topoGenerator(directory,inputname, outputname, rreal, ereal, m, n, etime, erdp_coords,final_noise)
	
	elev = np.loadtxt('%s/data/%s_elev.txt' %(directory,outputname))
	erdp = np.loadtxt('%s/data/%s_erdp.txt' %(directory,outputname))
	erdp_pts = np.loadtxt('%s/data/%s_erdp_pts.txt' %(directory,outputname))
	
	viewGrid(directory,'%s_elev' %(outputname), 'N/A', rreal, ereal, zData=elev, title='Export Slope Grid')
	# viewMap(directory,'%s_erdp' %(outputname), 'N/A', rreal, ereal,  zData=erdp, title='Export Slope Grid')
	# viewBar(directory,'%s_erdp_pts' %(outputname), 'N/A', rreal, ereal, xData = erdp_coords, yData=erdp_pts, title='Export Slope Grid')
	
	return

def viewGrid(directory,sample_num, likl, rain, erod, width = 1000, height = 1000, zmin = None, zmax = None, zData = None, title='Export Grid'):
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
		zmin = zData.min()

	if zmax == None:
		zmax = zData.max()

	data = Data([ Surface( x=zData.shape[0], y=zData.shape[1], z=zData, colorscale='YIGnBu' ) ])

	layout = Layout(
		title='Crater Elevation     rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
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
	graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/elev_grid_%s.html' %(directory,sample_num), validate=False)
	return    

def viewMap(directory,sample_num, likl, rain, erod, width = 600, height = 600, zmin = None, zmax = None, zData = None, title='Export Grid'):
	"""
	Use Plotly library to visualise the Erosion Deposition Heatmap.

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
			zaxis=ZAxis(range=[zmin, zmax],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
			bgcolor="rgb(244, 244, 248)"
		)
	)

	fig = Figure(data=data, layout=layout)
	graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/erdp_heatmap_%s.html' %(directory,sample_num), validate=False)
	return    

def viewBar(directory,sample_num, likl, rain, erod, width = 500, height = 500, xData = None, yData = None, title='Export Grid'):
		"""
		Use Plotly library to visualise the BarPlot of Erosion Deposition at certain coordinates.

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
		fig = Figure(data=data, layout=layout)
		graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='%s/images/erdppts_barcht_%s.html' %(directory, sample_num), validate=False)
		
		return
	
def main():
	"""
	
	"""
	choice = input("Please choose a Badlands example to create an Initial and Final Topography for:\n 1) crater_fast\n 2) crater\n 3) etopo_fast\n 4) etopo\n 5) delta\n")
	directory = ""

	# erdp_coords_crater = np.array([ [60,60], [72,66], [85,73], [90,75] ])
	erdp_coords_crater = np.array([ [2,3], [3,2], [5,5], [2,9] ])
	erdp_coords_etopo = np.array([ [10,60], [30,30], [60,10], [80,75] ])
	final_noise = True

	if choice == 0:
		
		tstart = time.clock()
		
		directory = 'Examples/crater_'
		visualiseInput(directory,'%s/crater.xml' %(directory) ,'initial', 1.5 , 5.e-5, 0.5, 1, 0, erdp_coords_crater)
		topoGenerator(directory,'%s/crater.xml' %(directory) , 'final', 1.5 , 5.e-5, 0.5, 1, 15000, erdp_coords_crater,final_noise)
		
		print 'TopoGen for crater_fast completed in (s):',time.clock()-tstart

	elif choice == 6:
		tstart = time.clock()
		
		directory = 'Examples/etopo_'
		visualiseInput(directory,'%s/etopo.xml' %(directory) ,'initial', 1.5 , 5.e-6, 0.5, 1, 0, erdp_coords_etopo)
		topoGenerator(directory,'%s/etopo.xml' %(directory) , 'final', 1.5 , 5.e-6, 0.5, 1, 500000, erdp_coords_etopo,final_noise)
		
		print 'TopoGen for etopo completed in (s):',time.clock()-tstart

	elif choice == 1:
		
		tstart = time.clock()
		
		directory = 'Examples/crater_fast'
		visualiseInput(directory,'%s/crater.xml' %(directory) ,'initial', 1.5 , 5.e-5, 0.5, 1, 0, erdp_coords_crater)
		topoGenerator(directory,'%s/crater.xml' %(directory) , 'final', 1.5 , 5.e-5, 0.5, 1, 15000, erdp_coords_crater,final_noise)
		
		print 'TopoGen for crater_fast completed in (s):',time.clock()-tstart
		
	elif choice == 2:
		
		tstart = time.clock()
		
		directory = 'Examples/crater'
		visualiseInput(directory,'%s/crater.xml' %(directory) ,'initial', 1.5 , 5.e-5, 0.5, 1, 0, erdp_coords_crater)
		topoGenerator(directory,'%s/crater.xml' %(directory) , 'final', 1.5 , 5.e-5, 0.5, 1, 50000, erdp_coords_crater,final_noise)
		
		print 'TopoGen for crater completed in (s):',time.clock()-tstart

	elif choice == 3:

		tstart = time.clock()
		
		directory = 'Examples/etopo_fast'
		visualiseInput(directory,'%s/etopo.xml' %(directory) ,'initial', 1.5 , 5.e-6, 0.5, 1, 0, erdp_coords_etopo)
		topoGenerator(directory,'%s/etopo.xml' %(directory) , 'final', 1.5 , 5.e-6, 0.5, 1, 500000, erdp_coords_etopo,final_noise)
		
		print 'TopoGen for etopo fast completed in (s):',time.clock()-tstart

	elif choice == 4:

		tstart = time.clock()
		
		directory = 'Examples/etopo'
		visualiseInput(directory,'%s/etopo.xml' %(directory) ,'initial', 1.5 , 5.e-6, 0.5, 1, 0, erdp_coords_etopo)
		topoGenerator(directory,'%s/etopo.xml' %(directory) , 'final', 1.5 , 5.e-6, 0.5, 1, 500000, erdp_coords_etopo,final_noise)
		
		print 'TopoGen for etopo completed in (s):',time.clock()-tstart

	elif choice == 5:

		tstart = time.clock()
		
		directory = 'Examples/mountain'
		visualiseInput(directory,'%s/mountain.xml' %(directory) ,'initial', 1.5 , 5.e-5, 0.5, 1, 0, erdp_coords_mountain)
		topoGenerator(directory,'%s/mountain.xml' %(directory) , 'final', 1.5 , 5.e-5, 0.5, 1, 500000, erdp_coords_mountain,final_noise)
		
		print 'TopoGen for mountain completed in (s):',time.clock()-tstart

if __name__ == "__main__": main()