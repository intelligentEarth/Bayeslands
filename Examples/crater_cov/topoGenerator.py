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
import plotly.graph_objs as go


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

def topoGenerator(inputname, outputname, rain, erodibility, m, n, etime):

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
    
    elev,erodep = interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)

    elev_mat=np.matrix(elev)
    erodep_mat=np.matrix(erodep)

    np.savetxt('data/%s_elev.txt' %(outputname),elev_mat,fmt='%.5f')
    np.savetxt('data/%s_erodep.txt' %(outputname),erodep_mat,fmt='%.5f')

    viewGrid('%s_elev' %(outputname), 'N/A', rreal, ereal, width=1000, height=1000, zData=elev_mat, title='Export Slope Grid')
    viewMap('%s_erodep' %(outputname), 'N/A', rreal, ereal, width=1000, height=1000, zData=erodep_mat, title='Export Slope Grid')

    return

def inputVisualisation(inputname, outputname, rain, erodibility, m, n, etime = 0):
    
    model = badlandsModel()
    # Define the XmL input file
    model.load_xml('real',inputname, verbose = False, muted = True)

    rreal = rain
    ereal = erodibility

    topoGenerator(inputname, outputname, rreal, ereal, m, n, etime)

    elev = np.loadtxt('data/%s_elev.txt' %(outputname))
    erodep = np.loadtxt('data/%s_erodep.txt' %(outputname))
    
    viewGrid('%s_elev' %(outputname), 'N/A', rreal, ereal, width=1000, height=1000, zData=elev, title='Export Slope Grid')
    viewMap('%s_erodep' %(outputname), 'N/A', rreal, ereal, width=1000, height=1000, zData=erodep, title='Export Slope Grid')
    return

def viewGrid(sample_num, likl, rain, erod, width = 1000, height = 1000, zmin = None, zmax = None, zData = None, title='Export Grid'):
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
    graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='images/plot_image_%s.html' %(sample_num), validate=False)
    return    

def viewMap(sample_num, likl, rain, erod, width = 600, height = 600, zmin = None, zmax = None, zData = None, title='Export Grid'):
    
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
            zaxis=ZAxis(range=[-100, 100],autorange=False,nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
            xaxis=XAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
            yaxis=YAxis(nticks=10,gridcolor='rgb(255, 255, 255)',gridwidth=2,zerolinecolor='rgb(255, 255, 255)',zerolinewidth=2),
            bgcolor="rgb(244, 244, 248)"
        )
    )

    fig = Figure(data=data, layout=layout)
    graph = plotly.offline.plot(fig, auto_open=False, output_type='file', filename='images/plot_heatmap_%s.html' %(sample_num), validate=False)
    return    

def main():

    choice = input('Would you like to: \n 1) Generate a simulated final topography \n 2) Visualise the initial topography \n ')

    if choice == 1:
        tstart = time.clock()
        topoGenerator('crater.xml', 'final', 1.5 , 5.e-5, 0.5, 1, 15000)
        print 'TopoGenerator model took (s):',time.clock()-tstart
        
    else:  
        inputVisualisation('crater.xml','initial', 1.5 , 5.e-5, 0.5, 1, 0)

if __name__ == "__main__": main()