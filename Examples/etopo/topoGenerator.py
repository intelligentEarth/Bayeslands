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

from pyBadlands.model import Model as badlandsModel
import numpy as np
import cmocean as cmo
from pylab import rcParams
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from pyBadlands.model import Model as badlandsModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import time
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()
from plotly.offline.offline import _plot_html
    
def elevArray(coords=None):

    x, y, z = np.hsplit(coords, 3)        
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
    z_vals = z[indices][:,:,0]
    zi = np.average(z_vals,weights=(1./distances), axis=1)
    onIDs = np.where(distances[:,0] == 0)[0]
    if len(onIDs) > 0:
        zi[onIDs] = z[indices[onIDs,0],0]
    zreg = np.reshape(zi,(ny,nx))
    
    return zreg

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
    
    XYZ = np.column_stack((model.FVmesh.node_coords[:, :2],model.elevation))
    regz = elevArray(XYZ)
    mat=np.matrix(regz)
    
    np.savetxt('data/%s.txt' %(outputname),mat,fmt='%.5f')

    #viewGrid('Final', 'N/A', rreal, ereal, width=1000, height=1000, zmin=-10, zmax=600, zData=mat, title='Export Slope Grid')

    return

def inputVisualisation(inputname, outputname, rain, erodibility, m, n, etime = 0):
    
    model = badlandsModel()
    # Define the XmL input file
    model.load_xml('real',inputname, verbose = False, muted = True)

    rreal = rain
    ereal = erodibility

    topoGenerator(inputname, outputname, rain, erodibility, m, n, etime)

    elev = np.loadtxt('data/%s.txt' %(outputname))
    
    viewGrid('Initial', 'N/A', rreal, ereal, width=1000, height=1000, zmin=-10, zmax=600, zData=elev, title='Export Slope Grid')

    return

def viewGrid(sample_num, likl, rain, erod, width = 1600, height = 1600, zmin = None, zmax = None, zData = None, title='Export Grid'):
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
        title='etopo Elevation     rain = %s, erod = %s, likl = %s ' %( rain, erod, likl),
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

def main():

    choice = input('Would you like to: \n 1) Generate a simulated final topography \n 2) Visualise the initial topography \n ')

    if choice == 1:
        tstart = time.clock()
        topoGenerator('etopo.xml', 'final', 1, 3.e-6, 0.5, 1, 500000)
        print 'TopoGenerator model took (s):',time.clock()-tstart
    else:  
        inputVisualisation('etopo.xml','initial', 1, 3.e-6, 0.5, 1, 0)

if __name__ == "__main__": main()