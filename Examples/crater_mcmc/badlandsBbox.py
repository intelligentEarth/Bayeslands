##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
##                                                                                   ##
##  This file forms part of the Badlands surface processes modelling companion.      ##
##                                                                                   ##
##  For full license and copyright information, please refer to the LICENSE.md file  ##
##  located at the project root, or contact the authors.                             ##
##                                                                                   ##
##~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~#~##
"""
This script is intended to define badlands as a "black box" model for bayesian methods.
"""

import os
import math
import numpy as np

import time as tm
import matplotlib.pyplot as plt
import cmocean as cmo
from pylab import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.spatial import cKDTree
from pyBadlands.model import Model as badlandsModel

def plotElev(size=(8,8),elev=None,erodep=None):

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

def blackbox(inputname=None, rain=0., erodibility=0., etime=1e6):
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

    tstart = tm.clock()
    # Re-initialise badlands model
    model = badlandsModel()

    # Load the XmL input file
    model.load_xml(inputname)

    # Adjust erodibility based on given parameter
    model.input.SPLero = erodibility
    model.flow.erodibility.fill(erodibility)

    # Adjust precipitation values based on given parameter
    model.force.rainVal[:] = rain

    # Run badlands simulation
    model.run_to_time(etime)

    # Extract
    elev,erodep = interpolateArray(model.FVmesh.node_coords[:, :2],model.elevation,model.cumdiff)

    print 'Badlands black box model took (s):',tm.clock()-tstart

    return elev,erodep
