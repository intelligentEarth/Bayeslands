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

def OptGenerator(inputname, rain, erodibility, m, n, etime):

    model = badlandsModel()
    # Define the XmL input file
    model.load_xml('Opt',inputname, verbose = False, muted = True)

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
    np.savetxt('badlands.txt',mat,fmt='%.5f')

    return

def main():
    OptGenerator('crater.xml', 7, 5.e-4, 0.5, 1, 5000)

if __name__ == "__main__": main()