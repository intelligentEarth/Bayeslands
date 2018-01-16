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


def plotElev(xyz=None):
    fig = plt.figure()
    rcParams['figure.figsize'] = (12,10)
    ax=plt.gca()
    im = ax.imshow(np.flipud(xyz),interpolation='nearest',cmap=cmo.cm.delta,
                   vmin=-100, vmax=800)
    plt.title('Elevation')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.2)

    plt.colorbar(im,cax=cax)
    plt.show()
    for z in xyz:
        all_z = ' '.join(str(z))
    print all_z
    
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


def OptGenerator(inputname, rain, erodibility, etime):

    model = badlandsModel()
    # Define the XmL input file
    model.load_xml('1234',inputname)
    
    #rreal=1
    #ereal=9.e-5

    rreal = rain
    ereal = erodibility

    model.input.SPLero = ereal
    model.flow.erodibility.fill(ereal)
    model.force.rainVal[:] = rreal   
    model.run_to_time(etime)
    
    XYZ = np.column_stack((model.FVmesh.node_coords[:, :2],model.elevation))
    regz = elevArray(XYZ)
    mat=np.matrix(regz)
    np.savetxt('crater.txt',mat,fmt='%.5f')

    return

def main():
    OptGenerator('crater.xml', 1, 9.e-5, 150000)

if __name__ == "__main__": main()