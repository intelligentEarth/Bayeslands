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

def main():
	folder_num = input("Enter folder number that you want the topography done for")
	Directory = "Topography/"
	Folder = "MC%s" % folder_num
	model = badlandsModel()
	model.load_xml('%s%s' %(Directory,Folder),'crater.xml')
	model.run_to_time(150000)

if __name__ == "__main__": main()