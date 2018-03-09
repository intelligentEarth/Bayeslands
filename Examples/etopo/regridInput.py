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

# Import badlands grid generation toolbox
import pybadlands_companion.resizeInput as resize


def rescale():
	
	# Load python class and set required resolution
	newRes = resize.resizeInput(requestedSpacing = 500)

	# Regrid DEM file
	newRes.regridDEM(inDEM='data/res_etopo.csv',outDEM='data/newnodes.csv')

	# Regrid Rain file
	# newRes.regridRain(inRain='data/rain.csv',outRain='newrain.csv')
	
	# Regrid Tectonic files (Vertical only file)
	# newRes.regridTecto(inTec='data/disp.csv', outTec='newdisp.csv')
	
	# Regrid Tectonic files (3D displacement file)
	# newRes.regridDisp(inDisp='data/disp.csv', outDisp='newdisp.csv')

def main():
	rescale()

if __name__ == "__main__": main()