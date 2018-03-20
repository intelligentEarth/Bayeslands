import numpy as np
import math
import time

def main():

	final = np.loadtxt('data/final.txt')
	mean_pred = np.loadtxt('mcmcresults_archive_2/mean_pred_elevation.txt')

	print 'final shape', final.shape
	print 'mean_pred', mean_pred.shape

	residual = final - mean_pred

	np.savetxt('data/sallyfinal.txt',final ,fmt='%.5f')
	np.savetxt('data/sallymean_pred.txt',mean_pred ,fmt='%.5f')
	np.savetxt('data/sallyfinal.csv',final ,fmt='%.5f')
	np.savetxt('data/sallymean_pred.csv',mean_pred ,fmt='%.5f')

	print 'residual matrix', residual
	print 'residual matrix shape', residual.shape
	print 'residual matrix sum', residual.sum()
	#print 'residual matrix sum', residual.sum(axis = 1)
	print 'residual matrix sum NP', np.sum(residual)

	np.savetxt('data/residual.txt', residual, fmt='%.5f')
	np.savetxt('data/residual.csv', residual, fmt='%.5f')

if __name__ == "__main__": main()
