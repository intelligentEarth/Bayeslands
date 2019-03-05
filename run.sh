
#!/bin/sh 
echo Running Bayes
 
 
for prob in  1 2 
	do
		python bl_mcmc.py -p $prob -s 3000
		python bl_postproc.py -p $prob -f 1
	done 



 