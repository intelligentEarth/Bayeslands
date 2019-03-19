
#!/bin/sh 
echo Running Bayes
 
 
for prob in 4
	do
		for s in 1000 500
			do
				python bl_mcmc.py -p $prob -s $s
				python bl_postproc.py -p $prob -f 1 -b 20
			done
	done 

echo Bash Succesfully executed

