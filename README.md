BayesLands - An MCMC implementation of pyBadlands
=====
    
<div align="center">
    <img width=1000 src="https://github.com/badlands-model/BayesLands/Examples/basin/images/elev_500kyr.png" alt="flowchart mcmc" title="Flowchart for MCMC scheme with Badlands"</img>
</div>

[![DOI](https://zenodo.org/badge/51286954.svg)](https://zenodo.org/badge/latestdoi/51286954)

## Overview
**BayesLands**, a Bayesian framework for [**Badlands**](https://github.com/badlands-model/pyBadlands) that fuses information obtained from complex forward models with observational data and prior knowledge. As a proof-of-concept, we consider a synthetic and real-world topography with two free  parameters, namely precipitation and erodibility, that we need to estimate through BayesLands. The results of the experiments shows that BayesLands yields a promising distribution of the  parameters. Moreover, the challenge in sampling due to multi-modality is presented through visualizing a likelihood surface that has a  range of suboptimal modes.

[Badlands overview](https://prezi.com/5y1usorz8e8k/badlands-overview/?utm_campaign=share&utm_medium=copy) - Basin Genesis Hub presentation (2017)

<div align="center">
    <img width=500 src="https://github.com/badlands-model/BayesLands/Examples/basin/images/elev_500kyr.png" alt="sketch Badlands" title="sketch of Badlands range of models."</img>
    <img width=500 src="https://github.com/badlands-model/BayesLands/Examples/basin/images/elev_500kyr.png" alt="sketch Badlands" title="sketch of Badlands range of models."</img>
</div>


### Usage Instructions

Installation: 
+ Git clone https://github.com/badlands-model/BayesLands.git
+ Stepwise instructions to install BayesLands and it's prerequisite python packages are provided in the **installation.txt** file.


+ bl_mcmc.py - File that executes an mcmc chain.

+ bl_preproc - File includes functions to crop rescale or edit input topographies to be used in the model.

+ bl_postproc - File used to produce figures for posterior distributions of the free parameters and time variant erosion deposition.

+ bl_surflikl - File used to generate the likelihood surface of the free parameters.

+ bl_topogenr - File used to generate the input and final-time topography used by the mcmc file. 

### Community driven

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program.  If not, see <http://www.gnu.org/licenses/lgpl-3.0.en.html>.

### Reporting  

If you come accross a bug or if you need some help compiling or using the code you can drop us a line at:
	- danial.azam@sydney.edu.au
	- rohitash.chandra@sydney.edu.au

### Documentation related to Badlands physics & assumptions

+ **Salles, T. & Hardiman, L.: [Badlands: An open-source, flexible and parallel framework to study landscape dynamics](http://dx.doi.org/10.1016/j.cageo.2016.03.011), Computers & Geosciences, 91, 77-89, doi:10.1016/j.cageo.2016.03.011, 2016.**

+ **Salles, T.: [Badlands: A parallel basin and landscape dynamics model](http://dx.doi.org/10.1016/j.softx.2016.08.005), SoftwareX, 5, 195–202, doi:10.1016/j.softx.2016.08.005, 2016.**

+ **Salles, T., Ding, X. and Brocard, G.: [pyBadlands: A framework to simulate sediment transport, landscape dynamics and basin stratigraphic evolution through space and time](https://doi.org/10.1371/journal.pone.0195557), PLOS ONE 13(4): e0195557, 2018.** 

### Other published research studies using Badlands:

+ **Salles, T., N. Flament, and D. Muller: [Influence of mantle flow on the drainage of eastern Australia since the Jurassic Period](http://dx.doi.org/10.1002/2016GC006617), Geochem. Geophys. Geosyst., 18, doi:10.1002/2016GC006617, 2017** -- [Supplementary materials: Australian Landscape Dynamic](https://github.com/badlands-model/g-cubed-2016)

+ **Salles, T., X. Ding, J.M. Webster, A. Vila-Concejo, G. Brocard and J. Pall: [A unified framework for modelling sediment fate from source to sink and its interactions with reef systems over geological times](https://doi.org/10.1038/s41598-018-23519-8), Nature Scientific Report, doi:10.1038/s41598-018-23519-8, 2018** 

When you use **Badlands** or **BayesLands**, please cite the above papers.
