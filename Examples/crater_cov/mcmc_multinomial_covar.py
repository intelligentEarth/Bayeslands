# !/usr/bin/python
# BayesReef: a MCMC random walk method applied to pyReef-Core
# Authors: Jodie Pall and Danial Azam (2017)
# Adapted from: [Chandra_ICONIP2017] R. Chandra, L. Azizi, S. Cripps, 'Bayesian neural learning via Langevin dynamicsfor chaotic time series prediction', ICONIP 2017.
# (to be addeded on https://www.researchgate.net/profile/Rohitash_Chandra)

import os
import math
import time
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from pyReefCore.model import Model
import fnmatch
import matplotlib as mpl
from cycler import cycler
from scipy import stats 

cmap=plt.cm.Set2
c = cycler('color', cmap(np.linspace(0,1,8)) )
plt.rcParams["axes.prop_cycle"] = c

class MCMC():
    def __init__(self, simtime, samples, communities, core_data, core_depths,data_vec, timestep,filename, xmlinput, sedsim, sedlim, flowsim, flowlim, max_a, max_m, vis, covar):
        self.filename = filename
        self.input = xmlinput
        self.communities = communities
        self.samples = samples       
        self.core_data = core_data
        self.core_depths = core_depths
        self.data_vec = data_vec
        self.timestep = timestep
        self.vis = vis
        self.sedsim = sedsim
        self.flowsim = flowsim
        self.sedlim = sedlim
        self.flowlim = flowlim
        self.max_a = max_a
        self.max_m = max_m
        self.simtime = simtime
        self.font = 10
        self.width = 1
        self.d_sedprop = float(np.count_nonzero(core_data[:,self.communities]))/core_data.shape[0]
        self.initial_sed = []
        self.initial_flow = []
        self.step_m = 0.05 
        self.step_a = 0.002  
        self.step_sed = 0.0001 
        self.step_flow = 0.0015
        self.true_m = 0.086
        self.true_ax = -0.01
        self.true_ay = -0.03
        self.covar_matrix = covar


    def run_Model(self, reef, input_vector):
        reef.convert_vector(self.communities, input_vector, self.sedsim, self.flowsim) #model.py
        self.initial_sed, self.initial_flow = reef.load_xml(self.input, self.sedsim, self.flowsim)
        if self.vis[0] == True:
            reef.core.initialSetting(size=(8,2.5), size2=(8,3.5)) # View initial parameters
        reef.run_to_time(self.simtime,showtime=100.)
        if self.vis[1] == True:
            from matplotlib.cm import terrain, plasma
            nbcolors = len(reef.core.coralH)+10
            colors = terrain(np.linspace(0, 1.8, nbcolors))
            nbcolors = len(reef.core.layTime)+3
            colors2 = plasma(np.linspace(0, 1, nbcolors))
            reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300)
        output_core = reef.plot.core_timetodepth(self.communities, self.core_depths) #modelPlot.py
        # predicted_core = reef.convert_core(self.communities, output_core, self.core_depths) #model.py
        # return predicted_core
        print 'output_core_shape', output_core.shape 
        return output_core

    def plot_results(self, pos_m, pos_ax, pos_ay, pos_sed1, pos_sed2, pos_sed3, pos_sed4, pos_flow1, pos_flow2, pos_flow3, pos_flow4, burn):
        nb_bins=30
        slen = np.arange(0,pos_m.shape[0],1)

        #   MALTHUS PARAMETER   #
        mmin, mmax = min(pos_m), max(pos_m)
        mspace = np.linspace(mmin,mmax,len(pos_m))
        mm,ms = stats.norm.fit(pos_m)
        pdf_m = stats.norm.pdf(mspace,mm,ms)
        mmean=np.mean(pos_m)
        mmedian=np.median(pos_m)
        mmode,count=stats.mode(pos_m)
    
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title(' Malthusian Parameter', fontsize= self.font+2)#, y=1.02)
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor('#f2f2f3')
        ax1.hist(pos_m, bins=25, alpha=0.5, facecolor='sandybrown', normed=True)
        ax1.axvline(self.true_m, linestyle='-', color='black', linewidth=1,label='True value')
        # ax1.axvline(mm,linestyle='-', color='black', linewidth=1,label='Mean')
        # ax1.axvline(mm+ms,linestyle='--', color='black', linewidth=1,label='5th and 95th %ile')
        # ax1.axvline(mm-ms,linestyle='--', color='black', linewidth=1,label=None)
        #ax1.axvline(mmode,linestyle='-', color='orangered', linewidth=1,label=None)

        # ax1.plot(mspace,pdf_m,label='Best fit',color='orangered',linestyle='--')
        ax1.grid(True)
        ax1.set_ylabel('Frequency',size=self.font+1)
        ax1.set_xlabel(r'$\varepsilon$', size=self.font+1)
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor('#f2f2f3')
        ax2.plot(slen,pos_m,linestyle='-', linewidth=self.width, color='k', label=None)
        ax2.set_title(r'Trace of $\varepsilon$',size=self.font+2)
        ax2.set_xlabel('Samples',size=self.font+1)
        ax2.set_ylabel(r'$\varepsilon$', size=self.font+1)
        ax2.set_xlim([0,np.amax(slen)])
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig('%s/malthus.png'% (self.filename), bbox_inches='tight', dpi=300, transparent=False)
        plt.clf()
        
        #    COMMUNITY MATRIX   #
        a1min, a1max = min(pos_ax), max(pos_ax)
        a1space = np.linspace(a1min,a1max,len(pos_ax))
        a1m,a1s = stats.norm.fit(pos_ax)
        pdf_a1 = stats.norm.pdf(a1space,a1m,a1s)
        a2min, a2max = min(pos_ay), max(pos_ay)
        a2space = np.linspace(a2min,a2max,len(pos_ay))
        a2m,a2s = stats.norm.fit(pos_ay)
        pdf_a2 = stats.norm.pdf(a2space,a2m,a2s)
        a1min=a1min
        a1max=a1max
        a1mean=np.mean(pos_ax)
        a1median=np.median(pos_ax)
        a1mode,count=stats.mode(pos_ax)
        a2min=a2min
        a2max=a2max
        a2mean=np.mean(pos_ay)
        a2median=np.median(pos_ay)
        a2mode,count=stats.mode(pos_ay)

        ####   main diagonal   
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title('Community Interaction Matrix Parameters', fontsize= self.font+2, y=1.03)
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor('#f2f2f3')
        ax1.hist(pos_ax, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
        ax1.axvline(self.true_ax, linestyle='-', color='black', linewidth=1,label='True value')
        # ax1.plot(a1space,pdf_a1,label='Best fit',color='orangered',linestyle='--')
        # ax1.axvline(a1m,linestyle='-', color='black', linewidth=1,label='Mean')
        # ax1.axvline(a1m+a1s,linestyle='--', color='black', linewidth=1,label='5th and 95th %ile')
        # ax1.axvline(a1m-a1s,linestyle='--', color='black', linewidth=1,label=None)
        #ax1.axvline(a1mode,linestyle='-', color='orangered', linewidth=1,label=None)
        ax1.grid(True)
        ax1.set_ylabel('Frequency',size=self.font+1)
        ax1.set_title(r'Main diagonal value ($\alpha_{ii}$)',size=self.font+2)
        ax1.set_xlabel(r'$\alpha_{ii}$', size=self.font+1)
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor('#f2f2f3')
        ax2.plot(slen,pos_ax,linestyle='-', linewidth=self.width, color='k', label=None)
        ax2.set_xlabel('Samples',size=self.font+1)
        ax2.set_ylabel(r'$\alpha_{ii}$', size=self.font+1)
        ax2.set_title(r'Trace of $\alpha_{ii}$',size=self.font+2)
        ax2.set_xlim([0,np.amax(slen)])
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig('%s/comm_ax.png'% (self.filename),bbox_inches='tight', dpi=300,transparent=False)
        plt.clf()

        ####   sub- and super-diagonal  
        fig = plt.figure(figsize=(6,8))
        ax = fig.add_subplot(111)
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
        ax.set_title('Community Interaction Matrix Parameters', fontsize= self.font+2, y=1.03)
        ax1 = fig.add_subplot(211)
        ax1.set_facecolor('#f2f2f3')
        ax1.hist(pos_ay, bins=25, alpha=0.5, facecolor='mediumaquamarine', normed=True)
        ax1.axvline(self.true_ay, linestyle='-', color='black', linewidth=1,label='True value')
        # ax1.axvline(a2m,linestyle='-', color='black', linewidth=1,label='Mean')
        # ax1.axvline(a2m+a2s,linestyle='--', color='black', linewidth=1,label='5th and 95th %ile')
        # ax1.axvline(a2m-a2s,linestyle='--', color='black', linewidth=1,label=None)
        #ax1.axvline(a2mode,linestyle='-', color='orangered', linewidth=1,label=None)

        # ax1.plot(a2space,pdf_a2,label='Best fit',color='orangered',linestyle='--')
        ax1.grid(True)
        ax1.set_title(r'Super- and sub-diagonal values ($\alpha_{i,i+1}$ and $\alpha_{i+1,i}$)',size=self.font+2)
        ax1.set_xlabel(r'$\alpha_{i,i+1}$ and $\alpha_{i+1,i}$', size=self.font+1)
        ax1.set_ylabel('Frequency',size=self.font+1)
        ax2 = fig.add_subplot(212)
        ax2.set_facecolor('#f2f2f3')
        ax2.plot(slen,pos_ay,linestyle='-', linewidth=self.width, color='k', label=None)
        ax2.set_title(r'Trace of $\alpha_{i,i+1}$ and $\alpha_{i+1,i}$',size=self.font+2)
        ax2.set_xlabel('Samples',size=self.font+1)
        ax2.set_ylabel(r'$\alpha_{i,i+1}$ and $\alpha_{i+1,i}$', size=self.font+1)
        ax2.set_xlim([0,np.amax(slen)])
        fig.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.savefig('%s/comm_ay.png' % (self.filename), dpi=300, bbox_inches='tight',transparent=False)
        plt.clf()

        if not os.path.isfile(('%s/summ_stats.txt' % (self.filename))):
            with file(('%s/summ_stats.txt' % (self.filename)),'w') as outfile:
                outfile.write('SUMMARY STATISTICS\n')
                outfile.write('MIN, MAX, MEAN, MEDIAN, MODE\n')
                outfile.write('Malthusian parameter\n{0}, {1}, {2}, {3}, \n{4}\n'.format(mmin,mmax,mmean,mmedian,mmode))
                outfile.write('Main diagonal\n{0}, {1}, {2}, {3}, \n{4}\n'.format(a1min,a1max,a1mean,a1median,a1mode))
                outfile.write('Super-/Sub-diagonal\n{0}, {1}, {2}, {3}, \n{4}\n'.format(a2min,a2max,a2mean,a2median,a2mode))


        # PLOT SEDIMENT AND FLOW RESPONSE THRESHOLDS #
        a_labels = ['Shallow windward', 'Moderate-deep windward', 'Deep windward']#, 'Shallow leeward', 'Moderate-deep leeward', 'Deep leeward']
        
        sed1_mu, sed1_ub, sed1_lb, sed2_mu, sed2_ub, sed2_lb, sed3_mu, sed3_ub, sed3_lb, sed4_mu, sed4_ub, sed4_lb = (np.zeros(self.communities) for i in range(12))
        if ((self.sedsim != False)):
            for a in range(self.communities):
                sed1_mu[a] = np.mean(pos_sed1[:,a])
                sed1_ub[a] = np.percentile(pos_sed1[:,a], 95, axis=0)
                sed1_lb[a] = np.percentile(pos_sed1[:,a], 5, axis=0)
                
                sed2_mu[a] = np.mean(pos_sed2[:,a])
                sed2_ub[a] = np.percentile(pos_sed2[:,a], 95, axis=0)
                sed2_lb[a] = np.percentile(pos_sed2[:,a], 5, axis=0)
                
                sed3_mu[a] = np.mean(pos_sed3[:,a])
                sed3_ub[a] = np.percentile(pos_sed3[:,a], 95, axis=0)
                sed3_lb[a] = np.percentile(pos_sed3[:,a], 5, axis=0)
                
                sed4_mu[a] = np.mean(pos_sed4[:,a])
                sed4_ub[a] = np.percentile(pos_sed4[:,a], 95, axis=0)
                sed4_lb[a] = np.percentile(pos_sed4[:,a], 5, axis=0)

                sed1_mu_=sed1_mu[a]
                sed2_mu_=sed2_mu[a]
                sed3_mu_=sed3_mu[a]
                sed4_mu_=sed4_mu[a]
                sed1_min=sed1_lb[a]
                sed2_min=sed2_lb[a]
                sed3_min=sed3_lb[a]
                sed4_min=sed4_lb[a]
                sed1_max=sed1_ub[a]
                sed2_max=sed2_ub[a]
                sed3_max=sed3_ub[a]
                sed4_max=sed4_ub[a]
                sed1_med=np.median(pos_sed1[:,a])
                sed2_med=np.median(pos_sed2[:,a])
                sed3_med=np.median(pos_sed3[:,a])
                sed4_med=np.median(pos_sed4[:,a])
                sed1_mode,count=stats.mode(pos_sed1[:,a])
                sed2_mode,count=stats.mode(pos_sed2[:,a])
                sed3_mode,count=stats.mode(pos_sed3[:,a])
                sed4_mode,count=stats.mode(pos_sed4[:,a])


                with file(('%s/summ_stats.txt' % (self.filename)),'a') as outfile:
                    outfile.write('\n# Sediment threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('Sed1\n{0}, {1}, {2}, {3}\n'.format(sed1_min,sed1_max,sed1_mu_,sed1_med))
                    outfile.write('Sed2\n{0}, {1}, {2}, {3}\n'.format(sed2_min,sed2_max,sed2_mu_,sed2_med))
                    outfile.write('Sed3\n{0}, {1}, {2}, {3}\n'.format(sed3_min,sed3_max,sed3_mu_,sed3_med))
                    outfile.write('Sed4\n{0}, {1}, {2}, {3}\n'.format(sed4_min,sed4_max,sed4_mu_,sed4_med))
                    outfile.write('Modes\n\tSed1:\t{0}\n\tSed2:\t{1}\n\tSed3:\t{2}\n\tSed4:\t{3}'.format(sed1_mode,sed2_mode,sed3_mode,sed4_mode))

                cy = [0,100,100,0]
                cmu = [sed1_mu[a], sed2_mu[a], sed3_mu[a], sed4_mu[a]]
                c_lb = [sed1_mu[a]-sed1_lb[a], sed2_mu[a]-sed2_lb[a], sed3_mu[a]-sed3_lb[a], sed4_mu[a]-sed4_lb[a]]
                c_ub = [sed1_ub[a]-sed1_mu[a], sed2_ub[a]-sed2_mu[a], sed3_ub[a]-sed3_mu[a], sed4_ub[a]-sed4_mu[a]]
                
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                ax.plot(self.initial_sed[a,:], cy, linestyle='--', linewidth=self.width, marker='.',color='k', label='Synthetic data')
                ax.plot(cmu, cy, linestyle='-', linewidth=self.width,marker='.', color='sandybrown', label='Mean')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1, color='darksalmon',mfc='darksalmon',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1, color='sienna',mfc='sienna',fmt='.',label=None)
                plt.title('Sediment exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=self.font+2, y=1.06)
                plt.ylabel('Proportion of maximum growth rate [%]',size=self.font+1)
                plt.xlabel('Sediment input [m/year]',size=self.font+1)
                plt.ylim(-2.,110)
                lgd = plt.legend(frameon=False, prop={'size':self.font+1}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/sediment_response_%s.png' % (self.filename, a+1), bbox_extra_artists=(lgd,),bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

        flow1_mu, flow1_ub,flow1_lb, flow2_mu, flow2_ub,flow2_lb, flow3_mu, flow3_ub,flow3_lb, flow4_mu, flow4_ub,flow4_lb = (np.zeros(self.communities) for i in range(12))
        if (self.flowsim != False):
            for a in range(self.communities):
                flow1_mu[a] = np.mean(pos_flow1[:,a])
                flow1_ub[a] = np.percentile(pos_flow1[:,a], 95, axis=0)
                flow1_lb[a] = np.percentile(pos_flow1[:,a], 5, axis=0)
                
                flow2_mu[a] = np.mean(pos_flow2[:,a])
                flow2_ub[a] = np.percentile(pos_flow2[:,a], 95, axis=0)
                flow2_lb[a] = np.percentile(pos_flow2[:,a], 5, axis=0)
                
                flow3_mu[a] = np.mean(pos_flow3[:,a])
                flow3_ub[a] = np.percentile(pos_flow3[:,a], 95, axis=0)
                flow3_lb[a] = np.percentile(pos_flow3[:,a], 5, axis=0)
                
                flow4_mu[a] = np.mean(pos_flow4[:,a])
                flow4_ub[a] = np.percentile(pos_flow4[:,a], 95, axis=0)
                flow4_lb[a] = np.percentile(pos_flow4[:,a], 5, axis=0)

                flow1_mu_ = flow1_mu[a]
                flow2_mu_ = flow2_mu[a]
                flow3_mu_ = flow3_mu[a]
                flow4_mu_ = flow4_mu[a]
                flow1_min= flow1_lb[a]
                flow1_max=flow1_ub[a]
                flow1_med=np.median(pos_flow1[:,a])
                flow2_min=flow2_lb[a]
                flow2_max=flow2_ub[a]
                flow2_med=np.median(pos_flow2[:,a])
                flow3_min=flow3_lb[a]
                flow3_max=flow3_ub[a]
                flow3_med=np.median(pos_flow3[:,a])
                flow4_min=flow4_lb[a]
                flow4_max=flow4_ub[a]
                flow4_med=np.median(pos_flow4[:,a])
                flow1_mode,count= stats.mode(pos_flow1[:,a])
                flow2_mode,count= stats.mode(pos_flow2[:,a])
                flow3_mode,count= stats.mode(pos_flow3[:,a])
                flow4_mode,count= stats.mode(pos_flow4[:,a])

                with file(('%s/summ_stats.txt' % (self.filename)),'a') as outfile:
                    outfile.write('\n# Water flow threshold: {0}\n'.format(a_labels[a]))
                    outfile.write('#5TH %ILE, 95TH %ILE, MEAN, MEDIAN\n')
                    outfile.write('# flow1\n{0}, {1}, {2}, {3}\n'.format(flow1_min,flow1_max,flow1_mu_,flow1_med))
                    outfile.write('# flow2\n{0}, {1}, {2}, {3}\n'.format(flow2_min,flow2_max,flow2_mu_,flow2_med))
                    outfile.write('# flow3\n{0}, {1}, {2}, {3}\n'.format(flow3_min,flow3_max,flow3_mu_,flow3_med))
                    outfile.write('# flow4\n{0}, {1}, {2}, {3}\n'.format(flow4_min,flow4_max,flow4_mu_,flow4_med))
                    outfile.write('Modes\n\tFlow1:\t{0}\n\tFlow2:\t{1}\n\tFlow3:\t{2}\n\tFlow4:\t{3}'.format(flow1_mode,flow2_mode,flow3_mode,flow4_mode))

                cy = [0,100,100,0]
                cmu = [flow1_mu[a], flow2_mu[a], flow3_mu[a], flow4_mu[a]]
                c_lb = [flow1_mu[a]-flow1_lb[a], flow2_mu[a]-flow2_lb[a], flow3_mu[a]-flow3_lb[a], flow4_mu[a]-flow4_lb[a]]
                c_ub = [flow1_ub[a]-flow1_mu[a], flow2_ub[a]-flow2_mu[a], flow3_ub[a]-flow3_mu[a], flow4_ub[a]-flow4_mu[a]]

                
                fig = plt.figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                ax.set_facecolor('#f2f2f3')
                ax.plot(self.initial_flow[a,:], cy, linestyle='--', linewidth=self.width, marker='.', color='k',label='Synthetic data')
                ax.plot(cmu, cy, linestyle='-', linewidth=self.width, marker='.', color='steelblue', label='Mean')
                ax.errorbar(cmu[0:2],cy[0:2],xerr=[c_lb[0:2],c_ub[0:2]],capsize=5,elinewidth=1,color='lightsteelblue',mfc='lightsteelblue',fmt='.',label=None)
                ax.errorbar(cmu[2:4],cy[2:4],xerr=[c_lb[2:4],c_ub[2:4]],capsize=5,elinewidth=1,color='lightslategrey',mfc='lightslategrey',fmt='.',label=None)
                plt.title('Hydrodynamic energy exposure threshold function\n(%s assemblage)' % (a_labels[a]), size=self.font+2, y=1.06)
                plt.ylabel('Proportion of maximum growth rate [%]', size=self.font+1)
                plt.xlabel('Fluid flow [m/sec]', size=self.font+1)
                plt.ylim(-2.,110.)
                lgd = plt.legend(frameon=False, prop={'size':self.font+1}, bbox_to_anchor = (1.,0.2))
                plt.savefig('%s/flow_response_%s.png' % (self.filename, a+1),  bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
                plt.clf()

    def save_params(self,naccept, pos_diff, pos_samples, proposal):    ### SAVE RECORD OF ACCEPTED PARAMETERS ###

        if not os.path.isfile(('%s/accept_diff.csv' % (self.filename))):
            with file(('%s/accept_diff.csv' % (self.filename)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,pos_diff]
                writer.writerow(data)
        else:
            with file(('%s/accept_diff.csv' % (self.filename)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,pos_diff]
                writer.writerow(data)

        # Save accepted samples
        if not os.path.isfile(('%s/accept_samples.csv' % (self.filename))):
            with file(('%s/accept_samples.csv' % (self.filename)),'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_samples)]
                writer.writerow(data)
        else:
            with file(('%s/accept_samples.csv' % (self.filename)),'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                data = [naccept,np.ndarray.tolist(pos_samples)]
                writer.writerow(data)
        # Save accepted proposals
        if not os.path.isfile('%s/accepted_proposals.csv' % (self.filename)):
            with file(('%s/accepted_proposals.csv' % (self.filename)), 'wb') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(proposal)
        else:
            with file(('%s/accepted_proposals.csv' % (self.filename)), 'ab') as outfile:
                writer = csv.writer(outfile, delimiter=',')
                writer.writerow(proposal)

    def convert_core_format(self, core, communities):
        vec = np.zeros(core.shape[0])
        for n in range(len(vec)):
            idx = np.argmax(core[n,:])# get index,
            vec[n] = idx+1 # +1 so that zero is preserved as 'none'
        return vec

    def diff_score(self, z,intervals):
        same= np.count_nonzero(z)
        same = float(same)/intervals
        diff = 1-same
        print 'diff:', diff
        return diff*100

    def rmse(self, sim, obs):
        # where there is 1 in the sed column, count
        sed = np.count_nonzero(sim[:,self.communities])
        p_sedprop = (float(sed)/sim.shape[0])
        sedprop = np.absolute((self.d_sedprop - p_sedprop)*0.5)
        rmse =(np.sqrt(((sim - obs) ** 2).mean()))*0.5
        
        return rmse + sedprop

    def likelihood_func(self, reef, core_data, input_v):
        pred_core = self.run_Model(reef, input_v)
        pred_core = pred_core.T
        intervals = pred_core.shape[0]
        z = np.zeros((intervals,self.communities+1))    
        for n in range(intervals):
            idx_data = np.argmax(core_data[n,:])
            idx_model = np.argmax(pred_core[n,:])
            if ((pred_core[n,self.communities] != 1.) and (idx_data == idx_model)): #where sediment !=1 and max proportions are equal:
                z[n,idx_data] = 1
        diff = self.diff_score(z,intervals)
        # rmse = self.rmse(pred_core, core_data)
        
        z = z + 0.1
        z = z/(1+(1+self.communities)*0.1)
        loss = np.log(z)
        # print 'sum of loss:', np.sum(loss)        
        return [np.sum(loss), pred_core, diff]
           
    def save_core(self,reef,naccept):
        path = '%s/%s' % (self.filename, naccept)
        if not os.path.exists(path):
            os.makedirs(path)
        
        #     Initial settings     #
        reef.core.initialSetting(size=(8,2.5), size2=(8,4.5), dpi=300, fname='%s/a_thres_%s_' % (path, naccept))
        from matplotlib.cm import terrain, plasma
        nbcolors = len(reef.core.coralH)+10
        colors = terrain(np.linspace(0, 1.8, nbcolors))
        nbcolors = len(reef.core.layTime)+3
        colors2 = plasma(np.linspace(0, 1, nbcolors))
        
        #      Community population evolution    #
        reef.plot.speciesDepth(colors=colors, size=(8,4), font=8, dpi=300, fname =('%s/b_popd_%s.png' % (path,naccept)))
        reef.plot.speciesTime(colors=colors, size=(8,4), font=8, dpi=300,fname=('%s/c_popt_%s.png' % (path,naccept)))
        reef.plot.accomodationTime(size=(8,4), font=8, dpi=300, fname =('%s/d_acct_%s.pdf' % (path,naccept)))
        
        #      Draw core      #
        reef.plot.drawCore(lwidth = 3, colsed=colors, coltime = colors2, size=(9,8), font=8, dpi=300, 
                           figname=('%s/e_core_%s' % (path, naccept)), filename=('%s/core_%s.csv' % (path, naccept)), sep='\t')
        return

    def proposal_jump(self, current, low_limit, high_limit, jump_width):
        proposal = current + np.random.normal(0, jump_width)
        if proposal >= high_limit:
            proposal = current
        elif proposal <= low_limit:
            proposal = current

        # while lim_condition:
        #     if proposal >= high_limit:
        #         proposal = current + np.random.normal(0, jump_width)
        #     elif proposal <= low_limit:
        #         proposal = current + np.random.normal(0, jump_width)
        #     else:
        #         lim_condition = False
        return proposal

    def sampler(self):
        data_size = self.core_data.shape[0]
        samples = self.samples
        x_data = self.core_depths
        y_data = self.core_data

        with file(('%s/description.txt' % (self.filename)),'a') as outfile:
            outfile.write('\n\tstep_m: {0}'.format(self.step_m))
            outfile.write('\n\tstep_a: {0}'.format(self.step_a))
            outfile.write('\n\tstep_sed: {0}'.format(self.step_sed))
            outfile.write('\n\tstep_flow: {0}'.format(self.step_flow))

        # Create space to store accepted samples for posterior 
        pos_sed1 = np.zeros((samples , self.communities)) # sample rows, self.communities column
        pos_sed2 = np.zeros((samples , self.communities)) 
        pos_sed3 = np.zeros((samples , self.communities))
        pos_sed4 = np.zeros((samples , self.communities))
        pos_flow1 = np.zeros((samples , self.communities))
        pos_flow2 = np.zeros((samples , self.communities))
        pos_flow3 = np.zeros((samples , self.communities))
        pos_flow4 = np.zeros((samples , self.communities))
        pos_ax = np.zeros(samples)
        pos_ay = np.zeros(samples)
        pos_m = np.zeros(samples)
        # Create space to store fx of all samples
        pos_samples = np.zeros((samples, self.core_data.shape[0]))
        
        #      INITIAL PREDICTION       #
        sed1 = np.zeros(self.communities)
        sed2 = np.zeros(self.communities)
        sed3 = np.zeros(self.communities)
        sed4 = np.zeros(self.communities)

        if self.sedsim == True:
            for s in range(self.communities):
                sed1[s] = pos_sed1[0,s] = np.random.uniform(0.,0.)
                sed2[s] = pos_sed2[0,s] = np.random.uniform(0.,0.)
                sed3[s] = pos_sed3[0,s] = np.random.uniform(0.005,0.005)
                sed4[s] = pos_sed4[0,s] = np.random.uniform(0.005,0.005)

        flow1 = np.zeros(self.communities)
        flow2 = np.zeros(self.communities)
        flow3 = np.zeros(self.communities)
        flow4 = np.zeros(self.communities)

        if self.flowsim == True:
            for s in range(self.communities):
                #     relaxed constraints 
                flow1[s] = pos_flow1[0,s] = np.random.uniform(0.,0.)
                flow2[s] = pos_flow2[0,s] = np.random.uniform(0.,0.)
                flow3[s] = pos_flow3[0,s] = np.random.uniform(0.3,0.3)
                flow4[s] = pos_flow4[0,s] = np.random.uniform(0.3,0.3)
        
        cm_ax = pos_ax[0] = np.random.uniform(self.max_a,0.)
        cm_ay = pos_ay[0] = np.random.uniform(self.max_a,0.)
        m = pos_m[0] = np.random.uniform(0., self.max_m)

        if (self.sedsim == True) and (self.flowsim == False):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4))
        elif (self.flowsim == True) and (self.sedsim == False):
            v_proposal = np.concatenate((flow1,flow2,flow3,flow4))
        elif (self.sedsim == True) and (self.flowsim == True):
            v_proposal = np.concatenate((sed1,sed2,sed3,sed4,flow1,flow2,flow3,flow4))
        v_proposal = np.append(v_proposal,(cm_ax,cm_ay,m))
        pos_v = np.zeros((samples, v_proposal.size))
        print v_proposal

        # Declare pyReef-Core and initialize
        reef = Model()

        [likelihood, pred_data, diff] = self.likelihood_func(reef, self.core_data, v_proposal)
        pos_diff = np.full(samples,diff)
        data_vec = self.convert_core_format(self.core_data, self.communities)
        data_vec_ = str(data_vec)
        with file(('%s/core_data_vec.txt' % (self.filename)),'w') as outfile:
            outfile.write(data_vec_)
            # outfile.write('\n\tstep_m: {0}'.format(self.step_m))
        core_vec = self.convert_core_format(pred_data, self.communities)
        pos_samples[0,:] = core_vec
        print '\tinitial likelihood:', likelihood, 'and difference score:', diff

        naccept = 0
        count_list = []
        count_list.append(0)
        self.save_core(reef, 'initial')
        self.save_params(naccept, pos_diff[0], pos_samples[0,],pos_v[0,])
        
        # print 'Begin sampling using MCMC random walk'
        x_tick_labels = ['No growth','Shallow', 'Mod-deep', 'Deep', 'Sediment']
        x_tick_values = [0,1,2,3,4]
        fig = plt.figure(figsize=(3,6))
        ax = fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        ax.plot(data_vec, x_data, label='Synthetic core', color='k')
        ax.plot(core_vec, x_data, label='Initial predicted core')
        ax.set_title("Data vs Initial Prediction", size=self.font+2)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=self.font+1)
        ax.set_ylabel("Core depth [m]",size=self.font+1)
        ax.set_ylim([0,np.amax(self.core_depths)])
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.legend(frameon=False, prop={'size':self.font+1},bbox_to_anchor = (1.,0.1))
        fig.savefig('%s/begin.png' % (self.filename), bbox_inches='tight',dpi=300,transparent=False)
        plt.clf()
        
        # ACCUMULATED FIGURE SET UP
        final_fig = plt.figure(figsize=(3,6))
        ax_append = final_fig.add_subplot(111)
        ax_append.set_facecolor('#f2f2f3')
        ax_append.plot(data_vec, x_data, label='Synthetic core', color='k')
        ax_append.plot(core_vec, x_data)
        ax_append.set_title("Accepted Proposals", size=self.font+2)
        plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=self.font+1)
        ax_append.set_ylabel("Depth [m]",size=self.font+1)
        ax_append.set_ylim([0,np.amax(self.core_depths)])
        ax_append.set_ylim(ax_append.get_ylim()[::-1])


        for i in range(samples - 1):
            print '\nSample: ', i
            start = time.time()

            # if self.sedsim == True:
            #     tmat = np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,self.communities)
            #     tmatrix = tmat.T
            #     t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
            #     for x in range(self.communities):
            #         for s in range(tmatrix.shape[1]):
            #             t2matrix[x,s] = self.proposal_jump(tmatrix[x,s], self.sedlim[0], self.sedlim[1], self.step_sed)
            #             # t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,self.step_sed)
            #             # if t2matrix[x,s] >= self.sedlimits[x,1]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #             # elif t2matrix[x,s] <= self.sedlimits[x,0]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #     # reorder each row , then transpose back as sed1, etc.
            #     tmp = np.zeros((self.communities,4))
            #     for x in range(t2matrix.shape[0]):
            #         a = np.sort(t2matrix[x,:])
            #         tmp[x,:] = a
            #     tmat = tmp.T
            #     p_sed1 = tmat[0,:]
            #     p_sed2 = tmat[1,:]
            #     p_sed3 = tmat[2,:]
            #     p_sed4 = tmat[3,:]
                
            # if self.flowsim == True:
            #     tmat = np.concatenate((flow1,flow2,flow3,flow4)).reshape(4,self.communities)
            #     tmatrix = tmat.T
            #     t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
            #     for x in range(self.communities):#-3):
            #         for s in range(tmatrix.shape[1]):
            #             t2matrix[x,s] = self.proposal_jump(tmatrix[x,s], self.flowlim[0], self.flowlim[1], self.step_flow)
            #             # t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,self.step_flow)
            #             # if t2matrix[x,s] >= self.flowlimits[x,1]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #             # elif t2matrix[x,s] <= self.flowlimits[x,0]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #     # reorder each row , then transpose back as flow1, etc.
            #     tmp = np.zeros((self.communities,4))
            #     for x in range(t2matrix.shape[0]):
            #         a = np.sort(t2matrix[x,:])
            #         tmp[x,:] = a
            #     tmat = tmp.T
            #     p_flow1 = tmat[0,:]
            #     p_flow2 = tmat[1,:]
            #     p_flow3 = tmat[2,:]
            #     p_flow4 = tmat[3,:]

            # p_ax = self.proposal_jump(cm_ax, self.max_a, 0, self.step_a)
            # p_ay = self.proposal_jump(cm_ay, self.max_a, 0, self.step_a)
            # p_m = self.proposal_jump(m, 0, self.max_m, self.step_m)
            
            # v_proposal = []
            # if (self.sedsim == True) and (self.flowsim == False):
            #     v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4))
            # elif (self.flowsim == True) and (self.sedsim == False):
            #     v_proposal = np.concatenate((p_flow1,p_flow2,p_flow3,p_flow4))
            # elif (self.sedsim == True) and (self.flowsim == True):
            #     v_proposal = np.concatenate((p_sed1,p_sed2,p_sed3,p_sed4,p_flow1,p_flow2,p_flow3,p_flow4))
            # v_proposal = np.append(v_proposal,(p_ax,p_ay,p_m))
            v_proposal = stats.multivariate_normal(v_proposal, self.covar_matrix).rvs()
            print 'v_prop:', v_proposal
            threshold_stack = self.communities*4
            sed_matrix = (v_proposal[0:threshold_stack]).reshape(4,self.communities)
            print 'sed_matrix:', sed_matrix
            sed_matrix = sed_matrix.T
            tmp = np.zeros((self.communities,4))
            for x in range(sed_matrix.shape[0]):
                a = np.sort(sed_matrix[x,:])
                tmp[x,:] = a
            sed_matrix = tmp.T
            print 'ordered sed_matrix:', sed_matrix 
            sed_matrix = sed_matrix.reshape(1,12)
            flow_matrix = (v_proposal[threshold_stack:(2*threshold_stack)]).reshape(4,self.communities)
            flow_matrix = flow_matrix.T
            tmp = np.zeros((self.communities,4))
            for x in range(flow_matrix.shape[0]):
                a = np.sort(flow_matrix[x,:])
                tmp[x,:] = a
            flow_matrix = tmp.T
            flow_matrix = flow_matrix.reshape(1,12)

            v_proposal[0:threshold_stack] = sed_matrix
            print 'v proposal again',v_proposal
            v_proposal[threshold_stack:(2*threshold_stack)] = flow_matrix

                        # tmat = np.concatenate((sed1,sed2,sed3,sed4)).reshape(4,self.communities)
            #     tmatrix = tmat.T
            #     t2matrix = np.zeros((tmatrix.shape[0], tmatrix.shape[1]))
            #     for x in range(self.communities):
            #         for s in range(tmatrix.shape[1]):
            #             t2matrix[x,s] = self.proposal_jump(tmatrix[x,s], self.sedlim[0], self.sedlim[1], self.step_sed)
            #             # t2matrix[x,s] = tmatrix[x,s] + np.random.normal(0,self.step_sed)
            #             # if t2matrix[x,s] >= self.sedlimits[x,1]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #             # elif t2matrix[x,s] <= self.sedlimits[x,0]:
            #             #     t2matrix[x,s] = tmatrix[x,s]
            #     # reorder each row , then transpose back as sed1, etc.
            #     tmp = np.zeros((self.communities,4))
            #     for x in range(t2matrix.shape[0]):
            #         a = np.sort(t2matrix[x,:])
            #         tmp[x,:] = a
            #     tmat = tmp.T
            #     p_sed1 = tmat[0,:]
            #     p_sed2 = tmat[1,:]
            #     p_sed3 = tmat[2,:]
            #     p_sed4 = tmat[3,:]

            [likelihood_proposal, pred_data, diff] = self.likelihood_func(reef, self.core_data, v_proposal)
            diff_likelihood = likelihood_proposal - likelihood # to divide probability, must subtract
            print 'likelihood_proposal:', likelihood_proposal, 'diff_likelihood',diff_likelihood
            mh_prob = min(1, math.exp(diff_likelihood))
            u = random.uniform(0, 1)
            print 'u', u, 'and mh_probability', mh_prob
            
            if u < mh_prob: # accept
                #   Update position
                print i, ' is accepted sample'
                naccept += 1
                count_list.append(i)
                likelihood = likelihood_proposal
                # m = p_m
                # cm_ax = p_ax
                # cm_ay = p_ay
                # if self.sedsim == True:
                #     sed1 = p_sed1
                #     sed2 = p_sed2
                #     sed3 = p_sed3
                #     sed4 = p_sed4
                # if self.flowsim == True:
                #     flow1 = p_flow1
                #     flow2 = p_flow2
                #     flow3 = p_flow3
                #     flow4 = p_flow4
                # # self.save_core(reef,naccept)

                print  'likelihood:',likelihood, ' and difference score:', diff, 'accepted'

                # if self.sedsim == True:
                #     pos_sed1[i + 1,] = sed1
                #     pos_sed2[i + 1,] = sed2
                #     pos_sed3[i + 1,] = sed3
                #     pos_sed4[i + 1,] = sed4
                # if self.flowsim == True:
                #     pos_flow1[i + 1,] = flow1
                #     pos_flow2[i + 1,] = flow2
                #     pos_flow3[i + 1,] = flow3
                #     pos_flow4[i + 1,] = flow4
                # pos_ax[i + 1] = cm_ax
                # pos_ay[i + 1] = cm_ay
                # pos_m[i + 1] = m
                pos_v[i + 1,] = v_proposal
                pos_samples[i + 1,] = self.convert_core_format(pred_data, self.communities)
                pos_diff[i + 1,] = diff
                
                ax_append.plot(pos_samples[i + 1,],x_data, label=None)
                self.save_params(i+1, pos_diff[i + 1,], pos_samples[i + 1,], pos_v[i + 1,])
           
            else: #reject
                pos_v[i + 1,] = pos_v[i,]
                pos_samples[i + 1,] = pos_samples[i,]
                pos_diff[i + 1,] = pos_diff[i,]
                print 'REJECTED\nLikelihood:',likelihood,'and difference score:', pos_diff[i,]
                #   Copy past accepted state
                # if self.sedsim == True:
                #     pos_sed1[i + 1,] = pos_sed1[i,]
                #     pos_sed2[i + 1,] = pos_sed2[i,]
                #     pos_sed3[i + 1,] = pos_sed3[i,]
                #     pos_sed4[i + 1,] = pos_sed4[i,]
                # if self.flowsim == True:
                #     pos_flow1[i + 1,] = pos_flow1[i,]
                #     pos_flow2[i + 1,] = pos_flow2[i,]
                #     pos_flow3[i + 1,] = pos_flow3[i,]
                #     pos_flow4[i + 1,] = pos_flow4[i,]
                # pos_ax[i+1] = pos_ax[i]
                # pos_ay[i+1] = pos_ay[i]
                # pos_m[i+1] = pos_m[i]
                print i, 'rejected and retained'
            end = time.time()
            total_time = end-start
            print 'Time elapsed:', total_time

            if i==samples - 2:
                self.save_core(reef, i+1)

        accepted_count =  len(count_list)   
        print accepted_count, ' number accepted'
        print len(count_list) / (samples * 0.01), '% was accepted'
        accept_ratio = accepted_count / (samples * 1.0) * 100

        lgd = ax_append.legend(frameon=False, prop={'size':self.font+1},bbox_to_anchor = (1.,0.1))
        final_fig.savefig('%s/proposals.png'% (self.filename), extra_artists = (lgd,),bbox_inches='tight',dpi=300,transparent=False)
        plt.clf()

        ##### PLOT DIFFERENCE SCORE EVOLUTION ########
        fig = plt.figure(figsize=(6,4))
        ax= fig.add_subplot(111)
        ax.set_facecolor('#f2f2f3')
        x_range = np.arange(0,samples,1)
        plt.plot(x_range,pos_diff,'-',label='Difference score')
        plt.title("Difference score evolution", size=self.font+2)
        plt.ylabel("Difference", size=self.font+1)
        plt.xlabel("Number of samples", size=self.font+1)
        plt.xlim(0,len(pos_diff)-1)
        lgd = plt.legend(frameon=False, prop={'size':self.font+1},bbox_to_anchor = (1.,0.1))
        plt.savefig('%s/diff_evolution.png' % (self.filename), bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300,transparent=False)
        plt.clf()

        return (pos_v, pos_samples, x_data, pos_diff, accept_ratio, accepted_count, data_vec)#, pos_sed1,pos_sed2,pos_sed3,pos_sed4,pos_flow1,pos_flow2,pos_flow3,pos_flow4, pos_ax,pos_ay,pos_m, x_data, pos_diff, accept_ratio, accepted_count, data_vec)

#####################################################################

def main():
    
    #    Set all input parameters    #
    random.seed(time.time())
    samples= input('How many samples?')
    description = ''
    nCommunities = 3
    simtime = 8500
    timestep = np.arange(0,simtime+1,50)
    xmlinput = 'input_synth.xml'
    datafile = 'data/synth_core_vec.txt'
    core_depths, data_vec = np.genfromtxt(datafile, usecols=(0, 1), unpack = True) 
    core_data = np.loadtxt('data/synth_core_prop.txt', usecols=(1,2,3,4))
    covar_matrix = np.genfromtxt(open('data/covar_matrix.csv'), delimiter=",")
    covar_matrix = covar_matrix[:,0:27]
    vis = [False, False] # first for initialisation, second for cores
    sedsim, flowsim = True, True
    sedlim = [0., 0.005]
    flowlim = [0.,0.3]
    max_a = -0.2
    max_m = 0.3
    run_nb = 0
    while os.path.exists('results_multinomial_%s' % (run_nb)):
        run_nb+=1
    if not os.path.exists('results_multinomial_%s' % (run_nb)):
        os.makedirs('results_multinomial_%s' % (run_nb))
    filename = ('results_multinomial_%s' % (run_nb))

    #    Save File of Run Description   #
    if not os.path.isfile(('%s/description.txt' % (filename))):
        with file(('%s/description.txt' % (filename)),'w') as outfile:
            outfile.write('Test Description\n')
            outfile.write(description)
            outfile.write('\nSpecifications')
            outfile.write('\n\tmcmc.py')
            outfile.write('\n\tSimulation time: {0} yrs'.format(simtime))
            outfile.write('\n\tSediment simulated: {0}'.format(sedsim))
            outfile.write('\n\tFlow simulated: {0}'.format(flowsim))
            outfile.write('\n\tNo. samples: {0}'.format(samples))
            outfile.write('\n\tXML input: {0}'.format(xmlinput))
            outfile.write('\n\tData file: {0}'.format(datafile))
    ##### max/min values for each assemblage #####
    sedlim_1 = [[0., 0.005]]
    sedlim_2 = [[0.001,0.005]]
    sedlim_3 = [[0.001,0.005]]

    flowlim_1 = [[0.01,0.3]]
    flowlim_2 = [[0.,0.2]]
    flowlim_3 = [[0.,0.1]]
    
    sedlimits = []
    flowlimits = []

    if sedsim == True:
        sedlimits = np.concatenate((sedlim_1,sedlim_2,sedlim_3))#sedlim_4,sedlim_5,sedlim_6))
    if flowsim == True:
        flowlimits = np.concatenate((flowlim_1,flowlim_2,flowlim_3))#flowlim_4,flowlim_5,flowlim_6))


    print 'core data ', core_data.shape
    print 'core_depths', core_depths.shape
    print 'data_vec', data_vec.shape

    mcmc = MCMC(simtime, samples, nCommunities, core_data, core_depths, data_vec, timestep,  filename, xmlinput, 
                sedsim, sedlim, flowsim,flowlim, max_a, max_m, vis, covar_matrix)
    [pos_v, pos_samples, x_data, pos_diff, accept_ratio, accepted_count, data_vec] = mcmc.sampler()
    print 'successfully sampled'
    
    burnin = 0.1 * samples  # use post burn in samples
    pos_v = pos_v[int(burnin):, ]
    pos_sed1 = pos_v[:,0:3]
    pos_sed2 = pos_v[:,3:6]
    pos_sed3 = pos_v[:,6:9]
    pos_sed4 = pos_v[:,9:12]
    pos_flow1 = pos_v[:,12:15]
    pos_flow2 = pos_v[:,15:18]
    pos_flow3 = pos_v[:,18:21]
    pos_flow4 = pos_v[:,21:24]
    pos_ax = pos_v[:,24]
    pos_ay = pos_v[:,25]
    pos_m = pos_v[:,26]
    # pos_sed1 = pos_sed1[int(burnin):, ]
    # pos_sed2 = pos_sed2[int(burnin):, ]
    # pos_sed3 = pos_sed3[int(burnin):, ]
    # pos_sed4 = pos_sed4[int(burnin):, ]
    # pos_flow1 = pos_flow1[int(burnin):, ]
    # pos_flow2 = pos_flow2[int(burnin):, ]
    # pos_flow3 = pos_flow3[int(burnin):, ]
    # pos_flow4 = pos_flow4[int(burnin):, ]
    # pos_ax = pos_ax[int(burnin):]
    # pos_ay = pos_ay[int(burnin):]
    # pos_m = pos_m[int(burnin):]
    diff_mu = np.mean(pos_diff[int(burnin):])
    diff_std = np.std(pos_diff[int(burnin):])
    diff_mode, count = stats.mode(pos_diff[int(burnin):])
    
    print 'mean diff:',diff_mu, 'standard deviation:', diff_std

    with file(('%s/out_results.txt' % (filename)),'w') as outres:
        outres.write('Mean diff: {0}\nStandard deviation: {1}\nMode: {2}\n'.format(diff_mu, diff_std,diff_mode))
        outres.write('Accept ratio: {0} %\nSamples accepted : {1} out of {2}'.format(accept_ratio, accepted_count, samples))

    if not os.path.isfile(('%s/out_GLVE.csv' % (filename))):
        np.savetxt("%s/out_GLVE.csv" % (filename), np.c_[pos_m,pos_ax,pos_ay], delimiter=',')

    if not os.path.isfile(('%s/out_sed.csv' % (filename))):
        np.savetxt("%s/out_sed.csv" % (filename), np.c_[pos_sed1,pos_sed2,pos_sed3, pos_sed4], delimiter=',')

    if not os.path.isfile(('%s/out_flow.csv' % (filename))):
        np.savetxt("%s/out_flow.csv" % (filename), np.c_[pos_flow1,pos_flow2,pos_flow3, pos_flow4], delimiter=',')

    if not os.path.isfile(('%s/pos_proposal.csv' % (filename))):
        np.savetxt("%s/pos_proposal.csv" % (filename), pos_v, delimiter=',')

    fx_mu = pos_samples.mean(axis=0)
    fx_high = np.percentile(pos_samples, 95, axis=0)
    fx_low = np.percentile(pos_samples, 5, axis=0)

    fig = plt.figure(figsize=(3,6))
    plt.plot(data_vec, x_data,label='Synthetic core', color='k')
    plt.plot(fx_mu,x_data, label='Pred. (mean)',linewidth=1,linestyle='--')
    plt.plot(fx_low, x_data, label='Pred. (5th %ile)',linewidth=1,linestyle='--')
    plt.plot(fx_high,x_data, label='Pred. (95th %ile)',linewidth=1,linestyle='--')
    plt.fill_betweenx(x_data, fx_low, fx_high, facecolor='mediumaquamarine', alpha=0.4, label=None)
    plt.title("Core Data vs MCMC Uncertainty", size=mcmc.font+2)
    plt.ylim([0.,np.amax(core_depths)])
    plt.ylim(plt.ylim()[::-1])
    plt.ylabel('Depth [m]', size=mcmc.font+1)
    x_tick_labels = ['No growth','Shallow', 'Mod-deep', 'Deep', 'Sediment']
    x_tick_values = [0,1,2,3,4]
    plt.xticks(x_tick_values, x_tick_labels,rotation=70, fontsize=mcmc.font+1)
    plt.legend(frameon=False, prop={'size':mcmc.font+1}, bbox_to_anchor = (1.,0.2))
    plt.savefig('%s/mcmcres.png' % (filename), bbox_inches='tight', dpi=300,transparent=False)
    plt.clf()

    #      MAKE BOX PLOT     #
    if nCommunities == 3:
        if ((sedsim == True) and (flowsim == False)) or ((sedsim == False) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,12:15]

            com_1=[0,3,6,9]
            com_2=[1,4,7,10]
            com_3=[2,5,8,11]
            new_v = np.zeros((pos_v.shape[0],12))
            
            for i in range(4):
                new_v[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                new_v[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                new_v[:,i] = pos_v[:,com_3[i-8]]

            mpl_fig = plt.figure(figsize=(8,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Posterior values', fontsize= mcmc.font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(121)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=mcmc.font+2)
            ax2 = mpl_fig.add_subplot(122)
            ax2.boxplot(new_v)
            ax2.set_xlabel('Assemblage exposure thresholds', size=mcmc.font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            plt.clf()
        elif ((sedsim == True) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,24:27]
            com_1=[0,3,6,9,12,15,18,21]
            com_2=[1,4,7,10,13,16,19,22]
            com_3=[2,5,8,11,14,17,20,23]

            v_sed = np.zeros((pos_v.shape[0],12))
            v_flow = np.zeros((pos_v.shape[0],12))
            for i in range(4):
                v_sed[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                v_sed[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                v_sed[:,i] = pos_v[:,com_3[i-8]]

            for i in range(4):
                v_flow[:,i] = pos_v[:,com_1[i+4]]
            for i in range(4,8):
                v_flow[:,i] = pos_v[:,com_2[i]]
            for i in range(8,12):
                v_flow[:,i] = pos_v[:,com_3[i-4]]


            mpl_fig = plt.figure(figsize=(14,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Posterior values', fontsize= mcmc.font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(131)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=mcmc.font+2)
            ax2 = mpl_fig.add_subplot(132)
            ax2.boxplot(v_sed)
            ax2.set_xlabel('Assemblage sediment exposure thresholds', size=mcmc.font+2)
            ax3 = mpl_fig.add_subplot(133)
            ax3.boxplot(v_flow)
            ax3.set_xlabel('Assemblage flow exposure thresholds', size=mcmc.font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # mpl_fig = plt.figure(figsize=(10,4))
            # ax = mpl_fig.add_subplot(111)
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=mcmc.font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.clf()
    elif nCommunities == 6:
    	if ((sedsim == True) and (flowsim == False)) or ((sedsim == False) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,24:27]

            com_1=[0,6,12,18]
            com_2=[1,7,13,19]
            com_3=[2,8,14,20]
            com_4=[3,9,15,21]
            com_5=[4,10,16,22]
            com_6=[5,11,17,23]
            new_v = np.zeros((pos_v.shape[0],24))
            for i in range(4):
                new_v[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                new_v[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                new_v[:,i] = pos_v[:,com_3[i-8]]
            for i in range(12,16):
                new_v[:,i] = pos_v[:,com_4[i-12]]
            for i in range(16,20):
                new_v[:,i] = pos_v[:,com_5  [i-16]]
            for i in range(20,24):
                new_v[:,i] = pos_v[:,com_6[i-20]]

            mpl_fig = plt.figure(figsize=(8,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Posterior values', fontsize= mcmc.font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(121)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=mcmc.font+2)
            ax2 = mpl_fig.add_subplot(122)
            ax2.boxplot(new_v)
            ax2.set_xlabel('Assemblage exposure thresholds', size=mcmc.font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # for i in range(3,7):
            #     new_v[:,i] = pos_v[:,com_1[i-3]]
            # for i in range(7,11):
            #     new_v[:,i] = pos_v[:,com_2[i-7]]
            # for i in range(11,15):
            #     new_v[:,i] = pos_v[:,com_3[i-11]]

            # mpl_fig = plt.figure(figsize=(6,4))
            # ax = mpl_fig.add_subplot(111)
            # print 'pos_v.size',pos_v.size, 'pos_v.shape',pos_v.shape
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=mcmc.font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.clf()
        elif ((sedsim == True) and (flowsim == True)):
            v_glve = np.zeros((pos_v.shape[0],3))
            v_glve[:,0:3] = pos_v[:,48:51]

            com_1=[0,6,12,18,24,30,36,42]
            com_2=[1,7,13,19,25,31,37,43]
            com_3=[2,8,14,20,26,32,38,44]
            com_4=[3,9,15,21,27,33,39,45]
            com_5=[4,10,16,22,28,34,40,46]
            com_6=[5,11,17,23,29,35,41,47]
            v_sed = np.zeros((pos_v.shape[0],24))
            v_flow = np.zeros((pos_v.shape[0],24))

            for i in range(4):
                v_sed[:,i] = pos_v[:,com_1[i]]
            for i in range(4,8):
                v_sed[:,i] = pos_v[:,com_2[i-4]]
            for i in range(8,12):
                v_sed[:,i] = pos_v[:,com_3[i-8]]
            for i in range(12,16):
                v_sed[:,i] = pos_v[:,com_4[i-12]]
            for i in range(16,20):
                v_sed[:,i] = pos_v[:,com_5[i-16]]
            for i in range(20,24):
                v_sed[:,i] = pos_v[:,com_6[i-20]]

            for i in range(4):
                v_flow[:,i] = pos_v[:,com_1[i+4]]
            for i in range(4,8):
                v_flow[:,i] = pos_v[:,com_2[i]]
            for i in range(8,12):
                v_flow[:,i] = pos_v[:,com_3[i-4]]
            for i in range(12,16):
                v_flow[:,i] = pos_v[:,com_4[i-8]]
            for i in range(16,20):
                v_flow[:,i] = pos_v[:,com_5[i-12]]
            for i in range(20,24):
                v_flow[:,i] = pos_v[:,com_6[i-16]]


            mpl_fig = plt.figure(figsize=(14,4))
            ax = mpl_fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
            ax.set_title('Posterior values', fontsize= mcmc.font+2, y=1.02)
            ax1 = mpl_fig.add_subplot(131)
            ax1.boxplot(v_glve)
            ax1.set_xlabel('GLVE parameters', size=mcmc.font+2)
            ax2 = mpl_fig.add_subplot(132)
            ax2.boxplot(v_sed)
            ax2.set_xlabel('Assemblage sediment exposure thresholds', size=mcmc.font+2)
            ax3 = mpl_fig.add_subplot(133)
            ax3.boxplot(v_flow)
            ax3.set_xlabel('Assemblage flow exposure thresholds', size=mcmc.font+2)
            plt.savefig('%s/v_pos_boxplot.png'% (filename), dpi=300,transparent=False)
            # mpl_fig = plt.figure(figsize=(10,4))
            # ax = mpl_fig.add_subplot(111)
            # ax.boxplot(new_v)
            # ax.set_ylabel('Posterior values')
            # ax.set_xlabel('Input vector')
            # plt.title("Boxplot of posterior distribution \nfor GLVE and threshold parameters", size=mcmc.font+2)
            # plt.savefig('%s/v_pos_boxplot.pdf'% (filename), dpi=300)
            # # plt.savefig('%s/v_pos_boxplot.svg'% (filename), format='svg', dpi=300)
            plt.clf()
    mcmc.plot_results(pos_m, pos_ax, pos_ay, pos_sed1, pos_sed2, pos_sed3, pos_sed4, pos_flow1, pos_flow2, pos_flow3, pos_flow4,burnin)

    print 'Finished simulations'
if __name__ == "__main__": main()
