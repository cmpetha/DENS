# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 16:50:16 2018

@author: Charlie
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def plot(plots, xc, Vc, zc, ai, da, eats, nKGB, KGB, DATA, NN):
    
    
    #plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('axes', labelsize=16)
    plt.rc('axes', titlesize=18)
    colorsa = pl.cm.magma(np.linspace(0,1,2*(len(eats)+len(nKGB)+2)))
    colors = colorsa[::-1]
    lines = ['-', '--', '-.', ':', '--', '-.', ':', '--', '-.', ':', '--', '-.', ':']
    
    if any(string == 'w' for string in plots):
        fig1 = plt.figure(1)
        ax1 = fig1.add_subplot(111)
        plt.hlines(-1, 0, DATA[2,0], label='$\Lambda$CDM', linewidth=0.85, color='k', linestyle=lines[0])
        plt.xlim(left=0, right=zc+3)
        #plt.ylim(top=-0.8)
        plt.xlabel('$z$')
        plt.ylabel('$ w $')
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[11+NN*j], color=colors[2+2*j], zorder=j, linestyle=lines[1+j],\
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        if KGB == 'y' :
            for i in range(len(nKGB)):
                plt.plot(DATA[2], DATA[20+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                         linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.fill_between(DATA[2], -1.08, -0.9, where=DATA[2] <= 0.2, edgecolor='none', \
#                         label='\emph{DES} WL + \emph{Planck} CMB + SN \n + BAO $2\sigma$ constraint', \
#                         alpha=0.25, facecolor='b', zorder=len(eats)+2)
        plt.legend(frameon=False, fontsize=15)
        #plt.legend(loc=3, bbox_to_anchor=(0.4,0.06), frameon=False, fontsize=13)
        ax1.tick_params(axis='both', direction='in')
        plt.savefig('w.png', dpi=500, format='png')
        
    ##H plot
    if any(string == 'H' for string in plots):
        fig2 = plt.figure(2)
        ax2 = fig2.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[12+NN*j], color=colors[2*j+2], \
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        if KGB == 'y' :
            for i in range(len(nKGB)):
                plt.plot(DATA[2], DATA[21+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                         linewidth=0.85, color=colors[len(eats)+2+2*i])
        plt.plot(DATA[2], DATA[3], label='$\Lambda$CDM', linewidth=0.85, color='k')
        plt.xlim(left=0, right=zc+3)
        plt.ylim(bottom=0 ,top=DATA[3,int(((1/(zc+5))-ai)/da)])
        plt.xlabel('$z$')
        plt.ylabel('$H$ [$GeV$]')
        plt.legend(frameon=False, fontsize='small')
        ax2.tick_params(axis='both', direction='in')
        format_label_string_with_exponent(ax2, axis='both')
        plt.savefig('H.png', dpi=500, format='png')
 
        fig15 = plt.figure(15)
        ax15 = fig15.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[15+NN*j], color=colors[2*j+2], \
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        if KGB == 'y' :
            for i in range(len(nKGB)):
                plt.plot(DATA[2], DATA[23+NN*i], color=colors[len(eats)+2+2*i], \
                         label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), linewidth=0.85, linestyle=lines[len(eats)+1+i])
        plt.xlim(left=0, right=zc+3)
        plt.ylim(bottom=0)
        plt.xlabel('$z$')
        plt.ylabel('$ \mid \Delta H / H_{\Lambda} \mid$')
        plt.legend(frameon=False, fontsize=14)
        ax15.tick_params(axis='both', direction='in')
        plt.savefig('Hdiff.png', dpi=500, format='png')
    
    ##dL plot   
    ##dL Redshift
    dLred = 2
    if any(string == 'dL' for string in plots):
        fig4 = plt.figure(4)
        ax4 = fig4.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[50+NN*j], color=colors[2*j+2],linestyle=lines[1+j], \
                    label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        plt.plot(DATA[2], DATA[51], label='$\Lambda$CDM', linewidth=0.85, color='k')
        if KGB == 'y' :
            for i in range(len(nKGB)):
                plt.plot(DATA[2], DATA[49+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                         linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
        plt.xlim(left=0, right=dLred)
        plt.ylim(bottom=0 ,top=DATA[51,int(((1/(dLred+2))-ai)/da)])
        plt.xlabel('$z$')
        plt.ylabel('$d_L$\ [$GeV$]')
        plt.legend(frameon=False, fontsize='small')
        ax4.tick_params(axis='both', direction='in')
        format_label_string_with_exponent(ax4, axis='both')
        plt.savefig('dL.png', dpi=500, format='png')
        
        fig5 = plt.figure(5)
        ax5 = fig5.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[52+NN*j], color=colors[2*j+2], linestyle=lines[1+j],\
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        if KGB == 'y' :
            for i in range(len(nKGB)):
                plt.plot(DATA[2], DATA[54+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                         linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])        
        plt.hlines(0.0018, 0, DATA[2,0], label='$WFIRST$ Aggregate Precision $1\sigma$', linewidth=0.85, color='b')
        plt.hlines(0.0036, 0, DATA[2,0], label='$WFIRST$ Aggregate Precision $2\sigma$', linewidth=0.85, color='g')
#        plt.hlines(0.02, 0, 0.4, linewidth=0.85, color='c')
#        plt.hlines(0.005, 0.4, 0.6, linewidth=0.85, color='c')
#        plt.hlines(0.02, 0.6, DATA[2,0], linewidth=0.85, color='c', \
#                   label='Current SN constraint on \n $\Lambda$CDM deviation')
        plt.xlim(left=0, right=2)
        plt.ylim(bottom=0)
        plt.xlabel('$z$', fontsize=18)
        plt.ylabel('$\mid \Delta d_L / d_{L,\Lambda} \mid$', fontsize=18)
        plt.legend(frameon=False, fontsize=12)
        ax5.tick_params(axis='both', direction='in', labelsize=14)
        plt.tight_layout()
        plt.savefig('dLdiff.png', dpi=500, format='png')   
        
    ##growth function plot
    if any(string == 'g' for string in plots):
        fig6 = plt.figure(6)
        ax6 = fig6.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[29+NN*j], color=colors[2*j+2], linestyle=lines[1+j],\
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        for i in range(len(nKGB)):
            plt.plot(DATA[2], DATA[31+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]),\
                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
        plt.plot(DATA[2], DATA[39], color='k', label='$\Lambda$CDM', linewidth=0.85)
        plt.xlim(left=0, right=3)
        #plt.ylim(bottom=0.7)
        plt.xlabel('$z$')
        plt.ylabel('$g(z)$')
        plt.legend(frameon=False, fontsize=14)
        ax6.tick_params(axis='both', direction='in')
        plt.savefig('g.png', dpi=500, format='png')
        
        
#        yerrpoints = [0.046, 0.034, 0.028, 0.028, 0.024, 0.022, 0.02, 0.02, \
#                      0.018, 0.018, 0.018, 0.018, 0.02, 0.022]
#        fig21 = plt.figure(21)
#        ax21 = fig21.add_subplot(111)
#        for j in range(len(eats)):
#            plt.plot(DATA[2], DATA[37+NN*j], color=colors[2*j+2], linestyle=lines[1+j],\
#                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
#        for i in range(len(nKGB)):
#            plt.plot(DATA[2], DATA[38+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
#                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.plot(DATA[2], DATA[42], color='k', label='$\Lambda$CDM', linewidth=0.85)
#        plt.xlim(left=0, right=2.1)
#        plt.text(0.1, 0.95, '$f = \Omega_{m}^{\\gamma}$', fontsize=14)
#        plt.errorbar(zpoints, fpoints, yerr=yerrpoints,fmt='none',ecolor='k',\
#                     elinewidth=0.75, capthick=0.5, capsize=1.5,\
#                     label='Euclid Forecasted $2\sigma$ Error')
#        #plt.ylim(bottom=0.7)
#        plt.xlabel('$z$')
#        plt.ylabel('$f$')
#        plt.legend(frameon=False, fontsize=14)
#        ax21.tick_params(axis='both', direction='in')
#        plt.savefig('f.png', dpi=500, format='png')       
        
        fig7 = plt.figure(7)
        ax7 = fig7.add_subplot(111)
        plt.hlines(0.54545454545, 0, DATA[2,0], color='k', label='$\Lambda$CDM Linder', linewidth=0.85)
        plt.plot(DATA[2], DATA[41], color='c', label='$\Lambda$CDM Numerical', linewidth=0.85)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[28+NN*j], color=colors[2*j+2],\
                     label='\emph{{STSQ}} Numerical $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        for i in range(len(nKGB)):
            plt.plot(DATA[2], DATA[33+NN*i], label='\emph{{KGB}} Numerical $n={0}$'.format(nKGB[i]),\
                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
            nall = [1,2,3,4,5]
            if any(x == nKGB[i] for iii, x in enumerate(nall)):
                plt.plot(DATA[2], DATA[34+NN*i], label='\emph{{KGB}} Analytic $n={0}$'.format(nKGB[i]), \
                         linewidth=0.85, color=colors[len(eats)+2+2*i])
#        plt.hlines(0.530454545450, 0, 3, color='b', label='$WFIRST$ Aggregate Precision $1\sigma$', linewidth=0.85)
#        plt.hlines(0.56045454545, 0, 3, color='b', linewidth=0.85)
#        plt.hlines(0.57545454545, 0, 3, label='$WFIRST$ Aggregate Precision $2\sigma$', linewidth=0.85, color='g')
#        plt.hlines(0.51545454545, 0, 3, linewidth=0.85, color='g')
        plt.xlim(left=0, right=zc+5)
        #plt.ylim(bottom=0.545, top=0.5555)
        plt.xlabel('$z$')
        plt.ylabel('$\gamma$')
        plt.legend(frameon=False, fontsize=12)
        ax7.tick_params(axis='both', direction='in')
        plt.savefig('gamma.png', dpi=500, format='png')
        

        
        fig13 = plt.figure(13)
        ax13 = fig13.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[30+NN*j], color=colors[2*j+2], linestyle=lines[1+j], \
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        for i in range(len(nKGB)):
            plt.plot(DATA[2], DATA[32+NN*i],label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.plot(DATA[2], DATA[0], label='$WFIRST$ Aggregate Precision $1\sigma$', linewidth=0.85, color='b')
#        plt.plot(DATA[2], 2*DATA[0], label='$WFIRST$ Aggregate Precision $2\sigma$', linewidth=0.85, color='g')
        plt.hlines(0.0013, 0, DATA[2,0], label='$WFIRST$ Aggregate Precision $1\sigma$', linewidth=0.85, color='b')
        plt.hlines(0.0026, 0, DATA[2,0], label='$WFIRST$ Aggregate Precision $2\sigma$', linewidth=0.85, color='g')
        plt.xlim(left=0, right=3)
        plt.xlabel('$z$')
        plt.ylabel('$\mid \Delta \delta / \delta_{\Lambda} \mid$')
        plt.legend(frameon=False, fontsize=10)
        ax13.tick_params(axis='both', direction='in')
        plt.savefig('gdiff.png', dpi=500, format='png')

#        fig19 = plt.figure(19)
#        ax19 = fig19.add_subplot(111)
#        plt.plot([], [], ' ', label='Linder and Numerical Fractional Difference')
#        plt.plot(DATA[2], DATA[40], color='k', label= '$\Lambda$CDM', linewidth=0.85)
#        for j in range(len(eats)):
#            plt.plot(DATA[2], DATA[36+NN*j], color=colors[2*j+2], linestyle=lines[1+j], \
#                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
#        for i in range(len(nKGB)):
#            plt.plot(DATA[2], DATA[35+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
#                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.xlim(left=0, right=DATA[2,0])
#        plt.xlabel('$z$')
#        plt.ylabel('$\Delta g$')
#        plt.legend(frameon=False, fontsize=14)
#        ax19.tick_params(axis='both', direction='in')
#        formatter = mticker.ScalarFormatter(useOffset=False)
#        ax19.yaxis.set_major_formatter(formatter)
#        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#        plt.savefig('gdiff2.png', dpi=500, format='png')

        
#        fig14 = plt.figure(14)
#        ax14 = fig14.add_subplot(111)
#        for j in range(len(eats)):
#            plt.plot(DATA[2], DATA[11+(j+1)*DEL], color=colors[2*j+2], linestyle=lines[1+j], \
#                    label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
#        for i in range(len(nKGB)):
#            plt.plot(DATA[2], DATA[30+(i+1)*DEL], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]),\
#                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.plot(DATA[2], DATA[9], label='$\Lambda$CDM', linewidth=0.85, color='k')
#        plt.xlim(left=0, right=3)
#        plt.xlabel('$z$')
#        plt.ylabel('$\sigma_8$')
#        plt.legend(frameon=False, fontsize='small')
#        ax14.tick_params(axis='both', direction='in')
#        plt.savefig('sig8.png', dpi=500, format='png')
#        
#        fig20 = plt.figure(20)
#        ax20 = fig20.add_subplot(111)
#        for j in range(len(eats)):
#            plt.plot(DATA[2], DATA[36+(j+1)*DEL], color=colors[2*j+2], linestyle=lines[1+j], \
#                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
#        for i in range(len(nKGB)):
#            plt.plot(DATA[2], DATA[37+(i+1)*DEL], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
#                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.xlim(left=0, right=3)
#        plt.xlabel('$z$')
#        plt.ylabel('| $\Delta \sigma_8 / \sigma_8$ |')
#        plt.legend(frameon=False, fontsize='small')
#        ax20.tick_params(axis='both', direction='in')
#        formatter = mticker.ScalarFormatter(useOffset=False)
#        ax20.yaxis.set_major_formatter(formatter)
#        plt.savefig('sig8diff.png', dpi=500, format='png')        
#        
#    if any(string == 'dens' for string in plots):
#        fig8 = plt.figure(8)
#        ax8 = fig8.add_subplot(111)
#        for j in range(len(eats)):
#            plt.plot(DATA[2], DATA[6+(j+1)*DEL], color=colors[2*j+2], linestyle=lines[1+j], \
#                     label='$\Omega_{\text{m}}$ \emph{STSQ}  $z_c$=%d $\eta$=%d'%(zc, eats[j]), linewidth=0.85)
##            plt.plot(DATA[2], DATA[18+(j+1)*DEL], color=colors[2*j+2], \
##                     label=' $\Omega_DE$ STSQ $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
#        if KGB == 'y':
#            for i in range(len(nKGB)):
#                plt.plot(DATA[2], DATA[22+(i+1)*DEL], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
#                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.plot(DATA[2], DATA[4], label='$\Omega_{\text{m}}$ $\Lambda$CDM', linewidth=0.85, color='k')
##        plt.plot(DATA[2], DATA[6], label='$\Omega_{DE}$ $\Lambda$CDM', linewidth=0.85, color='k')
#        plt.xlim(left=0, right=0.01)
#        plt.ylim(bottom=0.298, top=0.301)
#        plt.xlabel('$z$')
#        plt.ylabel('$\Delta$ $\Omega_\text{m}$')
#        plt.legend(frameon=False, fontsize='small')
#        ax8.tick_params(axis='both', direction='in')
#        plt.savefig('densfrac.png', dpi=500, format='png')
        
        
        
    if any(string == 'phi' for string in plots):
        fig11 = plt.figure(11)
        ax11 = fig11.add_subplot(111)
        plt.hlines(xc, 0, 1, label='\emph{STSQ} $\phi_c$', linewidth=0.85, color='k')
        for j in range(len(eats)):
            plt.plot(DATA[1], DATA[8+NN*j], color=colors[2*j+2], linestyle=lines[1+j], \
                    label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        for i in range(len(nKGB)):
            plt.plot(DATA[1], DATA[25+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
        plt.xlim(left=DATA[1,0], right=1)
        #plt.ylim(bottom=2.26*10**19, top=2.3*10**19)
        plt.xlabel('$a$')
        plt.ylabel('$\phi$ [$GeV$]')
        plt.legend(frameon=False, fontsize=14)
        ax11.tick_params(axis='both', direction='in')
        format_label_string_with_exponent(ax11, axis='both')
        plt.savefig('phi.png', dpi=500, format='png')

        
        
    if any(string == 'V' for string in plots):
        fig12 = plt.figure(12)
        ax12 = fig12.add_subplot(111)
        plt.hlines(Vc, 0, 1, label='\emph{STSQ} $V_c$', linewidth=0.85, color='k')
        for j in range(len(eats)):
            plt.plot(DATA[1], DATA[10+NN*j], color=colors[2*j+2], linestyle=lines[1+j], \
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        plt.xlim(left=DATA[1,0], right=1)
        #plt.ylim(bottom=2*10**-47, top=3*10**-47)
        plt.xlabel('$a$')
        plt.ylabel('$V$ [$GeV^{4}$]')
        plt.legend(frameon=False, fontsize=14)
        ax12.tick_params(axis='both', direction='in')
        format_label_string_with_exponent(ax12, axis='both')
        plt.savefig('V.png', dpi=500, format='png')

        
#        fig17 = plt.figure(17)
#        ax17 = fig17.add_subplot(111)
#        for j in range(len(eats)):
#            plt.plot(DATA[1], DATA[28+(j+1)*DEL], color=colors[2*j+2], linestyle=lines[1+j], \
#                     label='STSQ $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
##        for i in range(len(nKGB)):
##            plt.plot(DATA[1], DATA[27+(i+1)*DEL], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
##                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
#        plt.xlim(left=DATA[1,0], right=1)
#        #plt.ylim(bottom=0., top=0.3*10**-47.)
#        plt.xlabel('$a$')
#        plt.ylabel('$V/\\rho$')
#        plt.legend(frameon=False, fontsize='small')
#        ax17.tick_params(axis='both', direction='in')
#        plt.savefig('Vrho.png', dpi=500, format='png')
#        
    if any(string == 'q' for string in plots):
        fig16 = plt.figure(16)
        ax16 = fig16.add_subplot(111)
        plt.plot(DATA[2], DATA[5], label='$\Lambda$CDM', linewidth=0.85, color='k')
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[16+NN*j], color=colors[2*j+2], \
                     label='\emph{{STSQ}} $z_c={0}$ $\eta$={1}'.format(zc, eats[j]), linewidth=0.85)
        for i in range(len(nKGB)):
            plt.plot(DATA[2], DATA[24+NN*i], label='\emph{{KGB}}  $n={0}$'.format(nKGB[i]), \
                     linewidth=0.85, color=colors[len(eats)+2+2*i], linestyle=lines[len(eats)+1+i])
        plt.xlim(left=0, right=5)
        plt.yticks(np.arange(-0.7, 0.7, step=0.2))
        plt.xlabel('$z$')
        plt.ylabel('$q$')
        plt.legend(frameon=False, fontsize=14)
        ax16.tick_params(axis='both', direction='in')
        plt.savefig('q.png', dpi=500, format='png')
        
        
    if any(string == 'sc' for string in plots):
        fig18 = plt.figure(18)
        ax18 = fig18.add_subplot(111)
        for j in range(len(eats)):
            plt.plot(DATA[2], DATA[27+NN*j], color=colors[2*j+2], linestyle=lines[1+j], \
                     label='\emph{{STSQ}}  $z_c={0}$ $\eta={1}$'.format(zc, eats[j]), linewidth=0.85)
        plt.xlim(left=0, right=zc+3)
        plt.xlabel('$z$')
        plt.ylabel('$\\xi$')
        plt.legend(frameon=False, fontsize=14)
        ax18.tick_params(axis='both', direction='in')
        plt.savefig('scaling.png', dpi=500, format='png')

        
    if any(string == 'G' for string in plots):  
      
        fig23 = plt.figure(23)
        ax23 = fig23.add_subplot(111)
        for j in range(len(nKGB)):
            plt.plot(DATA[1], DATA[19+NN*j],linewidth=0.85, color=colors[2*j+2],\
                     label='$n={0}$'.format(nKGB[j]), linestyle=lines[1+j])
        plt.plot(DATA[1], DATA[7] ,linewidth=0.85, color='k', label='$n \\rightarrow$$\infty$ limit')
        plt.text(0.5, 1.8, '$G_{eff} = \\beta G$', fontsize=18)
        plt.xlabel('$a$')
        plt.ylabel('$\\beta$')
        plt.legend(frameon=False, fontsize=14)
        ax23.tick_params(axis='both', direction='in')
        plt.savefig('Geff.png', dpi=500, format='png')
        
        
        
def update_label(old_label, exponent_text):
    if exponent_text == "":
        return old_label
    
    try:
        units = old_label[old_label.index("[") + 1:old_label.rindex("]")]
    except ValueError:
        units = ""
    label = old_label.replace("[{0}]".format(units), "")
    
    exponent_text = exponent_text.replace("\\times", "")
    
    return "{0} [{1} {2}]".format(label, exponent_text, units)
    
def format_label_string_with_exponent(ax, axis='both'):  
    """ Format the label string with the exponent from the ScalarFormatter """
    ax.ticklabel_format(axis=axis, style='sci')

    axes_instances = []
    if axis in ['x', 'both']:
        axes_instances.append(ax.xaxis)
    if axis in ['y', 'both']:
        axes_instances.append(ax.yaxis)
    
    for ax in axes_instances:
        ax.major.formatter._useTex = True
        plt.draw() # Update the text
        exponent_text = ax.get_offset_text().get_text()
        label = ax.get_label().get_text()
        ax.offsetText.set_visible(False)
        ax.set_label_text(update_label(label, exponent_text))