#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:44:54 2019

@author: charlie
"""

# -*- coding: utf-8 -*-
"""
DEsimulator

Main script for simualting Quintessence dark energy, specfically a novel field
named Sharp Transition Scaling Quintessence, and comparing this with LCDM and
a modified gravity model - Kinetic Gravity Braiding.

@author: Charlie Mpetha
"""

import time
from os import sys
import numpy as np
from plots2 import plot
from scipy.optimize import fsolve
from multiprocessing.pool import Pool
from multiprocessing import Array
from functools import partial




###############################################################################
###                                                                         ###
###                            Defining Parameters                          ### 
###                                                                         ###
###############################################################################

##Curret values for cosmological parameters, as well as model dependent constants
##are set here. Also step size and some useful numbers


H0 = 1.437722539*10.**-42.   ##Current value in GeV from Planck 2018
#H0 = 1.535608363*10**-42.   ##Current value in GeV from Bonvin 2017
Mpl = 2.4*10.**18.   ##Reduced Planck mass in GeV
rhoc0 = 3.*(H0*Mpl)**2.   ##Current critical density
OmegaM0, OmegaR0, OmegaDE0 = 0.3, 8.25*10**-5., 0.7   ##Current density fractions
rhode0 =  OmegaDE0*rhoc0   ##Current dark energy energy density
rhom0 = OmegaM0*rhoc0   ##Current matter energy density                
rhor0 = OmegaR0*rhoc0   ##Current raditation energy density
sig80 = 0.806   ##simga_8,0 DES+Planck 2018          
ai = 0.02  ##Initial scale factor
da = 0.00001   ##Scale factor step size
n = int((1-ai)/da+1)   ##Number of steps in the integration
NN = 55
Vc = rhode0   ##Critical potential


###############################################################################
###                                                                         ###
###                               Data Array                                ### 
###                                                                         ###
###############################################################################

##Initialise a Shared data array to hold all calculated parameters
def initarr():
    
    A, B, zc, order, plots = usr_input()  ##Find user input
    xnum = NN*len(A)   ##Number of rows in array
    sharedDATA = Array('d', range(xnum*n), lock=False)   
    ##Multiprocessing Array, can be written to by each worker
    
    return sharedDATA, A, B, zc, order, plots  ###Return all parameters


###############################################################################
###                                                                         ###
###                               User input                                ### 
###                                                                         ###
###############################################################################

###Used to decide which parts of the code are run to produce the desired graphs
###For brevity, if yes chosen for 'test' option simple graphs with nominal
###parameter choices is run, takes approx 5 seconds. 
##Used to test functionality of code

def usr_input():
    
    test = str(input('test? (y):'))
    
    if test == 'y':
        
        zc = 10   ##Crossover redshift
        A = [0.1, 1]  ##Different values of STSQ free parameter eta
#        nKGB = [0] ##Different values of KGB free parameter n
        plots = ['w']   ##Which plots to produce
        #plots = ['g']
#        KGB = 'n'   ##Wether KGB is modelled (increases runtime)
        
    else:
        
        while True:
            order = str(input('Order of Taylor series approximation (1 or 2) ?: \n'))
            if order.replace(".", "", 1).isdigit():
                order = float(order)
                if order == 1. or order == 2.:
                    break 
                else:
                    print('Order must be 1 or 2')
#                continue
            if zc == 'exit':
                 sys.exit(0)
#            else:
#                print('Order must be 1 or 2')
#                continue
        
        ##YOU MAY WISH TO CHANGE THIS SECTION TO MAKE IT MORE RELEVANT FOR
        ##YOUR OWN QUINTESSENCE POTENTIAL. MULTIPLE VALUES OF A FREE
        ##PARAMETER CAN BE STUDIED AT ONCE, WHILE A SECOND IS REMAINED FIXED.
        while True:
             zc = input('Crossover Redshift (max = 49), zc = ')
             if zc.replace(".", "", 1).isdigit():
                 zc = float(zc)
                 break
             if zc == 'exit':
                 sys.exit(0)
                 
        if order == 1. or order == 2.:         
            while True:  
                try:
                    input_A = input('Choose values of A. Seperate each choice with a space: \n')       
                    if input_A == 'exit':
                        sys.exit(0)
                    if all(x.replace(".", "", 1).replace("-", "", 1).isdigit() for x \
                           in input_A.split()) :
                        A = [float(x) for x in input_A.split()]
                        if order == 1.:
                            B = np.zeros(len(A))
                        break
                except ValueError:
                    continue
                
        if order == 2. :         
            while True:  
                try:
                    input_B = input('Choose values of B. Seperate each choice with a space: \n')       
                    if input_B == 'exit':
                        sys.exit(0)
                    if all(x.replace(".", "", 1).replace("-", "", 1).isdigit() for x \
                           in input_B.split()) :
                        B = [float(x) for x in input_B.split()]
                        if len(B) != len(A):
                            print('Number of B values must be the same as number of A values.')
                            continue
                        break
                except ValueError:
                    continue
                
        
        input_string = input('Which graphs? Seperate each choice with a space. \n  Equation of State:"w" Hubble:"H"  Luminosity Distance:"dL"   Growth function:"g"  Field Value:"phi"  Potential:"V"  Deceleration Parameter:"q"  Scaling:"sc"  Gravitational Constant:"G"\n')
        
        plots = input_string.split()
        if any(val == 'exit' for val in plots):
            sys.exit(0)

    return A, B, zc, order, plots



###############################################################################
###                                                                         ###
###                        Parallelised integration                         ### 
###                                                                         ###
###############################################################################

##The worker module contains all the computation required for each user defined 
##value of eta for STSQ or n for KGB. This setup allows each different model
##to be simulated simuLtaneously - making use of available computing power and saving time.


    
def worker(j, A, B, zc, order, plots):   #
    #Create loop over all code so that it can be parallelised  
    ##Each worker performs calculations for each user-defined eta/nKGB      

    ac = 1./(1.+zc)   ##Crossover scalefactor
    Vc = rhode0   ##Critical potential
    l = ((6.*Vc + 3.*rhom0*(1.+zc)**3.)/(2.*Vc))**0.5   ##model dependent constant
    xc = (Mpl/l)*np.log(Mpl**4. / rhode0)   ##Critical field value   
        
    ##STSQ INITIAL CONDITIONS - MUST BE CHANGED ON CHANGING QUINTESSENCE 
    ##POTENTIAL                               
    xi = (Mpl/l)*np.log(((Mpl**4.)/(rhode0))*(ai/ac)**3.)   ##Initial field value                            
    yi = (2.*np.exp(np.log(Mpl**4.) - l*xi/Mpl))**0.5   ##Initial derivative
    
    Hi = ((rhom0*(ai**-3.)*ai + rhor0*(ai**-3.)) / (3. * (Mpl**2.)))**0.5
    ui = ai
    vi = ai*Hi
    vil = vi   
            
    def f(x, y, u, v, ul, vl, a):   ##Calculates useful parameters                
        
        
        ##CHANGE THE FOLLOWING POTENTIAL TO CHANGE THE QUINTESSENCE MODEL
        ##BEING STUDIED. N.B.ALSO CHANGE INITIAL CONDITIONS FUTHER UP
        def V(x):   ##Function for the STSQ potential and its derivative
                    ##Potetial can be changed for any Scalar Field model
            if order == 1. :                                   
                if x < xc:
                    return (Mpl**4.)*np.exp(-l*x/Mpl), \
                            -l*(Mpl**3.)*np.exp(-l*x/Mpl)
                else:
                    return (Mpl**4.)*np.exp(-l*xc/Mpl)*(1.+ \
                           A[j]*(x-xc)/Mpl), \
                           A[j]*(Mpl**3.)*np.exp(-l*xc/Mpl)  
                                  
            if order == 2. :                           
                if x < xc:
                    return (Mpl**4.)*np.exp(-l*x/Mpl), \
                            -l*(Mpl**3.)*np.exp(-l*x/Mpl)
                else:
                    return (Mpl**4.)*np.exp(-l*xc/Mpl)*(1. + \
                           A[j]*(x-xc)/Mpl + B[j]*((x-xc)/Mpl)**2.), \
                           (A[j]*(Mpl**3.)+2.*B[j]*(Mpl**2.)*\
                            (x-xc))*np.exp(-l*xc/Mpl)    
            
   
        Vp, Vdash = V(x)##Potential and its derivative
        rhor = rhor0*(a**-4.)   ##Radiation density
        rhoml = rhom0*(a**-3.)  ####Matter density in LCDM
        rhom = (((l**2.)/3.) - 1.)*2.*V(xi)[0]*(ai/a)**3.   ##Matter density STSQ
        rhophi = 0.5*(y**2.) + Vp   ##STSQ field density
        prphi = 0.5*(y**2.) - Vp    ##STSQ field pressure
        H = ((rhophi + rhom + rhor) / (3. * (Mpl**2.)))**0.5   ##STSQ Hubble parameter
        Hl = ((rhode0 + rhoml + rhor) / (3. * (Mpl**2.)))**0.5 ##LCDM Hubble parameter
        OmegaMl = rhoml/(3.*(Mpl*Hl)**2.)   ##Matter density fraction in LCDM
        OmegaDEl = rhode0/(3.*(Mpl*Hl)**2.)   ##L density fraction in LCDM
        OmegaRl = rhor/(3.*(Mpl*Hl)**2.)   ##Rad density fraction in STSQ
        OmegaM = rhom/(3.*(Mpl*H)**2.)   ##Matter density fraction in STSQ
        OmegaDE = rhophi/(3.*(Mpl*H)**2.)   ##DE density fraction in STSQ
        OmegaR = rhor/(3.*(Mpl*H)**2.)    ##Rad density fraction in STSQ
        wphi = prphi/rhophi   ##Barotropic parameter in STSQ
        q = OmegaR + 0.5*(OmegaM + (1.+3.*wphi)*OmegaDE)  ##Deceleration parameter in STSQ
        ql = OmegaRl + 0.5*OmegaMl - OmegaDEl   ##Deceleration parameter in LCDM

        return -3.*H*y - Vdash, H, rhophi, wphi, Vp, Hl, OmegaM, OmegaMl, \
                OmegaDE, OmegaDEl, q, ql, -2.*H*v+0.5*(Mpl**-2.)*rhom*u, \
                -2.*Hl*vl+0.5*rhoml*(Mpl**-2.)*ul     
                ##Return all parameters for use        
  
    
    DATA = np.zeros((55,n))  ##Create temporary array to hold calculated data
    
    def intstep(x, y, a):    ##Performs 4th order RK integration to 
                                     ##solve scalar field equation and more

        dm = np.zeros(n)   ##Initialise arrays for dL calculation
        dml = np.zeros(n)
        glarr = np.zeros(n)   ##Initialise arrays for growth calculation
        garr = np.zeros(n)
        
        u, ul = ui, ui   ##Initial conitions for growth ODE
        v, vl = vi, vil
        
        for i in range(int(n)):   
            ##Integrate in n steps of da from ai to 1 (present)
            
            l0, H, rho, w, V, Hl, OM, OMl, ODE, ODEl, q, ql, \
            n0, m0 = f(x, y, u, v, ul, vl, a) 
            ##Find all parameters at current a
            z = (1./a)-1.   ##Redshift
            dlna = da/a   ##Logarithmic step in a
            dtl = da/(Hl*a)   ##Timestep for LCDM
            dt = da/(H*a)   ##Timestep for STSQ
            
            ##4th Order Runge-Kutta integration method
            l0, m0, n0 = l0*dt, m0*dtl, n0*dt
            k0, j0, ll0 = dt*y, dtl*vl, dt*v
            k1, j1, ll1 = dt*(y+0.5*l0), dtl*(vl+0.5*m0), dt*(v+0.5*n0)
            l1, _, _, _, _, _, _, _, _,_, _, n1, m1, _ = \
            f(x+k0/2., y+l0/2., u+ll0/2., v+n0/2., ul+j0/2., vl+m0/2., a+da/2.)
            l1, m1, n1 = l1*dt, m1*dtl, n1*dt
            k2, j2, ll2 = dt*(y+0.5*l1), dtl*(vl+0.5*m1), dt*(v+0.5*n1)
            l2, _, _, _, _, _, _, _, _,_, _, n2, m2, _ = \
            f(x+k1/2., y+l1/2., u+ll1/2., v+n1/2., ul+j1/2., vl+m1/2., a+da/2.)
            l2, m2, n2 = l2*dt, m2*dtl, n2*dt
            k3, j3, ll3 = dt*(y+l2), dtl*(vl+m2), dt*(v+n2)
            l3, _, _, _, _, _, _, _, _,_, _, n3, m3, _ = \
            f(x+k2, y+l2, u+ll2, v+n2, ul+j2, vl+m2, a+da)
            l3, m3, n3 = l3*dt, m3*dtl, n3*dt
            
            dx = (k0 + 2.*k1 + 2.*k2 + k3)/6.   ##Field value stepsize
            dy = (l0 + 2.*l1 + 2.*l2 + l3)/6.   ##Field derivative stepsize
            dul = (j0 + 2.*j1 + 2.*j2 + j3)/6.  ##dens.pert. LCDM stepsize
            dvl = (m0 + 2.*m1 + 2.*m2 + m3)/6.  ##its derivative stepsize
            du = (ll0 + 2.*ll1 + 2.*ll2 + ll3)/6. ##dens.pert. STSQ stepsize
            dv = (n0 + 2.*n1 + 2.*n2 + n3)/6.   ##its derivative stepsize
            
   
            DATA[0,i] = 0.0013/a   ##Saving all relevant information to DATA
            DATA[1,i] = a
            DATA[2,i] = z
            DATA[3,i] = Hl
            DATA[4,i] = OMl
            DATA[5,i] = ql
            DATA[6,i] = ODEl
            DATA[7,i] = 0
            DATA[8,i] = x   ##Index depends on worker - ensures DATA  
            DATA[9,i] = rho ##is not overwritten for each model parameter
            DATA[10,i] = V
            DATA[11,i] = w
            DATA[12,i] = H
            DATA[13,i] = OM
            DATA[14,i] = abs((OM-OMl))      
            DATA[15,i] = abs((H-Hl)/Hl)
            DATA[16,i] = q
            DATA[17,i] = ODE
            DATA[18,i] = np.log10((0.5*(y**2.))/V)
            DATA[19,i] = 0
            DATA[20,i] = 0
            DATA[21,i] = 0
            DATA[22,i] = 0
            DATA[23,i] = 0
            DATA[24,i] = 0
            DATA[25,i] = 0
            DATA[26,i] = 0
            DATA[27,i] = V/rho
            
            
            if any(val == 'g' for val in plots) or any(val == 'sig8' for val in plots): 
                ##Saves time - only run code
                ##if we want to
                gl2 = ul/a   ##LCDM growth function from density pertuabtion
                fzl = (dul*a)/(da*ul)   ##f(z)
                gLCDM = (np.log(fzl))/(np.log(OMl))   #gamma for LCDM
                          
                g = u/a   ##STSQ growth function from density pertuabtion
                fz = (du*a)/(da*u)
                gSTSQ = (np.log(fz))/(np.log(OM))   #gamma for STSQ
                
                glarr[i] = dlna*(OMl**(gLCDM) - 1.) ##LCDM growth using Linder integral
                garr[i] = dlna*(OM**(gSTSQ) - 1.) ##STSQ growth using Linder integral
                
                gl = np.exp(glarr[0:i].sum() - glarr[0]) ##Normalise g to 1 at a=0.075
                g2 = np.exp(garr[0:i].sum() - garr[0])                
                
                DATA[28,i] = gSTSQ   ##Growth index STSQ
                DATA[29,i] = g                                         
                DATA[30,i] = abs((gl2 - g)/gl2) 
                DATA[31,i] = 0
                DATA[32,i] = 0
                DATA[33,i] = 0
                DATA[34,i] = 0
                DATA[35,i] = 0
                DATA[36,i] = abs((g-g2)/g)
                DATA[37,i] = fz
                DATA[38,i] = 0
                
                DATA[39,i] = gl2
                DATA[40,i] = abs((gl-gl2)/gl)
                DATA[41,i] = gLCDM   ##Growth index LCDM
                DATA[42,i] = fzl
                
                if i == n-1:
                    DATA[43] /= DATA[8,0]   ##Normalise so g(ai)=1
                    DATA[44] = DATA[8]*(sig80/gl)   ##LCDM sigma8(z)
               

                if i == n-1:
                    DATA[29] /= DATA[29,0]   ##Normalise so g(ai)=1
                    DATA[45] = DATA[29]*(sig80/g)   ##STSQ sigma8(z)
                    DATA[46] = 0
                    DATA[47] = abs((DATA[9]-DATA[29])/DATA[9])
                    ##STSQ sig8 diff
                    DATA[48] = 0
                    ##KGB sig8 diff
                
            if any(val == 'dm' for val in plots) or any(val == 'dL' for val in plots): 
                                                 ##Saves time - only run code
                                                 ##if we want to  
                if i == 0 :   ##dL integration needs only be performed once
                              ##as long as all data is saved
                    
                    a2, x2, y2 = a, x, y  ##Set initial conditions for
                                          ##inner integration
                    
                    for k in range(n):
                        
                        l02, H2, _, _, _, Hl2, HK2, _, _,\
                        _, _, _, _, _ = f(x2, y2, ui, vi, ui, vi, a2) 
                        ##Recalculate parameters for each step of dL integration
                        dt2 = da/(H2*a2)
                        
                        l02, k02 = l02 * dt2, dt2 * y2
                        k12 = dt2 * (y2+0.5*l02)
                        l12, _, _, _, _, _, _, _,_, _, _, _, _, _ = \
                        f(x2+k02/2., y2+l02/2., ui, vi, ui, vi, a2+da/2.)
                        l12 = l12 * dt2
                        k22 = dt2 * (y2+0.5*l12)
                        l22, _, _, _, _, _, _, _,_, _, _, _, _, _ = \
                        f(x2+k12/2., y2+l12/2., ui, vi, ui, vi, a2+da/2.)
                        l22 = l22 * dt2
                        k32 = dt2 * (y2+l22)
                        l32, _, _, _, _, _, _, _,_, _, _, _, _, _ = \
                        f(x2+k22, y2+l22, ui, vi, ui, vi, a2+da)
                        l32 = l32 * dt2
                        
                        dx2 = (k02 + 2.*k12 + 2.*k22 + k32)/6.
                        dy2 = (l02 + 2.*l12 + 2.*l22 + l32)/6.
                        
                        ddm = da/(H2*(a2**2.))  ##Proper motion distance step
                        ddml = da/(Hl2*(a2**2.))
                        dm[k] = ddm    
                        dml[k] = ddml   
                        
                        a2 += da   ##Increment scale factor,
                        x2 += dx2  ##Field value and
                        y2 += dy2  ##derivative of field for dL integration
                    
                    dm0 = np.sum(dm)   ##Total dm at a=ai
                    dml0 = np.sum(dml)
                
                dL = (dm0 - dm[0:i].sum())/a   ##Find dL at any a>ai using saved information
                dLl = (dml0 - dml[0:i].sum())/a   ##Using difference between dm at a=ai  
#                    if i == 4250:    ##Analytic test of dL code using LCDM
#                        print(a,dLl) ##i=4250, dL(z=1)=1.07*10**42
                DATA[49,i] = 0
                DATA[50,i] = dL
                
                
                DATA[51,i] = dLl
                    
                DATA[52,i] = abs((dLl - dL)/dLl)   ##STSQ LCDM dL difference
                DATA[53,i] = 0.01*dL/dLl   ##LISA precision (??)
                DATA[54,i] = 0
            
            a += da   ##Incrmement scale factor, ield values, growth parameters
            y += dy   
            x += dx   
            ul += dul
            vl += dvl
            u += du
            v += dv
            #print(a)   ##Progress checker

    
    
    intstep(xi, yi, ai)   ##Call intestep function using initial values
    
    
    numm = int(NN*n)

    for ii in range(numm):  ##Add flattened temporary data to sharedDATA array
            sharedDATA[j*numm+ii] = DATA.flat[ii]
        
        
        
###############################################################################
###                                                                         ###
###                              Main module                                ### 
###                                                                         ###
###############################################################################
    
##Creates pool of workers for multiprocessing
##Prints final values of relevant cosmological parameter from each model
    
    
def main():
    
    start=time.time()
    
    global sharedDATA  ##Make sharedDATA a global variable so it can be written
                       ##to by each worker
    
    sharedDATA, A, B, zc, order, plots = initarr()  ##Initialise array
    
    ##Create a pool of workers
    pool = Pool()
    pool.map(partial(worker,A=A, B=B, zc=zc, order=order, plots=plots), \
             range(len(A)))
    pool.close()
    pool.join()
    
    l = ((6.*Vc + 3.*rhom0*(1.+zc)**3.)/(2.*Vc))**0.5   ##model dependent constant
    xc = (Mpl/l)*np.log(Mpl**4. / rhode0) 
    
    xnum = NN*len(A)
    
    data = np.asarray(sharedDATA).reshape((xnum,n))
    
    plot(plots, xc, Vc, zc, ai, da, A, B, order, data, NN)
    
    end = time.time()
    
    print('time taken = {0}'.format(end-start))
    
    
if __name__ == '__main__':
    main()

        