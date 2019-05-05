# -*- coding: utf-8 -*-
"""
Script Used to check attractor solution in STSQ (KGB attractor in development)
Only works with DENSEwin.py / plotswin.py

@author: Charlie
"""

import time
from os import sys
import numpy as np
from plots import plot
from scipy.optimize import fsolve
from scipy.misc import derivative
from multiprocessing.pool import ThreadPool as Pool


###############################################################################
###                                                                         ###
###                               User input                                ### 
###                                                                         ###
###############################################################################

###Used to decide which parts of the code are run to produce the desired graphs
###For brevity, if yes chosen for 'test' option simple graphs with nominal
###parameter choices is run, takes approx 5 seconds. 
##Used to test functionality of code



zc=10
plots=['att']
nKGB=[1.]
KGB='n'
#rc = np.zeros(len(nKGB))
        
        
        
###############################################################################
###                                                                         ###
###                            Defining Parameters                          ### 
###                                                                         ###
###############################################################################

##Curret values for cosmological parameters, as well as model dependent constants
##are set here. Also step size, and DATA array for the main integration module

ai = 0.01   ##Initial scale factor
da = 0.0001   ##Scale factor step size
  ##Number of steps in the integration
DEL = 39   ##Number for DATA array indicies functionaility
  ##Big array to hold all calculated DATA

H0 = 1.437722539*10.**-42.   ##Current value in GeV from Planck 2018
#H0 = 1.535608363*10**-42.   ##Current value in GeV from Bonvin 2017
Mpl = 2.4*10.**18.   ##Reduced Planck mass in GeV
rhoc0 = 3.*(H0*Mpl)**2.   ##Current critical density
OmegaM0, OmegaR0, OmegaDE0 = 0.3, 8.25*10**-5., 0.7   ##Current density fractions
rhode0 =  OmegaDE0*rhoc0   ##Current dark energy energy density
rhom0 = OmegaM0*rhoc0   ##Current matter energy density                
rhor0 = OmegaR0*rhoc0   ##Current raditation energy density
sig80 = 0.806   ##simga_8,0 DES+Planck 2018
ac = 1./(1.+zc)   ##Crossover scalefactor
Vc = rhode0   ##Critical potential
l = ((6.*Vc + 3.*rhom0*(1.+zc)**3.)/(2.*Vc))**0.5   ##model dependent constant
xc = (Mpl/l)*np.log(Mpl**4. / rhode0)   ##Critical field value                                          
xiV = (Mpl/l)*np.log(((Mpl**4.)/(rhode0))*(ai/ac)**3.) #+ 0.005*(Mpl/l)*np.log(((Mpl**4.)/(rhode0))*(ai/ac)**3.)   ##Initial field value                            
yiV = (2.*np.exp(np.log(Mpl**4.) - l*xiV/Mpl))**0.5   ##Initial derivative
#xKi = 1.
#yKi = 1.
ui = 0.075
vi = 2.87470*10**-42.
vil = vi#2.87496*10**-42.
viK = vi#2.88*10**-42.

lngth=10
fact = np.zeros(3*lngth)
for ii in range(len(fact)):
    fact[ii] = -lngth+ii
eats = np.zeros(lngth) 
nKGB = np.zeros(lngth) 


aa=7.5*10**15
bb=7.5*10**-24
for jjj in range(lngth):
    
    eats[jjj] = xiV + (fact[jjj]+2)*aa
    nKGB[jjj] = yiV + (fact[jjj]+2)*bb
#    eats[jjj] = xiV + np.random.randint(-10,10)*2.5*10**15
#    nKGB[jjj] = yiV + np.random.randint(-10,10)*10**-24
    
print(np.exp(np.log(Mpl**4.) - l*xiV/Mpl)-l*(Mpl**3.)*np.exp(-l*xiV/Mpl), yiV**2.)
print(np.exp(np.log(Mpl**4.) - l*(xiV+lngth*aa)/Mpl)-l*(Mpl**3.)*np.exp(-l*(xiV+lngth*aa)/Mpl), (yiV+lngth*bb)**2.)
n = int((1-ai)/da)
DATA = np.zeros((5000,int(n))) 

###############################################################################
###                                                                         ###
###                        Parallelised integration                         ### 
###                                                                         ###
###############################################################################

##The worker module contains all the computation required for each user defined 
##value of eta for STSQ or n for KGB. This setup allows each different model
##to be simulated simuLtaneously - making use of available computing power and saving time.

zpoints = []
fpoints = []
    
def worker(j):   ##Create loop over all code so that it can be parallelised  
                 ##Each worker performs calculations for each user-defined eta/nKGB
    xi = eats[j]
    yi = nKGB[j]
    chi = np.zeros(6)

            
            
    def f(x, y, yK, u, v, ul, vl, uK, vK, a):   ##Calculates useful parameters                
        
        def V(x):   ##Function for the STSQ potential and its derivative
                    ##Potetial can be changed for any Scalar Field model
            try:                                                                
                if x < xc:
                    return np.exp(np.log(Mpl**4.) - l*x/Mpl), \
                            -l*(Mpl**3.)*np.exp(-l*x/Mpl)
                else:
                    return np.exp(np.log(Mpl**4.) - (l*xc/Mpl) + \
                                  0.1*l*(x-xc)/(Mpl)), \
                                  0.1*l*(Mpl**3.)*np.exp(-l*xc/Mpl + \
                                  0.1*l*(x-xc)/(Mpl))                      
                                  
            except IndexError:   ##In case number of KGB elements is larger
                return 10**-46., 0.   ##than number of STSQ elements
            
   
        Vp, Vdash = V(x)##Potential and its derivative
        rhor = rhor0*(a**-4.)   ##Radiation density
        rhoml = rhom0*(a**-3.)  ####Matter density in LCDM
        rhom = (((l**2.)/3.) - 1.)*2.*V(xi)[0]*(ai/a)**3.   ##Matter density in STSQ
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

        if KGB == 'y' :   ##Wether KGB code should run - user defined
           
            try:          ##try except clause in case j runs out of 
                          ##range of nKGB (eats is larger)
                func = lambda Hu : (Hu/H0)**2. - (1. - OmegaM0 - OmegaR0)*\
                (Hu/H0)**(-2./(2.*nKGB[j]-1.))-OmegaM0*(a**-3.)-OmegaR0*(a**-4.) ##KGB Friedman
                H_guess = 10**-42.   ##Guess for fsolve
                HKGB = fsolve(func, H_guess)   ##Use fsolve module to solve Friedmann
                OmegaRK = rhor/(3.*(Mpl*HKGB)**2.) ##Rad density fraction in KGB
                OmegaMKGB = rhoml/(3.*(Mpl*HKGB)**2.) ##Matter density fraction in KGB
                weff = (-6.*nKGB[j]-OmegaRK)/(3.*(2.*nKGB[j]-OmegaMKGB-OmegaRK)) ##Effective barotropic parameter in KGB
                qK =  -1. + ((3.*OmegaMKGB+4.*OmegaRK)*(2.*nKGB[j]-1.))*(2.*(2.*nKGB[j]\
                            - OmegaMKGB-OmegaRK))**-1.   ##Deceleration parameter in KGB
                FKGB = (HKGB*yK*(3.*OmegaMKGB+4.*OmegaRK)*(2.*nKGB[j]-1.))*(2.*(2.*nKGB[j]-1.)\
                        *(2.*nKGB[j] - OmegaMKGB - OmegaRK))**-1.   ##Second derivative of KGB potential
                Geff = (2.*nKGB[j]+(3.*nKGB[j]-1.)*OmegaMKGB)/(OmegaMKGB*(5.*\
                       nKGB[j] - OmegaMKGB))
            except IndexError:    ##If out of index range, set parameters to zero
                HKGB, OmegaMKGB, weff, qK, FKGB, Geff = 1., 2., 0., 0., 0., 0.
        
        else:   ##If user decides no KGB, set parameters to zero       
            HKGB, OmegaMKGB, weff, qK, FKGB, Geff = 1., 2., 0., 0., 0., 0.

        return -3.*H*y - Vdash, H, rhophi, wphi, Vp, Hl, OmegaM, OmegaMl, \
                OmegaMKGB, OmegaDE, OmegaDEl, weff, HKGB, q, ql, qK, FKGB,\
                -2.*HKGB*vK+0.5*(Mpl**-2.)*rhom*uK*Geff, -2.*H*v+\
                0.5*(Mpl**-2.)*rhom*u, -2.*Hl*vl+0.5*rhoml*(Mpl**-2.)*ul    ##Return all parameters for use        
  
       
    def intstep(x, y, xK, yK, a):    ##Performs 4th order RK integration to 
                                     ##solve scalar field equation and more

  
        
        u, ul, uK = ui, ui, ui   ##Initial conitions for growth ODE
        v, vl, vK = vi, vil , viK
        
        for i in range(int(n)):   ##Integrate in n steps of da from ai to 1

            l0, H, rho, w, V, Hl, OM, OMl, OMK, ODE, ODEl, weff, HKGB, q, ql, \
            qK, FK, p0, n0, m0 = f(x, y, yK, u, v, ul, vl, uK, vK, a) ##Find all parameters at current a
            z = (1./a)-1.   ##Redshift
            dlna = da/a   ##Logarithmic step in a
            dtl = da/(Hl*a)   ##Timestep for LCDM
            dt = da/(H*a)   ##Timestep for STSQ
            dtK = da/(HKGB*a)   ##Timestep for KGB
            dyK = FK * dtK   ##Integration steps for KGB field equation
            dxK = yK * dtK
            
            ##4th Order Runge-Kutta integration method
            l0, m0, n0, p0 = l0*dt, m0*dtl, n0*dt, p0*dtK
            k0, j0, ll0, r0 = dt*y, dtl*vl, dt*v, dtK*vK
            k1, j1, ll1, r1 = dt*(y+0.5*l0), dtl*(vl+0.5*m0), dt*(v+0.5*n0), dtK*(vK+0.5*p0)
            l1, _, _, _, _, _, _, _, _, _, _, _, _, _, _,_, _, p1, n1, m1 = \
            f(x+k0/2., y+l0/2., yK, u+ll0/2., v+n0/2., ul+j0/2., vl+m0/2., uK+r0/2., vK+p0/2., a+da/2.)
            l1, m1, n1, p1 = l1*dt, m1*dtl, n1*dt, p1*dtK
            k2, j2, ll2, r2 = dt*(y+0.5*l1), dtl*(vl+0.5*m1), dt*(v+0.5*n1), dtK*(vK+0.5*p1)
            l2, _, _, _, _, _, _, _, _, _, _, _, _, _, _,_, _, p2, n2, m2 = \
            f(x+k1/2., y+l1/2., yK, u+ll1/2., v+n1/2., ul+j1/2., vl+m1/2., uK+r1/2., vK+p1/2., a+da/2.)
            l2, m2, n2, p2 = l2*dt, m2*dtl, n2*dt, p2*dtK
            k3, j3, ll3, r3 = dt*(y+l2), dtl*(vl+m2), dt*(v+n2), dtK*(vK+p2)
            l3, _, _, _, _, _, _, _, _, _, _, _, _, _, _,_, _, p3, n3, m3 = \
            f(x+k2, y+l2, yK, u+ll2, v+n2, ul+j2, vl+m2, uK+r2, vK+p2, a+da)
            l3, m3, n3, p3 = l3*dt, m3*dtl, n3*dt, p3*dtK
            
            dx = (k0 + 2.*k1 + 2.*k2 + k3)/6.   ##Field value stepsize
            dy = (l0 + 2.*l1 + 2.*l2 + l3)/6. 
            
            
            if j == 0:   ##LCDM parameters only need to be added to DATA once
                DATA[0,i] = 0.0013/a   ##Saving all relevant information to DATA
                DATA[1,i] = a
                DATA[2,i] = z
                DATA[3,i] = Hl
                DATA[4,i] = OMl
                DATA[5,i] = ql
                DATA[6,i] = ODEl
                DATA[7,i] = 0.
            DATA[1+(j+1)*DEL,i] = x   ##Index depends on worker - ensures DATA  
            DATA[2+(j+1)*DEL,i] = rho ##is not overwritten for each model parameter
            DATA[3+(j+1)*DEL,i] = V
            DATA[4+(j+1)*DEL,i] = w
            DATA[5+(j+1)*DEL,i] = H
            DATA[6+(j+1)*DEL,i] = OM
            DATA[7+(j+1)*DEL,i] = abs((OM-OMl))      
            DATA[15+(j+1)*DEL,i] = abs((H-Hl)/Hl)
            DATA[16+(j+1)*DEL,i] = q
            DATA[18+(j+1)*DEL,i] = ODE
            DATA[19+(j+1)*DEL,i] = y**2.
            DATA[20+(j+1)*DEL,i] = weff
            DATA[21+(j+1)*DEL,i] = HKGB**-1.
            DATA[22+(j+1)*DEL,i] = OMK
            DATA[23+(j+1)*DEL,i] = abs((HKGB-Hl)/Hl)
            DATA[25+(j+1)*DEL,i] = qK
            DATA[26+(j+1)*DEL,i] = xK 
            DATA[27+(j+1)*DEL,i] = yK
            DATA[28+(j+1)*DEL,i] = y
            
                  ##LCDM sigma8(z)
               

              ##KGB LCDM dL difference
            
            a += da                                                             ##Incrmement scale factor
            y += dy                                                             ##field derivative and
            x += dx  
            yK += dyK
            xK += dxK 
            #print(a)                                                           ##Progress checker

                
    intstep(xi, yi, xiV, yi/50., ai)                                               ##Call intestep function using initial values
    
###############################################################################
###                                                                         ###
###                              Main module                                ### 
###                                                                         ###
###############################################################################
    
##Creates pool of workers for multiprocessing
##Prints final values of relevant cosmological parameter from each model
    
    
def main():
    start = time.time()

    pool = Pool()
    pool.map(worker, range(len(eats)))
    pool.close()
    pool.join()
        

    plot(plots, xc, Vc, zc, DEL, eats, nKGB, 'n', DATA, zpoints, fpoints, sig80)
    end = time.time()
    
    print('time taken = {0}'.format(end-start))
if __name__ == "__main__":
    main()
