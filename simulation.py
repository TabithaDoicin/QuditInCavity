# -*- coding: utf-8 -*-
"""
Created on Wed Jun 1 09:00:02 2022

@author: Tib
"""

import numpy as np
import qutip as qt


def vector2(d):
    out = np.empty([d,d],dtype=object)
    for n in range(d):
        for m in range(d):
            out[n,m] = (qt.states.basis(d,n)*qt.states.basis(d,m).dag())
    return out

class JC:
    
    def __init__(self, N, g, wc, wa, kappa, gamma, gamma_d, theta, omega=0, zeta=0):
        #system variables
        self.N = N
        self.g = g
        self.wc = wc
        self.wa = wa
        self.kappa = kappa
        self.gamma = gamma
        self.gamma_d = gamma_d
        self.theta = theta
        self.omega = omega
        self.zeta = zeta
        #system operators
        self.a = qt.tensor(qt.operators.destroy(self.N),qt.operators.qeye(2))
        self.adag = self.a.dag()
        self.sm = qt.tensor(qt.operators.qeye(self.N),qt.operators.destroy(2))
        self.smdag = self.sm.dag()
        
    def hamiltonian(self, accuracy=0, start=0, end=0):
        #constructing hamiltonian in RWA
        self.accuracy = accuracy
        self.start = start
        self.end = end
        self.H = self.wc*self.adag*self.a + self.wa*self.smdag*self.sm + self.g*(self.adag*self.sm + self.a*self.smdag) 
        
        if self.omega==0:
            return self.H
        elif self.omega!=0:
            if accuracy==0:
                self.H = self.H + self.omega*(self.a + self.adag)
            else:
                if self.start==0 and self.end==0:
                    self.start = -np.pi*self.g + self.wc
                    self.end = np.pi*self.g + self.wc
                else:
                    print(self.start)
                    print(self.end)
                    pass
                self.V = self.omega*(self.a + self.adag)+self.zeta*(self.sm - self.smdag)
                self.wl_list = np.linspace(self.start + self.wc, self.end + self.wc, accuracy)
                self.Htot = np.empty([accuracy],dtype=object)
                for i in range(accuracy):
                    self.Htot[i] = self.H + self.V - self.wl_list[i]*self.adag*self.a - self.wl_list[i]*self.smdag*self.sm
                return self.Htot
        
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.sm]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.smdag*self.sm]
        self.coop_pumping = [np.sqrt(self.theta)*self.smdag]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops

    def g2listcalc(self):
        self.g2list = np.empty([len(self.Htot)],dtype=np.float64)
        for i in range(len(self.wl_list)):
            self.g2list[i] = qt.coherence_function_g2(self.Htot[i], None, [0], self.c_ops, self.a)[0][0]
        return self.g2list
        
class MultiLevel:
    
    def __init__(self, N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, theta, omega=0, zeta=0, displacement = 0):
        #system variables
        self.N = N #max cavity population
        self.D = D #atomic levels
        self.geff = geff #strength of atom cavity interaction
        self.ep = ep #atomic energy level spacing
        self.wc = wc #cavity frequency
        self.wa = wa #atom frequency
        self.kappa = kappa #cavity decay
        self.gamma = gamma #radiative devay
        self.gamma_d = gamma_d #dephasing
        self.theta = theta #pumping
        self.omega = omega #cavity driving
        
        self.zeta1 = zeta #atomic driving first
        self.zeta2 = zeta.conjugate() #atomic driving second 
        
        self.alpha = displacement
        #multilevel energies
        self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        
        if self.D == 2:
            self.delta = [0]
        else:
            self.delta = np.linspace(-self.ep/2,self.ep/2,self.D-1)
            
        #system operators - cavity - displaced automatically by alpha
        self.a  = qt.tensor(qt.displace(N,self.alpha).dag()*qt.operators.destroy(self.N)*qt.displace(N,self.alpha), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        
        self.aori  = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adagori = self.aori.dag()
        #system operators - atom
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
    
    def hamiltonian_nodriving(self):
        #constructing hamiltonian in RWA
        self.H = self.wc*self.adag*self.a + sum([(self.wa + self.delta[i-1])*self.vec[i,i] for i in range(1,self.D)]) \
            + sum([self.glist[n-1]*(self.adag*self.vec[0,n] + self.a*self.vec[n,0]) for n  in  range(1,self.D)])
        return self.H


    def hamiltonian_withdriving(self):
        self.V = self.omega*(self.a + self.adag) + (sum([self.zeta2*self.vec[0,n] + self.zeta1*self.vec[n,0] for n in range(1,self.D)]))
        self.wl_list = np.linspace(self.start + self.wc, self.end +self.wc, self.accuracy)
        self.Htot = np.empty([self.accuracy],dtype=object)
        for i in range(self.accuracy):
            self.Htot[i] = self.H + self.V - self.wl_list[i]*self.adag*self.a \
                - self.wl_list[i]*sum([self.vec[n,n] for n in range(1,self.D)])
        return self.Htot
        
    def hamiltonian(self, accuracy=0, start=0, end=0):
        if self.omega==0 and self.zeta1==0:
            print("hamiltonian_nodriving working...")
            return self.hamiltonian_nodriving()
        elif accuracy ==0:
            print("hamiltonian_withdriving working... (single), confusion with take away laser frequency")
            self.H = self.hamiltonian_nodriving() + self.omega*(self.a + self.adag) + (sum([self.zeta2*self.vec[0,n] + self.zeta1*self.vec[n,0] for n in range(1,self.D)]))
            return self.H
        else:
            print("hamiltonian_withdriving working... (multiple)")
            if start==0 and end==0:
                start = -np.pi*self.geff + self.wc
                end = np.pi*self.geff + self.wc
                print("automatic driving bounds used")
            else:
                print("manual driving bounds used")
                pass
            self.start = start
            self.end = end
            self.accuracy = accuracy
            self.hamiltonian_nodriving()
            return self.hamiltonian_withdriving()
    
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(1,self.D)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(1,self.D)]
        self.coop_pumping = [np.sqrt(self.theta/(self.D-1))*self.vec[n,0] for n in range(1,self.D)]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops
    
    def g2listcalc(self):
        self.g2list = np.empty([len(self.Htot)],dtype=np.float64)
        for i in range(len(self.wl_list)):
            self.g2list[i] = qt.coherence_function_g2(self.Htot[i], None, [0], self.c_ops, self.a)[0][0]
        return self.g2list
    
    def ss_dm(self, driving=False): #steady state density matrix
        if driving == False:
            self.ss_dm = qt.steadystate(self.H,self.c_ops)
            return self.ss_dm
        elif driving==True:
            self.ss_dm = np.empty([self.accuracy],dtype=object)
            for i in range(self.accuracy):
                self.ss_dm[i] = qt.steadystate(self.Htot[i], self.c_ops)
            return self.ss_dm
        
    def darkstate_proportion(self, driving=False):
        self.bright = sum([self.glist[n-1]*qt.states.basis(self.D,n) for n in range(1,self.D)])
        if driving == False:
            self.pdark = 1-(self.ss_dm*(self.vec[0,0] + qt.tensor(qt.operators.qeye(self.N), self.bright*self.bright.dag())/self.geff**2)).tr()
            return np.real(self.pdark)
        elif driving == True:
            self.pdark = np.empty([self.accuracy],dtype=object)
            for i in range(self.accuracy):
                self.pdark[i] = np.real(1-(self.ss_dm[i]*(self.vec[0,0] + qt.tensor(qt.operators.qeye(self.N), self.bright*self.bright.dag())/self.geff**2)).tr())
            return self.pdark
