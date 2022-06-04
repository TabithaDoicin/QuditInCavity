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
    
    def __init__(self, N, g, wc, wa, kappa, gamma, gamma_d, theta, omega):
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
        #system operators
        self.a = qt.tensor(qt.operators.destroy(self.N),qt.operators.qeye(2))
        self.adag = self.a.dag()
        self.sm = qt.tensor(qt.operators.qeye(self.N),qt.operators.destroy(2))
        self.smdag = self.sm.dag()
        
    def hamiltonian(self):
        #constructing hamiltonian in RWA
        self.H = self.wc*self.adag*self.a + self.wa*self.smdag*self.sm + self.g*(self.adag*self.sm + self.a*self.smdag)
        
        self.V = self.omega*(self.a + self.adag)
        
        if self.omega==0:
            return self.H
        elif self.omega!=0:
            return self.H, self.V
        
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.sm]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.smdag*self.sm]
        self.coop_pumping = [np.sqrt(self.theta)*self.smdag]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops
        
class MultiLevel:
    
    def __init__(self, N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, theta, omega):
        #system variables
        self.N = N
        self.D = D
        self.geff = geff
        self.ep = ep
        self.wc = wc
        self.wa = wa
        self.kappa = kappa
        self.gamma = gamma
        self.gamma_d = gamma_d
        self.theta = theta
        self.omega = omega
        #multilevel energies
        self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        self.delta = np.linspace(-self.ep,self.ep,self.D-1)
        #system operators - cavity
        self.a  = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        #system operators - atom
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        
    def hamiltonian(self):
        #constructing hamiltonian in RWA
        self.H = self.wc*self.adag*self.a + sum([(self.wa + self.delta[i-1])*self.vec[i,i] for i in range(1,self.D)]) \
            + sum([self.glist[n-1]*(self.adag*self.vec[0,n] + self.a*self.vec[n,0]) for n  in  range(1,self.D)])
        
        self.V = self.omega*(self.a + self.adag)
        
        if self.omega==0:
            return self.H
        elif self.omega!=0:
            return self.H, self.V
    
    def collapse(self):
        #collapse operators
        self.coop_cavity_decay = [np.sqrt(self.kappa)*self.a]
        self.coop_radiative_decay = [np.sqrt(self.gamma)*self.vec[0,n] for n in range(1,self.D)]
        self.coop_dephasing = [np.sqrt(self.gamma_d)*self.vec[n,n] for n in  range(1,self.D)]
        self.coop_pumping = [np.sqrt(self.theta/(self.D-1))*self.vec[n,0] for n in range(1,self.D)]
        
        self.c_ops = self.coop_cavity_decay + self.coop_radiative_decay + self.coop_dephasing + self.coop_pumping
        
        return self.c_ops