# -*- coding: utf-8 -*-
"""
Created on Wed Jun 1 09:00:02 2022

@author: Tib
"""

import numpy as np
import qutip as qt
import multiprocess as mp
from multiprocess import Process, Queue
import scipy as sp
import math

def vector2(d):
    out = np.empty([d,d],dtype=object)
    for n in range(d):
        for m in range(d):
            out[n,m] = (qt.states.basis(d,n)*qt.states.basis(d,m).dag())
    return out

def matpower(matrix,n):
    y = (( np.linalg.matrix_power(matrix, n)) / int(math.factorial(n)))
    return y

def expmToN(mat,order):
    result = sum([matpower(mat,k) for k in range(0,order+1)])
    return result


class MultiLevel:
    
    def __init__(self, N, D, geff, ep, wc, wa, kappa, gamma, gamma_d, theta, omega=0, zeta=0, displacement = 0, rwa=True):
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
        self.rwa = rwa
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
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators vec(n,m) = |n><m|
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
        
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,0]*self.vec[0,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a 
    def hamiltonian_nodriving(self):
        if self.rwa==True:
            #constructing hamiltonian in RWA
            self.H_i = sum([self.glist[n-1]*(self.adag*self.vec[0,n] + self.a*self.vec[n,0]) for n  in  range(1,self.D)])
        
        elif self.rwa==False:
            #constructing hamiltonian without RWA
            self.H_i = sum([self.glist[n-1]*(self.adag + self.a)*(self.vec[0,n] + self.vec[n,0]) for n  in  range(1,self.D)])
        
        self.H = self.wc*self.adag*self.a + sum([(self.wa + self.delta[i-1])*self.vec[i,i] for i in range(1,self.D)]) \
            + self.H_i
        return self.H
    
    def hamiltonian_withdriving(self):
        self.V = self.omega*(self.a + self.adag) + (sum([self.zeta2*self.vec[0,n] + self.zeta1*self.vec[n,0] for n in range(1,self.D)]))
        self.wl_list = np.linspace(self.start + self.wc, self.end +self.wc, self.accuracy)
        self.Htot = np.empty([self.accuracy],dtype=object)
        for i in range(self.accuracy):
            self.Htot[i] = self.H + self.V - self.wl_list[i]*self.adag*self.a \
                - self.wl_list[i]*sum([self.vec[n,n] for n in range(1,self.D)]) #needs to be original as done after disp transform
        return self.Htot
        
    def hamiltonian(self, accuracy=0, start=0, end=0, suppress=False):
        if self.omega==0 and self.zeta1==0:
            if suppress==False:
                print("hamiltonian_nodriving working...")
            else:
                pass
            return self.hamiltonian_nodriving()
        elif accuracy ==0:
            if suppress==False:
                print("hamiltonian_withdriving working... (single), confusion with take away laser frequency, automatically 0")
            else:
                pass
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

    def g2listcalc(self,operator):
        self.g2list = np.empty([len(self.Htot)],dtype=np.float64)
        for i in range(len(self.wl_list)):
            self.g2list[i] = qt.coherence_function_g2(self.Htot[i], None, [0], self.c_ops, operator)[0][0]
            print(i/len(self.wl_list))
        return self.g2list

    def g2listcalcmp(self,operator):
      num_sims = len(self.Htot)
      num_threads = mp.cpu_count() if mp.cpu_count()<num_sims else num_sims
      manager = mp.Manager()
      return_dict = manager.dict()
      jobs = []

      def g2listcalcmp_helper(start,end,procnum) -> None:
        ## TODO: refactor this into multiple functions
        self.g2list_temp = np.empty([end-start],dtype=np.float64)
        for s in range(start,end):
          self.g2list_temp[s-start] = np.real(qt.coherence_function_g2(self.Htot[s], None, [0], self.c_ops, operator)[0][0])

          print(f"Process #{procnum}: {int(((s-start)/(end-start))*100)}% complete")

        print(f"Process #{procnum}: 100% complete")

        return_dict[procnum] = self.g2list_temp

      ## Create processes and dynamically allocate simulations
      for i in range(num_threads):
        start_index = 0 if i==0 else int(i/num_threads * num_sims)
        end_index = num_sims if i+1 == num_threads else int((i+1)/num_threads * num_sims)

        p = Process(target=g2listcalcmp_helper, args=[start_index,end_index,i])
        jobs.append(p)
        p.start()

      for p in jobs: # wait for all jobs to finish before continuing
        p.join()

      ## Stitch values returned from each process back together into a single array
      self.g2list = np.empty([num_sims],dtype=np.float64)
      index = 0
      for i in range(num_threads):
        for j in return_dict[i]:
          self.g2list[index] = j
          index+=1

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
        
class DegenBlochSiegert:
    
    def __init__(self, N, D, geff, wc, wa):
        self.N=N
        self.D=D
        self.geff=geff
        self.wc=wc
        self.wa=wa
        self.g=self.geff/np.sqrt(self.D-1)
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
                
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a
        
        self.Op = sum([self.vec[n,0] for n in range(1,self.D)])
        self.Om = self.Op.dag()
        
        self.Oz = sum([self.vec[n,n] for n in range(1,self.D)]) - self.vec[0,0]
        self.I=qt.tensor(qt.operators.qeye(self.N),qt.operators.qeye(self.D))
        
    def hamiltonian(self):
        self.H=self.wc*self.adag*self.a + 0.5*self.wa*(self.Oz+self.I) + self.g*(self.a*self.Op + self.adag*self.Om) + self.g**2/(self.wa+self.wc)*(self.adag*self.a*(self.Op*self.Om-self.Om*self.Op)-(self.D-1)/2*(self.I-self.Oz))
        return self.H
    
class GeneralBlochSiegert:
    
    def __init__(self, N, D, geff, ep, wc, wa):
        self.N=N
        self.D=D
        self.geff=geff
        self.ep=ep
        self.wc=wc
        self.wa=wa
        self.beta=wc+wa
        self.glist = np.linspace(self.geff/np.sqrt(self.D-1),self.geff/np.sqrt(self.D-1),self.D-1)
        self.a = qt.tensor(qt.operators.destroy(self.N), qt.operators.qeye(self.D))
        self.adag = self.a.dag()
        self.vectorsmat = vector2(self.D) #basis * basis.dag matrix
        self.vec = np.empty([self.D,self.D],dtype=object) #atomic generalised ladder operators
        self.Lambda=[self.glist[k-1]/(self.beta+self.ep*((k-1)/(self.D-1)-0.5)) for k in range(1,self.D)]
        self.phi = 1/(2*wc) * np.array([[self.Lambda[k-1]*self.glist[j-1] for j in range(1,self.D)] for k in range(1,self.D)])
        for n in range(self.D):
            for m in range(self.D):
                self.vec[n,m] = qt.tensor(qt.operators.qeye(self.N),self.vectorsmat[n,m]) #vec[n,m].dag = vec[m,n]
            
        self.n_op_tot = self.adag*self.a + sum([self.vec[n,n] for n in range(1,self.D)])
        self.n_op = self.adag*self.a
        
        self.Op = sum([self.vec[n,0] for n in range(1,self.D)])
        self.Om = self.Op.dag()
        
        self.Oz = sum([self.vec[n,n] for n in range(1,self.D)]) - self.vec[0,0]
        self.I=qt.tensor(qt.operators.qeye(self.N),qt.operators.qeye(self.D))    
        
        self.U1 = np.real(sp.linalg.expm(sum([self.Lambda[k-1]*(self.adag*self.vec[k,0] - self.a*self.vec[0,k]) for k in range(1,self.D)])))
        self.U2 = np.real(sp.linalg.expm(sum([sum([self.phi[j-1,k-1] * (self.adag**2 * qt.operators.commutator(self.vec[j,0],self.vec[0,k])\
                                                                -self.a**2 * qt.operators.commutator(self.vec[k,0],self.vec[0,j]))for j in range(1,self.D)]) for k in range(1,self.D)])))
        self.U = qt.Qobj(np.real(self.U2 @ self.U1), dims=[[self.N, self.D], [self.N, self.D]])
        self.Udag = self.U.dag()
    
    def hamiltonian(self):
        self.H_0 = self.wc*self.adag*self.a + 0.5*self.wa*(self.Oz+self.I)
        self.H_ep = self.ep/(self.D-2) * sum([(k-1)*self.vec[k,k] for k in range(1,self.D)]) -self.ep/4 * (self.I+ self.Oz)
        self.H_r = self.a*sum([self.glist[k-1]*self.vec[k,0] for k in range(1,self.D)]) + self.adag*sum([self.glist[k-1]*self.vec[0,k] for k in range(1,self.D)])
        self.H_n = self.adag*self.a*sum([sum([(qt.operators.commutator(self.vec[k,0],self.vec[0,j])*\
                        (self.glist[j-1]*self.Lambda[k-1]+self.glist[k-1]*self.Lambda[j-1]\
                        -(self.beta+self.ep*(j+k-2)/(2*(self.D-2)) -self.ep/2)*self.Lambda[j-1]*self.Lambda[k-1])) for j in range(1,self.D)]) for k in range(1,self.D)])
        self.H_i = -0.5*(self.I-self.Oz)*sum([2*self.glist[j-1]*self.Lambda[j-1]\
                        -(self.beta+self.ep*(j-1)/(self.D-2)-self.ep/2)*self.Lambda[j-1]**2 for j in range(1,self.D)])
        self.H = self.H_0 + self.H_ep + self.H_r + self.H_n + self.H_i
        return self.H
    
class Dicke:
    
    def __init__(self, N, M, g, wc, wa):
        self.N = N #cavity levels 
        self.M = M #number of atoms
        self.g = g #coupling
        self.wc = wc #cavity frequency
        self.wa = wa #atomic frequency
        
        self.j = M/2
        self.n = 2*self.j+1
        self.a  = qt.tensor(qt.operators.destroy(N), qt.operators.qeye(int(self.n)))
        self.adag = self.a.dag()
        
        self.Jp = qt.tensor(qt.operators.qeye(N), qt.operators.jmat(self.j, '+'))
        self.Jm = qt.tensor(qt.operators.qeye(N), qt.operators.jmat(self.j, '-'))
        self.Jz = qt.tensor(qt.operators.qeye(N), qt.operators.jmat(self.j, 'z'))
    
    def hamiltonian(self):
        self.H = self.wc*self.adag*self.a + self.wa*self.Jz + self.g/np.sqrt(self.M)*(self.a+self.adag)*(self.Jp+self.Jm)
        return self.H