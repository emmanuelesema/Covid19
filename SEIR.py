import matplotlib.pyplot as plt
import numpy as np
from ODESolver import *


class Region:
    def __init__(self, name, S_0, E2_0):
        self.name = name
        self.S_0, self.E2_0 = S_0, E2_0 
        self.E1_0= 0
        self.I_0 = 0
        self.Ia_0 = 0
        self.R_0 = 0
        
        self.p = self.S_0, self.E1_0, self.E2_0, self.I_0, self.Ia_0, self.R_0  
        p =self.p
        pop = sum(p) 
        self.population = pop
        
    def set_SEIR_values(self, u, t):
        self.u,self.t = u,t
        self.S = u[:,0]
        self.E1 = u[:,1]
        self.E2 = u[:,2]
        self.I = u[:,3]
        self.Ia = u[:,4]  
        self.R = u[:,5] 
        
    def plot(self):
        plt.plot(self.t,self.S,label='Susceptibles')
        plt.plot(self.t,self.I,label='Infected')
        plt.plot(self.t,self.Ia,label='Infected (asymptomatics)')
        plt.plot(self.t,self.R,label='Recovered') 
        plt.legend() 
        plt.xlabel('Time (days)')
        plt.ylabel('Population')
        plt.title(self.name) 
    

class ProblemSEIR:
    def __init__(self, region, beta, r_ia = 0.1, r_e2=1.25, lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        if isinstance(beta, (float, int)): 
            self.beta = lambda t: beta 
        elif callable(beta):
            self.beta = beta
        self.region = region
        self.r_ia = r_ia  
        self.r_e2 = r_e2 
        self.lmbda_1 = lmbda_1
        self.lmbda_2 = lmbda_2 
        self.p_a = p_a 
        self.mu = mu         
        self.set_initial_condition() 
        
    def set_initial_condition(self):
        U0 = [self.region.S_0, self.region.E1_0, self.region.E2_0, self.region.I_0, self.region.Ia_0, self.region.R_0 ]
        self.initial_condition = U0
        
    def get_population(self):
        ppl = self.region
        return ppl.population
     
    def solution(self, u, t):
        val = self.region.set_SEIR_values(u, t) 
     
    def __call__(self, u, t): 
        S, E1, E2, I, Ia, R = u 
        
        N = sum(u) 
        dS = -self.beta(t)*S*I/N - self.r_ia*self.beta(t)*S*Ia/N - self.r_e2*self.beta(t)*S*E2/N
        dE1 = self.beta(t)*S*I/N + self.r_ia*self.beta(t)*S*Ia/N + self.r_e2*self.beta(t)*S*E2/N - self.lmbda_1*E1
        dE2 = self.lmbda_1*(1-self.p_a)*E1 - self.lmbda_2*E2
        dI = self.lmbda_2*E2 - self.mu*I 
        dIa = self.lmbda_1*self.p_a*E1 - self.mu*Ia 
        dR = self.mu*(I + Ia)
        return [dS, dE1, dE2, dI, dIa, dR]

class SolverSEIR:
    def __init__(self, problem, T, dt):
        self.problem = problem
        self.T = T
        self.dt = dt
        self.total_population = (problem.get_population())

    def solve(self, method=RungeKutta4):
        solver = method(self.problem) 
        solver.set_initial_condition((self.problem.initial_condition) )
        n = int(self.T/self.dt)   
        t = np.linspace(0, self.T, n)
        u, t = solver.solve(t) 
        
        # Send the values of S, E1, E2, I, Ia, R, and t
        # from the Problem class to the Region class:
        (self.problem.solution(u, t))

if __name__ == '__main__':
    nor = Region('Norway',S_0=5e6,E2_0=100)
    print(nor.name, nor.population)
    print()  
    S_0, E1_0, E2_0 = nor.S_0, nor.E1_0, nor.E2_0
    I_0, Ia_0, R_0 = nor.I_0, nor.Ia_0, nor.R_0
    print(f'S_0 = {S_0}, E1_0 = {E1_0}, E2_0 = {E2_0}')
    print(f'I_0 = {I_0}, Ia_0 = {Ia_0}, R_0 = {R_0}')
    u = np.zeros((2,6)) 
    u[0,:] = [S_0, E1_0, E2_0, I_0, Ia_0, R_0]
    nor.set_SEIR_values(u,0)
    print()
    print(nor.S, nor.E1, nor.E2, nor.I, nor.Ia, nor.R)
    
    problem = ProblemSEIR(nor,beta=0.5)
    problem.set_initial_condition()
    print(f'Initial Conditions: {problem.initial_condition}')
    print() 
    print(f'Total Population: {problem.get_population()}')
    print()
    print(problem([1,1,1,1,1,1],0))

    solver = SolverSEIR(problem,T=100,dt=1.0)
    solver.solve() 
    nor.plot()  
    plt.legend()
    plt.show() 

    


