import matplotlib.pyplot as plt
import numpy as np
from ODESolver import *
from SEIR import *

class RegionInteraction(Region):
    def __init__(self, name, S_0, E2_0, lat, long):
        self.lat = lat * (np.pi/180)
        self.long = long * (np.pi/180)
        super().__init__(name, S_0, E2_0)
    
    def distance(self,other): 
        #ø = lat
        #λ = long
        self.other = other
        delta_ij = (np.arccos(np.sin(self.lat)*np.sin(other.lat) + np.cos(other.lat) * np.cos(self.lat) * np.cos(abs(self.long - other.long))))
        if delta_ij > 1:
            delta_ij = 1
        self.d = 64 * delta_ij 
        return self.d
   
class ProblemInteraction(ProblemSEIR):
    def __init__(self, region, area_name, beta, r_ia = 0.1, r_e2=1.25,lmbda_1=0.33, lmbda_2=0.5, p_a=0.4, mu=0.2):
        self.region = [region]
        self.area_name = area_name
        self.beta = beta
        super().__init__(region, beta) 
    def get_population(self):
        N = 0
        for area in self.region:
            N += area.population
        return N 
    def set_initial_condition(self):   
        self.initial_condition = []
        for area in self.region:
            self.initial_condition += [area.S_0, area.E1_0, area.E2_0, 
                                       area.I_0, area.Ia_0, area.R_0]
    def __call__(self, u, t): 
        n = len(self.region)
        
        # create a nested list: 
        # SEIR_list[i] = [S_i, E1_i, E2_i, I_i, Ia_i, R_i]:
        SEIR_list = [u[i:i+6] for i in range(0, len(u), 6)]
        # Create separate lists containing E2 and Ia values:
        E2_list = [u[i] for i in range(2, len(u), 6)]
        Ia_list = [u[i] for i in range(4, len(u), 6)]
        derivative = [] 
        for i in range(n):
            N = self.region[i].population 
            S, E1, E2, I, Ia, R = SEIR_list[i]
            #dS = -self.beta*S*I/N
            dS = 0
            for j in range(n):
                Nj = self.region[j].population
                d = self.region[i].distance(self.region[j])
                E2_other = E2_list[j]
                Ia_other = Ia_list[j]
                dS += np.exp(-d)*(self.r_ia*self.beta(t)*S*Ia/Nj - self.r_e2*self.beta(t)*S*E2/Nj)
            dE1 = -(self.beta(t)*S*I/Nj + self.r_ia*self.beta(t)*S*Ia/Nj + self.r_e2*self.beta(t)*S*E2/Nj) - self.lmbda_1*E1
            dE2 = self.lmbda_1*(1-self.p_a)*E1 - self.lmbda_2*E2
            dI = self.lmbda_2*E2 - self.mu*I 
            dIa = self.lmbda_1*self.p_a*E1 - self.mu*Ia 
            dR = self.mu*(I + Ia)
            
            # calculate dE1, dE2, dI, dIa, dR
            # put the values in the end of derivative
            derivative += [dS, dE1, dE2, dI, dIa, dR]
        return derivative 
    
    def solution(self, u, t):
        n = len(t)
        n_reg = len(self.region)
        self.t = t
        self.S = np.zeros(n)
        self.E1 = np.zeros(n)
        self.E2 = np.zeros(n) 
        self.I = np.zeros(n) 
        self.Ia = np.zeros(n) 
        self.R = np.zeros(n)
        SEIR_list = [u[:, i:i+6] for i in range(0, n_reg*6, 6)]
        for part, SEIR in zip(self.region, SEIR_list):
            part.set_SEIR_values(SEIR, t)
            self.S += part.S
            self.E1 = part.E1
            self.E2 = part.E2 
            self.I = part.I 
            self.Ia = part.Ia 
            self.R = part.R
    
    def plot(self):
        plt.plot(self.t,self.S,label='Susceptibles')
        plt.plot(self.t,self.I,label='Infected')
        plt.plot(self.t,self.Ia,label='Infected (asymptomatics)')
        plt.plot(self.t,self.R,label='Recovered') 
        plt.legend()
        plt.xlabel('Time(days)')
        plt.ylabel('Population') 
        plt.title(self.area_name)  
    

if __name__ == '__main__':
   innlandet = RegionInteraction('Innlandet',S_0=371385, E2_0=0, \
                                 lat=60.7945,long=11.0680)
   oslo = RegionInteraction('Oslo',S_0=693494,E2_0=100, \
                             lat=59.9,long=10.8)
   print(oslo.distance(innlandet)) 


problem = ProblemInteraction([oslo,innlandet], 'Norway_east', beta=0.5)
print(problem.get_population())
problem.set_initial_condition()
print(problem.initial_condition) #non-nested list of length 12
u = problem.initial_condition
print(problem(u,0)) #list of length 12. Check that values make sense 



    