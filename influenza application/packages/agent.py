# agent.py
import random
import numpy as np

class Agent:
    S, I, R = 'S', 'I', 'R'

    def __init__(self, status='S', beta=0.45, gamma=14, infected_age: int = 0):
        if status not in (self.S, self.I, self.R):
            raise ValueError(f"Invalid status: {status}")
        self.status = status
        self.beta = float(beta)
        self.gamma = int(gamma)

       
        self.days_infected = int(infected_age) if status == self.I else 0

    def _infection_prob(self, infected_neighbors: int) -> float:
        if infected_neighbors <= 0:
            return 0.0
        return 1.0 - np.exp(-self.beta * infected_neighbors / 4.0)

    def step(self, infected_neighbors: int):
        if self.status == self.S:
            p = self._infection_prob(infected_neighbors)
            if random.random() < p:
                self.status = self.I
                self.days_infected = 0  
            return self.status

       
        if self.status == self.I:
            self.days_infected += 1
            if self.days_infected >= self.gamma:
                self.status = self.R
                self.days_infected = 0
            return self.status

        
        return self.status
