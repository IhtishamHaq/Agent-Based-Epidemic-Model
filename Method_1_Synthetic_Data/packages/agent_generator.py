import random
from packages.abm.agent import Agent

class AgentGenerator:
    
    def __init__(self, m, num_infected):
       
        self.m = m
        self.num_infected = num_infected

        if self.num_infected > m:
            raise ValueError('Number of infected cannot exceed total number of agents.')

    def is_infected(self, current_infected):
        
        return current_infected < self.num_infected

    def get_status(self, infected):
       
        return 'I' if infected else 'S'

    def generate_agents(self):
        agents = []
        current_infected = 0

        for i in range(self.m):
            infected = self.is_infected(current_infected)
            if infected:
                current_infected += 1

            status = self.get_status(infected)

            
            new_agent = Agent(position=(-1, -1), status=status)
            agents.append(new_agent)

        return agents
