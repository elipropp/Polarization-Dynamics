import copy
import random
from dataclasses import dataclass
from typing import List
from uuid import UUID, uuid4
from enum import Enum
import numpy as np

class Tolerance(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2

THRESHOLDS = {Tolerance.LOW: 0.4,
              Tolerance.MEDIUM: 0.8,
              Tolerance.HIGH: 1.2}

@dataclass
class Distribution:
    mean: float
    std_dev: float
    lower_bound: float
    upper_bound: float
    dist_func: np.random


class Agent:
    def __init__(self, uuid: UUID, opinions: list[float], currentStrat: Tolerance):
        self.uuid: UUID = uuid
        self.opinions: List = []
        self.interactionHistory: List = []
        self.currentStrategy: Tolerance = currentStrat
        self.strategyHistory: List = []
        self.reward = 0


def generateAgents(
    distribution: Distribution,
    agentsPerTeam: int = 100,
) -> dict[Agent]:
    
    agents = {}

    opinions = np.clip(
        distribution.dist_func(
            distribution.mean, distribution.std_dev, agentsPerTeam
        ),
        distribution.lower_bound,
        distribution.upper_bound,
    )

    for opinion in opinions:
        random_number = random.randint(0, 2)
        uuid = uuid4()
        agents[uuid] = Agent(uuid, np.array([opinion]), random_number)

    return agents


def create_agent_pairs(agentKeys: list):
    shuffledAgents = random.shuffle(agentKeys)
    agentsPairs = [(shuffledAgents[2*i], shuffledAgents[2*i + 1]) for i in range(0, len(agentKeys)//2)]
    return agentsPairs

def calculateDelta(agentA, agentB) -> float:
    delta = np.sum(np.abs(agentA.opinions - agentB.opinions))
    return delta


def get_reward(agent1: Agent, agent2: Agent):
    distance = calculateDelta(agent1, agent2)

    agent1Strategy = agent1.currentStrategy
    agent2Strategy = agent2.currentStrategy


    ### A1 IS LOW COST
    if agent1Strategy == Tolerance.LOW and agent2Strategy == Tolerance.LOW:
        if distance <= THRESHOLDS[Tolerance.LOW]:
            cost = 0.3*distance
            reward = 1
            # low cost - easy to talk to the other person
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            cost = 2*distance
            reward = 0
            # cost = distance
            pass

    elif agent1Strategy == Tolerance.LOW and agent2Strategy == Tolerance.MEDIUM:
        if distance <= THRESHOLDS[Tolerance.LOW]:
            cost = 0.3*distance
            reward = 1.5
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            # cost = distance*0.5
            cost = 2*distance
            reward = 0
            pass

    elif agent1Strategy == Tolerance.LOW and agent2Strategy == Tolerance.HIGH:
        if distance <= THRESHOLDS[Tolerance.LOW]:
            cost = 0.3*distance
            reward = 2
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            # cost = distance*0.25
            cost = 2*distance
            reward = 0
            pass

    ### A1 IS MEDIUM COST
    elif agent1Strategy == Tolerance.MEDIUM and agent2Strategy == Tolerance.LOW:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.2*distance
            reward = 0.5
            # low cost 
            # med reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            cost = 1*distance
            # cost = distance
            pass

    elif agent1Strategy == Tolerance.MEDIUM and agent2Strategy == Tolerance.MEDIUM:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.2*distance
            reward = 1
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            # cost = distance*0.5
            cost = 1*distance
            # cost = distance
            pass

    elif agent1Strategy == Tolerance.MEDIUM and agent2Strategy == Tolerance.HIGH:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.2*distance
            reward = 1.5
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            # cost = distance*0.25
            cost = 1*distance
            pass

    ### A1 IS HIGH COST
    elif agent1Strategy == Tolerance.HIGH and agent2Strategy == Tolerance.LOW:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.1*distance
            reward = 1
            # low cost 
            # med reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            cost = 0.5*distance
            # cost = distance
            pass

    elif agent1Strategy == Tolerance.HIGH and agent2Strategy == Tolerance.MEDIUM:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.1*distance
            reward = 1
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            # cost = distance*0.5
            cost = 0.5*distance
            # cost = distance
            pass

    elif agent1Strategy == Tolerance.HIGH and agent2Strategy == Tolerance.HIGH:
        if distance <= THRESHOLDS[Tolerance.MEDIUM]:
            cost = 0.1*distance
            reward = 1
            # low cost -
            # high reward - despite being low tolerance you found someone to talk to
            pass
        
        else:
            cost = 0.5*distance
            pass

    



def sim(
    agents: dict[Agent],
    numIterations: int,
) -> list[Agent]:
    for i in range(numIterations):
        agentKeys = agents.keys()
        agentPairs = create_agent_pairs(agentKeys)

        for agent1UUID, agent2UUID in agentPairs:
            agent1 = agents[agent1UUID]
            agent2 = agents[agent2UUID]

            reward1 = get_reward(agent1, agent2)
            reward2 = get_reward(agent2, agent1)
        


    
