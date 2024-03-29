import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import List
from uuid import UUID, uuid4

import numpy as np


class Tolerance(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


THRESHOLDS = {Tolerance.LOW: 0.4, Tolerance.MEDIUM: 0.8, Tolerance.HIGH: 1.2}


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
        distribution.dist_func(distribution.mean, distribution.std_dev, agentsPerTeam),
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
    agentsPairs = [
        (shuffledAgents[2 * i], shuffledAgents[2 * i + 1])
        for i in range(0, len(agentKeys) // 2)
    ]
    return agentsPairs


def calculateDelta(agentA, agentB) -> float:
    delta = np.sum(np.abs(agentA.opinions - agentB.opinions))
    return delta


def get_reward(agent1: Agent, agent2: Agent):
    distance = calculateDelta(agent1, agent2)
    
    reward = 0
    cost = 0
    agent1Strategy = agent1.currentStrategy
    agent2Strategy = agent2.currentStrategy

    a1_within_threshold_cost_factor = {
        Tolerance.LOW: 0.5,
        Tolerance.MEDIUM: 0.3,
        Tolerance.HIGH: 0.1,
    }
    a1_out_of_threshold_cost_factor = {
        Tolerance.LOW: 3,
        Tolerance.MEDIUM: 2,
        Tolerance.HIGH: 1,
    }
    reward_when_within_threshold = {
        Tolerance.LOW: {Tolerance.LOW: 3, Tolerance.MEDIUM: 3.5, Tolerance.HIGH: 4}, # Low and meets Low, Medium, High
        Tolerance.MEDIUM: {Tolerance.LOW: 1, Tolerance.MEDIUM: 1.75, Tolerance.HIGH: 2.5}, # Medium and meets Low, Medium, High
        Tolerance.HIGH: {Tolerance.LOW: 0.5, Tolerance.MEDIUM: 0.75, Tolerance.HIGH: 1},
    }
    if distance <= THRESHOLDS[agent1Strategy]:
        reward = reward_when_within_threshold[agent1Strategy][agent2Strategy]
        cost = a1_within_threshold_cost_factor[agent1Strategy] * distance
    else: 
        reward = 0
        cost = a1_out_of_threshold_cost_factor[agent1Strategy] * distance
    
    return reward - cost

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