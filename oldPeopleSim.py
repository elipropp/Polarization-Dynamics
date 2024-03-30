import random
from dataclasses import dataclass
from enum import Enum
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


@dataclass
class Interaction:
    other_opinions: list[float]
    other_strategy: Tolerance


class Agent:
    def __init__(self, uuid: UUID, opinions: list[float], currentStrat: Tolerance):
        self.uuid: UUID = uuid
        self.opinions: np.array = opinions
        self.interactionHistory: list[Interaction] = []
        self.currentStrategy: Tolerance = currentStrat
        self.strategyHistory: list = []
        self.reward = 0
        self.rewardSinceLastUpdate = 0

    def add_interaction(self, other_opinions: list[float], other_strategy: Tolerance):
        self.interactionHistory.append(Interaction(other_opinions, other_strategy))

    def clear_interaction_history(self):
        self.interactionHistory.clear()

    def get_strategy_history(self):
        return self.strategyHistory


# Agent generation with type hints
def generateAgents(opinions) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    print(len(opinions)
    for opinion in opinions:
        # trunk-ignore(bandit/B311)
        random_number = random.randint(0, 2)
        agent_id = str(uuid4())
        agents[agent_id] = Agent(
            agent_id, np.array([opinion]), Tolerance(random_number)
        )

    return agents

def create_agent_pairs(agentKeys: list):
    random.shuffle(agentKeys)
    agentsPairs = [
        (agentKeys[2 * i], agentKeys[2 * i + 1]) for i in range(0, len(agentKeys) // 2)
    ]
    return agentsPairs


def calculate_delta(a1Opinions: np.array, a2Opinions: np.array) -> float:
    delta = np.sum(np.abs(a1Opinions - a2Opinions))
    return delta


def get_reward(
    a1Strat: Tolerance, a1Opinions: np.array, a2Strat: Tolerance, a2Opinions: np.array
):
    distance = calculate_delta(a1Opinions, a2Opinions)

    reward = 0
    cost = 0

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
        Tolerance.LOW: {
            Tolerance.LOW: 3,
            Tolerance.MEDIUM: 3.5,
            Tolerance.HIGH: 4,
        },  # Low and meets Low, Medium, High
        Tolerance.MEDIUM: {
            Tolerance.LOW: 1,
            Tolerance.MEDIUM: 1.75,
            Tolerance.HIGH: 2.5,
        },  # Medium and meets Low, Medium, High
        Tolerance.HIGH: {
            Tolerance.LOW: 0.5,
            Tolerance.MEDIUM: 0.75,
            Tolerance.HIGH: 1,
        },  # High and meets Low, Medium, High
    }
    if distance <= THRESHOLDS[a1Strat]:
        reward = reward_when_within_threshold[a1Strat][a2Strat]
        cost = a1_within_threshold_cost_factor[a1Strat] * distance
    else:
        reward = 0
        cost = a1_out_of_threshold_cost_factor[a1Strat] * distance

    return reward - cost


def sim(
    agents: dict[Agent],
    numIterations: int,
) -> dict[Agent]:
    for iteration in range(numIterations):
        agentKeys = list(agents.keys())
        agentPairs = create_agent_pairs(agentKeys)

        for agent1UUID, agent2UUID in agentPairs:
            agent1: Agent = agents[agent1UUID]
            agent2: Agent = agents[agent2UUID]

            reward1 = get_reward(
                agent1.currentStrategy,
                agent1.opinions,
                agent2.currentStrategy,
                agent2.opinions,
            )
            reward2 = get_reward(
                agent2.currentStrategy,
                agent2.opinions,
                agent1.currentStrategy,
                agent1.opinions,
            )

            agent1.reward += reward1
            agent2.reward += reward2

            agent1.rewardSinceLastUpdate += reward1
            agent2.rewardSinceLastUpdate += reward1

            if (iteration > 0) and (iteration % 10 == 0):
                update_strategy(agent1)
                update_strategy(agent2)

    return agents


def update_strategy(agent: Agent):
    agent.strategyHistory.append(agent.currentStrategy)
    stratToReward = {
        Tolerance.LOW: 0,
        Tolerance.MEDIUM: 0,
        Tolerance.HIGH: 0,
    }
    for strat in Tolerance:
        if strat == agent.currentStrategy:
            stratToReward[strat] = agent.rewardSinceLastUpdate
        else:
            # Calculate reward if agent were to switch to this strategy for the past 10 rounds
            for interaction in agent.interactionHistory[
                -20:
            ]:  # Assuming you meant the last 10 interactions
                reward = get_reward(
                    strat,
                    agent.opinions,
                    interaction.other_strategy,
                    interaction.other_opinions,
                )
                stratToReward[strat] += reward

    bestStrat = max(stratToReward, key=stratToReward.get)
    # print(f"bestStrat is equal to current strat: {bestStrat == agent.currentStrategy}")
    agent.currentStrategy = bestStrat
    agent.rewardSinceLastUpdate = 0


if __name__ == "__main__":
    # OTHER DIST OPTION
    # np.random.uniform
    distribution = Distribution(
        mean=0, std_dev=0.25, lower_bound=-1, upper_bound=1, dist_func=np.random.normal
    )
    agents = generateAgents(distribution, 200)
    agents = sim(agents, 100)
