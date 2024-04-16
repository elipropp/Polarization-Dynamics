import random
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4
import time
import numpy as np
import math

from generateOpinions import bimodal_opinions

class Tolerance(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

TOLERANCES = [1, 2, 3]

@dataclass
class Distribution:
    mean: float
    std_dev: float
    lower_bound: float
    upper_bound: float
    dist_func: np.random

class Agent:
    def __init__(
        self, uuid: UUID, team: int, opinions: list[float], strategy_weights: np.array
    ):
        self.uuid: UUID = uuid
        self.team = team
        self.opinions: np.array = opinions
        self.utility = 0
        self.strategy_weights = strategy_weights
        self.old_strategy_weights = []
        self.strategy_loss = {1: 0, 2: 0, 3: 0}
        self.agent_loss = 0

# Agent generation with type hints
def generate_agents(opinions, strategy_weights) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    print(len(opinions))
    for i, opinion in enumerate(opinions):
        team = 0 if opinion < 0 else 1 # team based on opinion
        # team = i % 2 # mixed opinions on each team
        # trunk-ignore(bandit/B311)
        # random_number = random.randint(0, 2)
        agent_id = str(uuid4())
        agents[agent_id] = Agent(
            agent_id, team, np.array([opinion]), strategy_weights[i]
        )

    # num_same_team_connections = int(total_connections_per_agent * percent_same_team)
    # num_opposite_team_connections = total_connections_per_agent - num_same_team_connections
    # print(f'num_same_team_connections: {num_same_team_connections}')
    # print(f'num_opposite_team_connections: {num_opposite_team_connections}')
    # generate_agent_connections(agents, num_same_team_connections, num_opposite_team_connections)

    return agents


def calculate_delta(a1Opinions: np.array, a2Opinions: np.array) -> float:
    delta = np.sum(np.abs(a1Opinions - a2Opinions))
    return delta

def get_strategy(agent: Agent):
    probabilities = agent.strategy_weights / np.sum(agent.strategy_weights)
    # print(f"weights: {agent.strategy_weights}, sum: {np.sum(agent.strategy_weights)}")
    action = np.random.choice(TOLERANCES, p=probabilities)
    return action

def get_utility(strategy, other_strategy, distance):
    reward = math.log(1 + math.exp(max(strategy, other_strategy))) * \
                    (((strategy + other_strategy)/distance) - strategy*distance)
    return reward


def sim(
    agents: dict[UUID, Agent],
    numIterations: int,
    epsilon: float
) -> dict[Agent]:
    start_time = time.time()
    for iteration in range(numIterations):
        # print(f"iteration: {iteration}")
        for uuid, agent in agents.items():
            other_agents = [key for key in agents.keys() if key != uuid]
            other_agent_uuid = random.choice(other_agents)
            other_agent = agents[other_agent_uuid]

            # agent.add_interaction(other_agent.opinions, other_agent.currentStrategy)

            agent.old_strategy_weights.append(agent.strategy_weights.copy())

            # choose strat based on weights
            agent_strategy = get_strategy(agent)
            other_agent_strategy = get_strategy(other_agent)

            distance = calculate_delta(agent.opinions, other_agent.opinions)
            utility = get_utility(agent_strategy, other_agent_strategy, distance)
            agent.utility += utility

            # find max reward from other strats
            utility_map: dict = {agent_strategy: utility}
            for strategy in TOLERANCES:
                if strategy == agent_strategy:
                    continue
                utility = get_utility(strategy, other_agent_strategy, distance)
                utility_map[strategy] = utility

            max_reward = max(utility_map.values())
            min_reward = min(utility_map.values())

            # calculate loss for all strats and update weights
            # scale utility to 0-1 range
            weight_multiplier = 1
            if np.sum(agent.strategy_weights) < 10:
                weight_multiplier = 10

            # print(f"mult: {weight_multiplier}, sum: {np.sum(agent.strategy_weights)}")
            for strategy in TOLERANCES:
                # 1 - so highest reward possible has 0 loss
                loss = 1 - ((utility_map[strategy] - min_reward) / (max_reward - min_reward))
                agent.strategy_loss[strategy] += loss
                if strategy == agent_strategy:
                    agent.agent_loss += loss
                # - 1 bc it's an np.array
                agent.strategy_weights[strategy - 1] *= math.exp(-epsilon*loss)*weight_multiplier
                # print(f"new strat: {agent.strategy_weights[strategy - 1]}")

        if iteration % 100 == 0:
            print(f"iteration: {iteration}")

    return agents


if __name__ == "__main__":
    # OTHER DIST OPTION
    # np.random.uniform
    opinions = bimodal_opinions(
        num_agents=1000,
        mean1=-0.5,
        mean2=0.5,
        std_dev1=0.1,
        std_dev2=0.1,
        lower_bound=-10,
        upper_bound=10,
        proportion_first_mode=0.5,
    )

    agents = generate_agents(opinions)
    agents = sim(agents, 100)
