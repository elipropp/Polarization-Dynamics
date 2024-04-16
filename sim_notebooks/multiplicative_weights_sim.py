import math
import multiprocessing
import random
import time
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4

import numpy as np


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
    for i, opinion in enumerate(opinions):
        team = 0 if opinion < 0 else 1  # team based on opinion
        agent_id = str(uuid4())
        agents[agent_id] = Agent(
            agent_id, team, np.array([opinion]), strategy_weights[i]
        )
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
    reward = math.log(1 + math.exp(max(strategy, other_strategy))) * (
        ((strategy + other_strategy) / distance) - strategy * distance
    )
    return reward


def sim(agents: dict[UUID, Agent], numIterations: int, epsilon: float) -> dict[Agent]:
    sim_start = time.time()
    iter_start_time = time.time()
    for iteration in range(numIterations):

        for uuid, agent in agents.items():
            other_agents = [key for key in agents.keys() if key != uuid]
            # trunk-ignore(bandit/B311)
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
                loss = 1 - (
                    (utility_map[strategy] - min_reward) / (max_reward - min_reward)
                )
                agent.strategy_loss[strategy] += loss
                if strategy == agent_strategy:
                    agent.agent_loss += loss
                # - 1 bc it's an np.array
                agent.strategy_weights[strategy - 1] *= (
                    math.exp(-epsilon * loss) * weight_multiplier
                )

        if iteration % 100 == 0 and iteration > 0:
            print(
                f"iterations: {iteration - 100} - {iteration} took {time.time() - iter_start_time} seconds"
            )
            iter_start_time = time.time()

    print(f"total sim time: {time.time() - sim_start}")

    return agents


def run_sim(args):
    percent_low, agents = args
    print(f"Simulating low preference percent {percent_low}")
    num_iterations = 1000
    learning_rate = 0.1
    return percent_low, sim(agents, num_iterations, learning_rate)

def start_multiprocess_sim(all_agents: list[tuple[str, list[Agent]]]) -> list[tuple[str, list[Agent]]]:
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    all_agents_post_sim: list[tuple[str, list[Agent]]] = pool.map(run_sim, all_agents)

    pool.close()
    pool.join()
    return all_agents_post_sim
