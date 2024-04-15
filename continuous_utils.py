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
EPSILON = 0.1 # learning rate


# THRESHOLDS = {Tolerance.LOW: 0.4, Tolerance.MEDIUM: 0.8, Tolerance.HIGH: 1.2}


@dataclass
class Distribution:
    mean: float
    std_dev: float
    lower_bound: float
    upper_bound: float
    dist_func: np.random

def convert_weights_to_probabilities(low_w, med_w, high_w):
    total_weight = sum([low_w, med_w, high_w])
    return (low_w / total_weight, med_w / total_weight, high_w / total_weight)

class Agent:
    def __init__(
        self, uuid: UUID, team: int, opinions: list[float]
    ):
        self.uuid: UUID = uuid
        self.team = team
        self.opinions: np.array = opinions
        # self.interactionHistory: list[Interaction] = []
        # self.currentStrategy: Tolerance = currentStrat
        # self.strategyHistory: list = []
        self.utility = 0
        # self.rewardSinceLastUpdate = 0
        # self.connected_agents = []
        self.strategy_weights = np.ones(3)
        self.old_strategy_weights = [np.ones(3)]
        self.strategy_loss = {1: 0, 2: 0, 3: 0}
        self.agent_loss = 0
        self.agent_strat_prob_history = []
        # self.stratToReward = {
        #     Tolerance.LOW: 0,
        #     Tolerance.MEDIUM: 0,
        #     Tolerance.HIGH: 0,
        # }
    def update_agent_strat_prob_history(self) -> list[tuple[float, float, float]]:
        self.agent_strat_prob_history = [convert_weights_to_probabilities(*weights) for weights in self.old_strategy_weights]


    # def add_interaction(self, other_opinions: list[float], other_strategy: Tolerance):
    #     self.interactionHistory.append(Interaction(other_opinions, other_strategy))

    # def clear_interaction_history(self):
    #     self.interactionHistory = self.interactionHistory[-50:]

    # def trim_interaction_history(self, num_to_keep: int):
    #     self.interactionHistory = self.interactionHistory[-num_to_keep:]

    # def get_strategy_history(self):
    #     return self.strategyHistory

    # def get_random_connections(self, num_connections: int):
    #     if self.connected_agents == []:
    #         raise ValueError("No connected agents")
    #     if len(self.connected_agents) < num_connections:
    #         raise ValueError("Not enough connected agents")
    #     return random.sample(self.connected_agents, num_connections)

    # def is_strategy_stable(self):
    #     if len(self.strategyHistory) < 10:
    #         return False
    #     return len(set(self.strategyHistory[-10:])) == 1

    # def reset_strat_to_reward(self):
    #     self.stratToReward = {
    #         Tolerance.LOW: 0,
    #         Tolerance.MEDIUM: 0,
    #         Tolerance.HIGH: 0,
    #     }


# Agent generation with type hints
def generate_agents(opinions) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    print(len(opinions))
    for i, opinion in enumerate(opinions):
        team = 0 if opinion < 0 else 1 # team based on opinion
        # team = i % 2 # mixed opinions on each team
        # trunk-ignore(bandit/B311)
        # random_number = random.randint(0, 2)
        agent_id = str(uuid4())
        agents[agent_id] = Agent(
            agent_id, team, np.array([opinion])
        )

    # num_same_team_connections = int(total_connections_per_agent * percent_same_team)
    # num_opposite_team_connections = total_connections_per_agent - num_same_team_connections
    # print(f'num_same_team_connections: {num_same_team_connections}')
    # print(f'num_opposite_team_connections: {num_opposite_team_connections}')
    # generate_agent_connections(agents, num_same_team_connections, num_opposite_team_connections)

    return agents


# Modifiest the agents in place
# def generate_agent_connections(
#     agents: dict[str, Agent], same_team_connections=5, opposite_team_connections=2
# ):
#     team0_agents = [agent_id for agent_id, agent in agents.items() if agent.team == 0]
#     team1_agents = [agent_id for agent_id, agent in agents.items() if agent.team == 1]

#     for agent_id, agent in agents.items():
#         # Exclude the current agent from potential connections
#         potential_team_mates = [
#             id
#             for id in (team0_agents if agent.team == 0 else team1_agents)
#             if id != agent_id
#         ]
#         potential_opposite_team = [
#             id for id in (team1_agents if agent.team == 0 else team0_agents)
#         ]

#         # Ensure there are enough agents to select from
#         if len(potential_team_mates) >= 10 and len(potential_opposite_team) >= 5:
#             # Randomly select 10 members from the same team and 5 from the opposite team
#             same_team_selection = random.sample(
#                 potential_team_mates, same_team_connections
#             )
#             opposite_team_selection = random.sample(
#                 potential_opposite_team, opposite_team_connections
#             )

#             agent.connected_agents = same_team_selection + opposite_team_selection
#         else:
#             # trunk-ignore(bandit/B608)
#             print(f"Not enough agents to select from for agent {agent_id}")


# def get_team_opinions(agents: dict[str, Agent], team: int):
#     team_agents = [agent for agent in agents.values() if agent.team == team]
#     team_opinions = [agent.opinions for agent in team_agents]
#     return team_opinions


# def create_agent_pairs(agentKeys: list):
#     random.shuffle(agentKeys)
#     agentsPairs = [
#         (agentKeys[2 * i], agentKeys[2 * i + 1]) for i in range(0, len(agentKeys) // 2)
#     ]
#     return agentsPairs


def calculate_delta(a1Opinions: np.array, a2Opinions: np.array) -> float:
    delta = np.sum(np.abs(a1Opinions - a2Opinions))
    return delta

def get_strategy(agent: Agent):
    probabilities = agent.strategy_weights / np.sum(agent.strategy_weights)
    action = np.random.choice(TOLERANCES, p=probabilities)
    return action

def get_utility(strategy, other_strategy, distance):
    reward = math.log(1 + math.exp(max(strategy, other_strategy))) * \
                    (((strategy + other_strategy)/distance) - strategy*distance)
    return reward


def sim(
    agents: dict[UUID, Agent],
    numIterations: int,
) -> dict[Agent]:
    start_time = time.time()
    for iteration in range(numIterations):
        for uuid, agent in agents.items():
            other_agents = [key for key in agents.keys() if key != uuid]
            other_agent_uuid = random.choice(other_agents)
            other_agent = agents[other_agent_uuid]

            # agent.add_interaction(other_agent.opinions, other_agent.currentStrategy)
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
            for strategy in TOLERANCES:
                # 1 - so highest reward possible has 0 loss
                loss = 1 - ((utility_map[strategy] - min_reward) / (max_reward - min_reward))
                agent.strategy_loss[strategy] += loss
                if strategy == agent_strategy:
                    agent.agent_loss += loss
                # - 1 bc it's an np.array
                agent.strategy_weights[strategy - 1] *= math.exp(-EPSILON*loss)

            agent.old_strategy_weights.append(agent.strategy_weights.copy())

        if iteration % 100 == 0:
            print(f"iteration: {iteration}. 100 iterations completed in: {time.time() - start_time} seconds")

    #     if iteration % 500 == 0:
    #         num_strats_stable = sum([agent.is_strategy_stable() for agent in agents.values()])
    #         print(f"iteration: {iteration}")
    #         print(f"percentage of agents with stable strategies: {num_strats_stable / len(agents) * 100}%")
    #         print(f"iteration time: {time.time() - start_time} seconds")
    #         start_time = time.time()
    # print(f"iteration: {iteration}")
    # print(f"percentage of agents with stable strategies: {num_strats_stable / len(agents) * 100}%")
    return agents


# def update_strategy(agent: Agent, lookBackDistance: int):
#     agent.strategyHistory.append(agent.currentStrategy)
#     agent.reset_strat_to_reward()
#     for strat in Tolerance:
#         if strat == agent.currentStrategy:
#             agent.stratToReward[strat] += agent.rewardSinceLastUpdate
#         else:
#             # Calculate reward if agent were to switch to this strategy for the past 10 rounds
#             for interaction in agent.interactionHistory[-lookBackDistance:]:  # Assuming you meant the last 10 interactions
#                 reward = get_reward(
#                     strat,
#                     agent.opinions,
#                     interaction.other_strategy,
#                     interaction.other_opinions,
#                 )
#                 agent.stratToReward[strat] += reward

#     bestStrat = max( agent.stratToReward, key= agent.stratToReward.get)
#     # print(f"bestStrat is equal to current strat: {bestStrat == agent.currentStrategy}")
#     agent.currentStrategy = bestStrat
#     agent.rewardSinceLastUpdate = 0
#     # agent.trim_interaction_history(numIterationsToUpdate)


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
