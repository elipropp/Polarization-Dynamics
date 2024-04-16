import random
from dataclasses import dataclass
from enum import Enum
from uuid import UUID, uuid4
import time
import numpy as np

from sim_notebooks.generateOpinions import bimodal_opinions

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
    def __init__(
        self, uuid: UUID, team: int, opinions: list[float], currentStrat: Tolerance
    ):
        self.uuid: UUID = uuid
        self.team = team
        self.opinions: np.array = opinions
        self.interactionHistory: list[Interaction] = []
        self.currentStrategy: Tolerance = currentStrat
        self.strategyHistory: list = []
        self.reward = 0
        self.rewardSinceLastUpdate = 0
        self.connected_agents = []
        self.stratToReward = {
            Tolerance.LOW: 0,
            Tolerance.MEDIUM: 0,
            Tolerance.HIGH: 0,
        }

    def add_interaction(self, other_opinions: list[float], other_strategy: Tolerance):
        self.interactionHistory.append(Interaction(other_opinions, other_strategy))

    def clear_interaction_history(self):
        self.interactionHistory = self.interactionHistory[-50:]

    def trim_interaction_history(self, num_to_keep: int):
        self.interactionHistory = self.interactionHistory[-num_to_keep:]

    def get_strategy_history(self):
        return self.strategyHistory

    def get_random_connections(self, num_connections: int):
        if self.connected_agents == []:
            raise ValueError("No connected agents")
        if len(self.connected_agents) < num_connections:
            raise ValueError("Not enough connected agents")
        return random.sample(self.connected_agents, num_connections)

    def is_strategy_stable(self):
        if len(self.strategyHistory) < 10:
            return False
        return len(set(self.strategyHistory[-10:])) == 1

    def reset_strat_to_reward(self):
        self.stratToReward = {
            Tolerance.LOW: 0,
            Tolerance.MEDIUM: 0,
            Tolerance.HIGH: 0,
        }


# Agent generation with type hints
def generate_agents(opinions, total_connections_per_agent: int, percent_same_team: float) -> dict[str, Agent]:
    agents: dict[str, Agent] = {}
    print(len(opinions))
    for i, opinion in enumerate(opinions):
        team = 0 if opinion < 0 else 1 # team based on opinion
        # team = i % 2 # mixed opinions on each team
        # trunk-ignore(bandit/B311)
        random_number = random.randint(0, 2)
        agent_id = str(uuid4())
        agents[agent_id] = Agent(
            agent_id, team, np.array([opinion]), Tolerance(random_number)
        )

    num_same_team_connections = int(total_connections_per_agent * percent_same_team)
    num_opposite_team_connections = total_connections_per_agent - num_same_team_connections
    print(f'num_same_team_connections: {num_same_team_connections}')
    print(f'num_opposite_team_connections: {num_opposite_team_connections}')
    generate_agent_connections(agents, num_same_team_connections, num_opposite_team_connections)

    return agents


# Modifiest the agents in place
def generate_agent_connections(
    agents: dict[str, Agent], same_team_connections=5, opposite_team_connections=2
):
    team0_agents = [agent_id for agent_id, agent in agents.items() if agent.team == 0]
    team1_agents = [agent_id for agent_id, agent in agents.items() if agent.team == 1]

    for agent_id, agent in agents.items():
        # Exclude the current agent from potential connections
        potential_team_mates = [
            id
            for id in (team0_agents if agent.team == 0 else team1_agents)
            if id != agent_id
        ]
        potential_opposite_team = [
            id for id in (team1_agents if agent.team == 0 else team0_agents)
        ]

        # Ensure there are enough agents to select from
        if len(potential_team_mates) >= 10 and len(potential_opposite_team) >= 5:
            # Randomly select 10 members from the same team and 5 from the opposite team
            same_team_selection = random.sample(
                potential_team_mates, same_team_connections
            )
            opposite_team_selection = random.sample(
                potential_opposite_team, opposite_team_connections
            )

            agent.connected_agents = same_team_selection + opposite_team_selection
        else:
            # trunk-ignore(bandit/B608)
            print(f"Not enough agents to select from for agent {agent_id}")


def get_team_opinions(agents: dict[str, Agent], team: int):
    team_agents = [agent for agent in agents.values() if agent.team == team]
    team_opinions = [agent.opinions for agent in team_agents]
    return team_opinions


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
    agents: dict[UUID, Agent],
    numIterations: int,
    numIterationsToUpdate: int = 10,
    lookBackDistance: int = 10,
) -> dict[Agent]:
    start_time = time.time()
    for iteration in range(numIterations):
        for agent in agents.values():
            other_agent_uuid = agent.get_random_connections(1)[0]
            other_agent = agents[other_agent_uuid]

            agent.add_interaction(other_agent.opinions, other_agent.currentStrategy)

            reward = get_reward(
                agent.currentStrategy,
                agent.opinions,
                other_agent.currentStrategy,
                other_agent.opinions,
            )
            agent.reward += reward
            agent.rewardSinceLastUpdate += reward

            if (iteration > 0) and (iteration % numIterationsToUpdate == 0):
                update_strategy(agent, lookBackDistance)
        if iteration % 500 == 0:
            num_strats_stable = sum([agent.is_strategy_stable() for agent in agents.values()])
            print(f"iteration: {iteration}")
            print(f"percentage of agents with stable strategies: {num_strats_stable / len(agents) * 100}%")
            print(f"iteration time: {time.time() - start_time} seconds")
            start_time = time.time()
    print(f"iteration: {iteration}")
    print(f"percentage of agents with stable strategies: {num_strats_stable / len(agents) * 100}%")
    return agents


def update_strategy(agent: Agent, lookBackDistance: int):
    agent.strategyHistory.append(agent.currentStrategy)
    agent.reset_strat_to_reward()
    for strat in Tolerance:
        if strat == agent.currentStrategy:
            agent.stratToReward[strat] += agent.rewardSinceLastUpdate
        else:
            # Calculate reward if agent were to switch to this strategy for the past 10 rounds
            for interaction in agent.interactionHistory[-lookBackDistance:]:  # Assuming you meant the last 10 interactions
                reward = get_reward(
                    strat,
                    agent.opinions,
                    interaction.other_strategy,
                    interaction.other_opinions,
                )
                agent.stratToReward[strat] += reward

    bestStrat = max( agent.stratToReward, key= agent.stratToReward.get)
    # print(f"bestStrat is equal to current strat: {bestStrat == agent.currentStrategy}")
    agent.currentStrategy = bestStrat
    agent.rewardSinceLastUpdate = 0
    # agent.trim_interaction_history(numIterationsToUpdate)


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
