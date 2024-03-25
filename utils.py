import copy
import random
from dataclasses import dataclass
from typing import List
from uuid import UUID, uuid4

import numpy as np

# [mean, std_dev, low bound, upper bound]
teamADistribution = [-0.5, 0.25, -1, 0]
teamBDistribution = [0.5, 0.25, 0, 1]


@dataclass
class Distribution:
    mean: float
    std_dev: float
    lower_bound: float
    upper_bound: float
    dist_func: np.random


class Agent:
    def __init__(self, uuid: UUID, teamId: str, opinions: list[float]):
        self.uuid: UUID = uuid
        self.team: str = teamId
        self.opinions: np.array = opinions
        self.probReproduce = 0
        self.childOpinions = np.array([])

    def getTeam(self):
        return self.team

    def __repr__(self) -> str:
        return f"AgentID: {self.uuid}, Opinions: {self.opinions}"


class Team:
    def __init__(self, teamID: str, agents=None):
        self.teamID = teamID
        if agents is None:
            self.agents: list[Agent] = []
        else:
            self.agents = agents

    def addAgent(self, agent: Agent):
        self.agents.append(agent)

    def updateAllAgents(self, agents: list[Agent]):
        self.agents = agents

    def get_agent_opinions(self):
        return [agent.opinions[0] for agent in self.agents]

    def get_mean_opinions(self):
        return np.mean(self.get_agent_opinions())

    def get_std_dev_opinions(self):
        return np.std(self.get_agent_opinions())

    def __repr__(self) -> str:
        # Maybe Print the Distribution, average, std_dev, etc
        return f"TeamID: {self.teamID}, Agents: {self.agents}"


def generateAgents(
    teamADistribution: Distribution,
    teamBDistribution: Distribution,
    agentsPerTeam: int = 100,
) -> tuple[list[Agent], list[Agent]]:
    teamA = Team("A")
    teamB = Team("B")

    teamAOpinions = np.clip(
        teamADistribution.dist_func(
            teamADistribution.mean, teamADistribution.std_dev, agentsPerTeam
        ),
        teamADistribution.lower_bound,
        teamADistribution.upper_bound,
    )
    teamBOpinions = np.clip(
        teamBDistribution.dist_func(
            teamBDistribution.mean, teamBDistribution.std_dev, agentsPerTeam
        ),
        teamBDistribution.lower_bound,
        teamBDistribution.upper_bound,
    )

    for teamAAgentOpinions, teamBAgentOpinions in zip(teamAOpinions, teamBOpinions):
        teamA.addAgent(Agent(uuid4(), teamA.teamID, np.array([teamAAgentOpinions])))
        teamB.addAgent(Agent(uuid4(), teamB.teamID, np.array([teamBAgentOpinions])))

    return teamA, teamB


# def getAgentPairs(teamA: Team, teamB: Team) -> list[tuple[Agent, Agent]]:
#     teamACopy = teamA.copy()
#     teamBCopy = teamB.copy()
#     combinedAgents = teamACopy.agents + teamBCopy.agents

#     random.shuffle(combinedAgents)
#     agentPairs = []
#     while len(combinedAgents) >= 2: # if odd number the last agents just dies (maybe some other way to solve this)
#         agentPairs.append((combinedAgents.pop(), combinedAgents.pop()))

#     return teamACopy, teamBCopy, agentPairs


def calculateDelta(agentA, agentB) -> float:
    delta = np.sum(np.abs(agentA.opinions - agentB.opinions))
    return delta


# UTILITY_MATRIX = [[1.0, 1.5],
#                  [1.5, 1.0]]
def getUtility(agent1: Agent, agent2: Agent):
    if agent1.team == agent2.team:
        return 1.0  # UTILITY MATRIX[0][0] or UTILITY MATRIX[1][1]
    else:
        return 1.5  # UTILITY MATRIX[0][1] or UTILITY MATRIX[1][0]


"""Create a new agent with the opinions of the parent 1 but with a fraction of the opinions of parent 2"""


def gen_child_opinions(
    parent1: Agent, parent2: Agent, FRACTION_MOVE: float = 0.25
) -> List[float]:
    opinions = []
    for i in range(0, len(parent1.opinions)):
        opinion_delta = abs(parent1.opinions[i] - parent2.opinions[i])
        if parent1.opinions[i] < parent2.opinions[i]:
            opinions.append(parent1.opinions[i] + opinion_delta * FRACTION_MOVE)
        elif parent1.opinions[i] > parent2.opinions[i]:
            opinions.append(parent1.opinions[i] - opinion_delta * FRACTION_MOVE)
        else:
            print("Opinions are the same! WOAH")
            opinions.append(parent1.opinions[i])
    return np.array(opinions)


def sim(
    teamA: Team,
    teamB: Team,
    numIterations: int,
    FRACTION_MOVE: float = 0.25,
    TOLERANCE: float = 0.75,
) -> tuple[Team, Team]:
    history = []
    for i in range(0, numIterations):
        # print(f"Starting with iteration {i}")
        # print(f"Num Agents Team A: {len(teamA.agents)}, Team B: {len(teamB.agents)}")

        # print(teamA)
        combinedAgents = teamA.agents + teamB.agents
        random.shuffle(combinedAgents)
        agentPairs = []
        while (
            len(combinedAgents) >= 2
        ):  # if odd number the last agents just dies (maybe some other way to solve this)
            agentPairs.append((combinedAgents.pop(), combinedAgents.pop()))
        # teamACopy, teamBCopy, agentPairs = getAgentPairs(teamA, teamB)

        # check if agents can work together pair by pair - either same team auto yes, diff team check threshold, or both cases check threshold
        for agent1, agent2 in agentPairs:
            delta = calculateDelta(agent1, agent2)
            # print(f"same team: {pair[0].team == pair[1].team}, delta: {delta}")
            if delta <= TOLERANCE:
                # can work together
                utility = getUtility(agent1, agent2)
                agent1.probReproduce = utility
                agent2.probReproduce = utility
                # child opinions will move closer the midpoint between agents by the delta/4
                if agent1.team == agent2.team:  # Same Team
                    agent1.childOpinions = gen_child_opinions(
                        agent1, agent2, FRACTION_MOVE
                    )
                    agent2.childOpinions = gen_child_opinions(
                        agent2, agent1, FRACTION_MOVE
                    )
                else:
                    agent1.childOpinions = gen_child_opinions(
                        agent1, agent2, FRACTION_MOVE
                    )
                    agent2.childOpinions = gen_child_opinions(
                        agent2, agent1, FRACTION_MOVE
                    )
            else:
                # can't work together
                agent1.probReproduce = 0
                agent2.probReproduce = 0
                agent1.childOpinions = np.array([])
                agent2.childOpinions = np.array([])

            # print(f"utility: {agent.probReproduce}")
        # Save this Iteration

        history.append((copy.deepcopy(teamA), copy.deepcopy(teamB)))
        teamA = updateTeam(teamA)
        teamB = updateTeam(teamB)

        # if they can't then prob reproduce is 0
        # if they can then get prob reporduce from utility
        # calculate opinions of offspring
    return history
    # return (teamA, teamB)


def updateTeam(team: Team):
    # print(team.agents)
    teamID = team.teamID
    newTeam = Team(teamID)

    for agent in team.agents:
        numChildren = int(agent.probReproduce)
        if random.random() < (agent.probReproduce - numChildren):
            numChildren += 1

        for _ in range(0, numChildren):
            # make new agents
            # Maybe add randomization to the childOpinions
            if len(agent.childOpinions) > 0:
                # print("REPRODUCING")
                child = Agent(uuid4(), teamID, agent.childOpinions.copy())
                newTeam.addAgent(child)
                # new_agents.append(Agent(uuid4(), teamID, agent.childOpinions))
    return newTeam


if __name__ == "__main__":
    # OTHER DIST OPTION
    # np.random.uniform
    teamADist = Distribution(
        mean=-0.5,
        std_dev=0.25,
        lower_bound=-1,
        upper_bound=0,
        dist_func=np.random.normal,
    )

    teamBDist = Distribution(
        mean=0.5, std_dev=0.25, lower_bound=0, upper_bound=1, dist_func=np.random.normal
    )

    teamA, teamB = generateAgents(teamADist, teamBDist, agentsPerTeam=3)
    FRACTION_MOVE = 0.25
    TOLERANCE = 0.75
    history = sim(teamA, teamB, 2, FRACTION_MOVE, TOLERANCE)
