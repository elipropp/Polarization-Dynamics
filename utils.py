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

    def get_num_agents(self):
        return len(self.agents)

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
        return 1 # UTILITY MATRIX[0][0] or UTILITY MATRIX[1][1]
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
            # print("Opinions are the same! WOAH")
            opinions.append(parent1.opinions[i])
    return np.array(opinions)

def get_agent_by_uuid(uuid, teamA, teamB):
    for agent in teamA.agents:
        if agent.uuid == uuid:
            return agent
    for agent in teamB.agents:
        if agent.uuid == uuid:
            return agent
    raise ValueError(f"Agent with UUID {uuid} not found in either team.")

def sim(
    teamA: Team,
    teamB: Team,
    numIterations: int,
    FRACTION_MOVE: float = 0.25,
    TOLERANCE: float = 0.75,
    within_team_percentage: float = 0.5,
) -> tuple[Team, Team]:
    num_agents_start = len(teamA.agents) + len(teamB.agents)
    history = []
    for iteration in range(0, numIterations):
        total_agents = len(teamA.agents) + len(teamB.agents)
        print(f"Starting iteration {iteration}/{numIterations}, total agents: {total_agents}")
        if len(teamA.agents) == 0 or len(teamB.agents) == 0:
            print("No Agents Left. Exiting.")
            break
        if total_agents > 5*num_agents_start:
            print("Too many agents in the simulation. Exiting.")
            break
        # print(f"Starting with iteration {i}")
        # print(f"Num Agents Team A: {len(teamA.agents)}, Team B: {len(teamB.agents)}")
        team_a_uuids = [agent.uuid for agent in teamA.agents]
        team_b_uuids = [agent.uuid for agent in teamB.agents]
        agentPairs = create_agent_pairs(team_a_uuids, team_b_uuids, within_team_percentage)
        # combinedAgents = teamA.agents + teamB.agents
        # random.shuffle(combinedAgents)
        # agentPairs = []
        # while (
        #     len(combinedAgents) >= 2
        # ):  # if odd number the last agents just dies (maybe some other way to solve this)
        #     agentPairs.append((combinedAgents.pop(), combinedAgents.pop()))
        # teamACopy, teamBCopy, agentPairs = getAgentPairs(teamA, teamB)

        # check if agents can work together pair by pair - either same team auto yes, diff team check threshold, or both cases check threshold
        for pair_num, (agent1_uuid, agent2_uuid) in enumerate(agentPairs):
            if pair_num > 1000:
                break
            agent1 = get_agent_by_uuid(agent1_uuid, teamA, teamB)
            agent2 = get_agent_by_uuid(agent2_uuid, teamA, teamB)

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


def create_agent_pairs(teamA_agents, teamB_agents, within_team_percentage):
    """
    Create agent pairs with a specified percentage of within-team matches.

    Parameters:
    - teamA_agents: List of agents from Team A.
    - teamB_agents: List of agents from Team B.
    - within_team_percentage: Percentage of matches that should be within the same team.

    Returns:
    - List of tuples, where each tuple represents a pair of agents.
    """
    # Calculate the number of agents to be paired within each team
    min_number_of_agents = min(len(teamA_agents), len(teamB_agents))

    #40 pairs AA, 40 pairs BB, 20 pairs AB

    num_within_team_pairs_per_team = int(min_number_of_agents * within_team_percentage) // 2

    # Separate the within-team pairing process
    def create_within_team_pairs(agents, num_pairs_needed):
        random.shuffle(agents)
        return [(agents.pop(), agents.pop()) for i in range(num_pairs_needed)]

    within_team_A = create_within_team_pairs(teamA_agents, num_within_team_pairs_per_team)
    within_team_B = create_within_team_pairs(teamB_agents, num_within_team_pairs_per_team)

    # Combine the remaining agents for between-team pairing
    random.shuffle(teamA_agents)
    random.shuffle(teamB_agents)
    min_remaining_agents = min(len(teamA_agents), len(teamB_agents))

    # Create pairs from the remaining agents for between-team matches
    between_team_pairs = [(teamA_agents.pop(), teamB_agents.pop()) for _ in range(min_remaining_agents)]

    # Combine within-team and between-team pairs
    agent_pairs = within_team_A + within_team_B + between_team_pairs

    return agent_pairs


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

    teamA, teamB = generateAgents(teamADist, teamBDist, agentsPerTeam=10)
    FRACTION_MOVE = 0.25
    TOLERANCE = 0.75
    history = sim(teamA, teamB, 10, FRACTION_MOVE, TOLERANCE)




distance = Agent1.opinions - Agent2.opinions
if Agent1.strat == LOW and Agent2.strat == LOW:
    if withinThreshold:
        No Cost -- Easy to talk 
        minimal reward
    outsideThreshold:
        high cost -- Because your blood pressure goes up from being annoyed
        or low cost becuase you just don't talk to them.
    
if Agent1.strat == LOW and Agent2.strat == MED:
    if within Threshold:
        Low Cost 
    
    
if Agent1.strat == LOW and Agent2.strat == HIGH:
    
if Agent1.strat == MED and Agent2.strat == LOW:

if Agent1.strat == MED and Agent2.strat == MED:
    
if Agent1.strat == MED and Agent2.strat == HIGH:
    
if Agent1.strat == HIGH and Agent2.strat == LOW:
    
if Agent1.strat == HIGH and Agent2.strat == MED:
    
if Agent1.strat == HIGH and Agent2.strat == HIGH:
    

# Mean field equilibrium 
# More than 50 
# Whaat is the action: 
# - Setting the threshold
# Action --- High Tolerance or low tolerance
 # They enter the room and they observe eachothers actions... 
# They could learn these actions. 

# We need a cost. For moving your opinion. 

# What are the learnt strategies and the learnt outcomes. 

# WE want a strategy that outputs action. It could depend only on my beliefs... to a threshold.
# If you are extreme then your tolerance is less

# The straategy could depend on the distribution of beliefs in the system. 
# The strategy could depend on the history of the system. 

# WE either set the dymanics... Set the strategy... then see the behaviour... --- This is fine but not very interesting. 
# (this doesn't make sense unless we know that this is what people will do)
# In the absesne of this it is an open question.. what should people do.. This is where we should study the equilibirum strategy. 


# Can we introduce a coordinator? 

# 1/ distance  and 1 - 1/distance


## Another dimension of the game: 
# I am the goverment-- I am rewarding them... What should I set the reward to incentivize them to cooperate. 



# No regret learning... 
# You take a set of actions. and then after a period of time you look back to see if I regret not setting my threshold to 0. You want your threshold to a value where you will not regret it later.

# with no regret learning you will learn a CE


# Best response dymanics. Ficitious play.  Looking at the time average of the action. 50% they do high threshold, 50% low.. 
# 
# Assume that is there. Observe past actions and asssume that is the strategy they are playing
# Deep Q learning... take the reward then update the weights. 


# right now we have a symmetric and constant strategy. of tolerance. 
# We have to do a mechanism design ... fix them to a strategy.
# If we are going to do something with a stragey. See if we can do The math... 

# When you do mechanism design.. You look at the optimal strategy... 
# All these agents are self interesed. Acting in their own way. 