# trunk-ignore-all(bandit/B403)
import pickle

def save_agents(agents, filename):
    filename = "../saved_sim_runs/" + filename + ".pkl"
    with open(filename, "wb") as file:
        pickle.dump(agents, file)
        
def load_agents(filename):
    with open(filename, 'rb') as file:
        # trunk-ignore(bandit/B301)
        return pickle.load(file)