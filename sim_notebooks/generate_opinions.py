import numpy as np

def trimodal_opinions(mean1: float, mean2: float, mean3: float, std_dev: float, num_agents: int, lower_bound: float, upper_bound: float) -> np.ndarray:
    thirds = num_agents // 3
    opinions_first = np.random.normal(mean1, std_dev, thirds)
    opinions_second = np.random.normal(mean2, std_dev, thirds)
    opinions_third = np.random.normal(mean3, std_dev, num_agents - 2*thirds)  # Ensure total count is num_agents
    
    opinions = np.concatenate((opinions_first, opinions_second, opinions_third))
    np.random.shuffle(opinions)
    return np.clip(opinions, lower_bound, upper_bound)


def normal_opinions(
    mean: float, std_dev: float, num_agents: int, lower_bound: float, upper_bound: float
) -> np.ndarray:
    opinions = np.random.normal(mean, std_dev, num_agents)
    return np.clip(opinions, lower_bound, upper_bound)


# Bimodal opinion generation
def bimodal_opinions(
    mean1: float,
    mean2: float,
    std_dev1: float,
    std_dev2: float,
    num_agents: int,
    lower_bound: float,
    upper_bound: float,
    proportion_first_mode: float = 0.5,
) -> np.ndarray:
    num_agents_first = int(num_agents * proportion_first_mode)
    num_agents_second = num_agents - num_agents_first

    opinions_first = np.random.normal(mean1, std_dev1, num_agents_first)
    opinions_second = np.random.normal(mean2, std_dev2, num_agents_second)

    opinions = np.concatenate((opinions_first, opinions_second))
    np.random.shuffle(opinions)  # Ensure mixed distribution

    return np.clip(opinions, lower_bound, upper_bound)

def symmetric_bimodal_opinions(
    mean: float,
    std_dev: float,
    num_agents: int,
    lower_bound: float,
    upper_bound: float
) -> np.ndarray:
    # Calculate the number of agents in each mode (half the total)
    num_agents_half = num_agents // 2

    # Generate opinions for the first half
    opinions_first = np.random.normal(mean, std_dev, num_agents_half)

    # Create a symmetric copy by negating the first half
    opinions_second = -opinions_first

    # If the total number of agents is odd, add one more entry from the first distribution
    if num_agents % 2 != 0:
        extra_opinion = np.random.normal(mean, std_dev, 1)
        opinions = np.concatenate((opinions_first, opinions_second, extra_opinion))
    else:
        opinions = np.concatenate((opinions_first, opinions_second))

    np.random.shuffle(opinions)  # Ensure mixed distribution

    # Clip opinions to stay within the bounds
    return np.clip(opinions, lower_bound, upper_bound)

def uniform_opinions(lower_bound: float, upper_bound: float, num_agents: int) -> np.ndarray:
    return np.random.uniform(lower_bound, upper_bound, num_agents)


def step_opinions(steps: list[tuple[float, int]], num_agents: int, lower_bound: float, upper_bound: float) -> np.ndarray:
    opinions = []
    for step_mean, step_size in steps:
        opinions.extend(np.random.normal(step_mean, 0.05, step_size))
    
    opinions = np.array(opinions[:num_agents])  # Trim or extend to match num_agents
    np.random.shuffle(opinions)
    return np.clip(opinions, lower_bound, upper_bound)


def random_walk_opinions(start: float, num_agents: int, step_size: float = 0.1) -> np.ndarray:
    opinions = [start + step_size * 2 * (np.random.randint(0, 2) - 0.5) for _ in range(num_agents)]
    opinions = np.cumsum(opinions)
    return np.clip(opinions, -1, 1)
