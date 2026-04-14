import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from abc import ABC, abstractmethod
from loguru import logger

# Configuration
BANDIT_REWARDS = [1, 2, 3, 4]
NUM_TRIALS = 20000
EPSILON = 0.1
PRECISION = 1.0  # known precision for Thompson Sampling

class Bandit(ABC):
    @abstractmethod
    def __init__(self, m):
        """Initialize the bandit with parameter m."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return string representation."""
        pass

    @abstractmethod
    def pull(self):
        """Simulate pulling the bandit arm."""
        pass

    @abstractmethod
    def update(self, x):
        """
        Update internal estimates based on observed reward.

        Parameters:
        x : float
            Observed reward.
        """
        pass

    @abstractmethod
    def experiment(self):
        """Run the bandit experiment."""
        pass

    def report(self, data, selected_bandits, algorithm_name):
        """Generate performance report."""
        
        df = pd.DataFrame({
            'Bandit': selected_bandits,
            'Reward': data,
            'Algorithm': algorithm_name
        })

        os.makedirs('exported_results', exist_ok=True)
        df.to_csv(f'exported_results/{algorithm_name}_results.csv', index=False)
        
        cumulative_reward = np.sum(data)

        # Regret calculation
        optimal_val = np.max(BANDIT_REWARDS)
        true_means_chosen = np.array([BANDIT_REWARDS[i] for i in selected_bandits])
        cumulative_regret = np.sum(optimal_val - true_means_chosen)  
              
        logger.info(f"[{algorithm_name}] Cumulative Reward: {cumulative_reward:.2f}")
        logger.info(f"[{algorithm_name}] Cumulative Regret: {cumulative_regret:.2f}")

        # selection counts
        counts = np.bincount(selected_bandits, minlength=len(BANDIT_REWARDS))
        logger.debug(f"[{algorithm_name}] Arm Selection Counts: ...")
        for i, c in enumerate(counts):
            logger.debug(f"  Arm {i} (mean={BANDIT_REWARDS[i]}): {c} times")

        return cumulative_reward, cumulative_regret

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy bandit algorithm with decaying epsilon (1/t).

    Parameters:
    m : float
        True mean reward of the bandit.
    """
    def __init__(self, m):
        self.m = m
        self.m_estimate = 0.0
        self.N = 0

    def __repr__(self):
        return f"EpsilonGreedyBandit(true_mean={self.m})"

    def pull(self):
        """
        Draw a reward sample from a normal distribution.

        Returns:
        float
            Sampled reward.
        """
        return np.random.randn() + self.m

    def update(self, x):
        """
        Update estimated mean using incremental average.

        Parameters:
        x : float
            Observed reward.
        """
        self.N += 1
        self.m_estimate = self.m_estimate + (x - self.m_estimate) / self.N

    def experiment(self, rewards_list, n_trials=NUM_TRIALS):
        """
        Run epsilon-greedy experiment.

        Parameters:
        rewards_list : list
            True means of bandits.
        n_trials : int
            Number of trials.

        Returns:
        tuple
            (rewards array, selected bandits array)
        """
        bandits = [EpsilonGreedy(m) for m in rewards_list]
        
        data = np.empty(n_trials)
        selected_bandits = np.empty(n_trials, dtype=int) 

        for t in range(1, n_trials + 1):
            eps = 1 / t 
            
            if np.random.random() < eps:
                j = np.random.randint(len(bandits))
            else:
                j = np.argmax([b.m_estimate for b in bandits])
            
            x = bandits[j].pull()
            bandits[j].update(x)
            
            data[t-1] = x
            selected_bandits[t-1] = j
            
        return data, selected_bandits

#--------------------------------------#

class ThompsonSampling(Bandit):
    """
    Thompson Sampling using Gaussian prior and likelihood.

    Parameters:
    true_mean : float
        True mean reward of the bandit.
    """
    def __init__(self, true_mean):
        self.true_mean = true_mean

        # Prior: N(0,1)
        self.m = 0
        self.lambda_ = 1
        self.tau = PRECISION

        self.N = 0
        self.sum_x = 0

    def __repr__(self):
        return f"Thompson Bandit with True Mean: {self.true_mean}"

    def pull(self):
        """
        Sample reward from true distribution.

        Returns:
        float
            Sampled reward.
        """
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def sample(self):
        """
        Sample from posterior distribution.

        Returns:
        float
            Sampled mean estimate.
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m
    
    def update(self, x):
        """
        Update posterior parameters.

        Parameters:
        x : float
            Observed reward.
        """
        self.lambda_ += self.tau
        self.sum_x += x
        self.m = (self.tau * self.sum_x) / self.lambda_
        self.N += 1

    def experiment(self, rewards_list, n_trials=NUM_TRIALS):
        """
        Run Thompson Sampling experiment.

        Returns:
        tuple
            (rewards array, selected bandits array)
        """
        bandits = [ThompsonSampling(m) for m in rewards_list]
        rewards = np.empty(n_trials)
        selected_bandits = np.empty(n_trials, dtype=int) 

        for i in range(n_trials):
            # Select bandit with highest sample from posterior
            j = np.argmax([b.sample() for b in bandits])
            
            x = bandits[j].pull()
            bandits[j].update(x)
            
            rewards[i] = x
            selected_bandits[i] = j
            
        return rewards, selected_bandits

#--------------------------------------#

class Visualization():
    """Handles plotting of experiment results."""

    def plot1(self, eg_data, ts_data):
        """
        Plot average reward convergence.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
        for ax, scale in zip(axes, ['linear', 'log']):
            ax.plot(np.cumsum(eg_data) / (np.arange(NUM_TRIALS) + 1), label="E-Greedy")
            ax.plot(np.cumsum(ts_data) / (np.arange(NUM_TRIALS) + 1), label="Thompson Sampling")
            ax.axhline(np.max(BANDIT_REWARDS), color='r', linestyle='--', label="Optimal")
            ax.set_xscale(scale)
            ax.set_title(f"Avg Reward Convergence ({scale} scale)")
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot2(self, eg_data, ts_data, eg_selected, ts_selected):
        """
        Plot cumulative rewards and regrets.
        """
        eg_cum_reward = np.cumsum(eg_data)
        ts_cum_reward = np.cumsum(ts_data)
        
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        
        ax[0].plot(eg_cum_reward, label="E-Greedy")
        ax[0].plot(ts_cum_reward, label="Thompson")
        ax[0].set_title("Cumulative Rewards")
        ax[0].legend()
        
        optimal_val = np.max(BANDIT_REWARDS)

        eg_true_means = np.array([BANDIT_REWARDS[i] for i in eg_selected])
        ts_true_means = np.array([BANDIT_REWARDS[i] for i in ts_selected])

        eg_regret = np.cumsum(optimal_val - eg_true_means)
        ts_regret = np.cumsum(optimal_val - ts_true_means)
        
        ax[1].plot(eg_regret, label="E-Greedy")
        ax[1].plot(ts_regret, label="Thompson")
        ax[1].set_title("Cumulative Regrets")
        ax[1].legend()
        
        plt.show()

def comparison():
    """
    Run full comparison between Epsilon-Greedy and Thompson Sampling.
    """
    eg_solver = EpsilonGreedy(0)
    ts_solver = ThompsonSampling(0)
    viz = Visualization()
    
    logger.info("Starting Epsilon-Greedy Experiment...")
    eg_data, eg_selected = eg_solver.experiment(BANDIT_REWARDS)
    
    logger.info("Starting Thompson Sampling Experiment...")
    ts_data, ts_selected = ts_solver.experiment(BANDIT_REWARDS)
    
    eg_solver.report(eg_data, eg_selected, "E_Greedy")
    ts_solver.report(ts_data, ts_selected, "Thompson")
    
    viz.plot1(eg_data, ts_data)
    viz.plot2(eg_data, ts_data, eg_selected, ts_selected)


if __name__ == '__main__':
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

    comparison()