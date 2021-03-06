import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def gather_stats(agent, env):
    """ Compute average rewards over 10 episodes
    """
    score = []
    for k in range(10):
        old_state = env.reset()
        cumul_r, done = 0, False
        while not done:
            a = agent.policy_action(old_state, is_test=True, legal_action=env.legal_actions)
            old_state, r, done, _ = env.step(a)
            cumul_r += r
        score.append(cumul_r)
    return np.mean(np.array(score)), np.std(np.array(score))


def plot_reward(file_path):
    sns.set(style="darkgrid")

    # Load an example dataset with long-form data
    data = pd.read_csv(file_path)

    # Plot the responses for different events and regions
    sns.relplot(x="Episode", y="Mean", kind="line",
                data=data)
    plt.savefig(f'{file_path}_Mean.png')
    # plt.show()


if __name__ == '__main__':
    plot_reward('../log/logs_1598334995.2496798.csv')
