import matplotlib.pyplot as plt
import seaborn as sns

from run import run

if __name__ == '__main__':
    sns.set()
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(tight_layout=True)
    ax.set_xscale("log")
    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel("Mean reward")

    M = 200  # Number of episodes
    T = 1000 # Max number of time steps per episode
    # Learning rates to try
    learning_rates = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00001]
    for optimizer in ("Adam", "SGD", "RMSProp"):
        avg_rewards = []
        for i, lr in enumerate(learning_rates):
            analyzer = run(M, T, learning_rate=lr, optimizer=optimizer, show=False)
            avg_rewards.append(analyzer.average_reward(lag=100))

        ax.plot(learning_rates, avg_rewards, marker="o", label=optimizer)

    ax.legend()
    fig.savefig("../latex/report/figures/learning_rate.eps")
