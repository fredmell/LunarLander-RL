from run import run

def main():
    M = 1500 # Max number of episodes
    T = 400  # Max number of steps per episode
    η = 0.0005 # Learning rate
    optimizer = "RMSProp"
    analyzer = run(M, T, learning_rate=η, optimizer=optimizer, show=False)
    analyzer.plot_training("../latex/figures/training.pdf")

if __name__ == '__main__':
    main()
