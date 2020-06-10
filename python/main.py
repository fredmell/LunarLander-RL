from run import run

def main():
    M = 1500 # Max number of episodes
    T = 400  # Max number of steps per episode
    analyzer = run(M, T)
    analyzer.plot_training("../latex/figures/training.pdf")

if __name__ == '__main__':
    main()
