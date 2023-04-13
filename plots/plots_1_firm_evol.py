import pandas as pd
import matplotlib.pyplot as plt

def visualize_marketeq_stability(seed):

    df = pd.read_csv(f'data/individual_{seed}.csv')

    fig, axs = plt.subplots(2, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Price Firm 1 {seed}')
    axs[0].plot(df.price, color='orange')
    axs[0].set_title("Price")
    axs[1].plot(df.cost, color='green')
    axs[1].set_title("Cost")
    fig.tight_layout()

    fig.savefig("plots/individual.png")

if __name__ == "__main__":
    seed = 9
    visualize_marketeq_stability(seed)