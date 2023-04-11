import pandas as pd
import matplotlib.pyplot as plt

def visualize_marketeq_stability(seed):

    df = pd.read_csv(f'data/sim_{seed}.csv')

    fig, axs = plt.subplots(5, figsize=(6, 6))
    fig.suptitle(f'Market Equilibium Over Different Number of Firms Seed {seed}')
    axs[0].plot(df.avg_price, color='orange')
    axs[0].set_title("Price")
    axs[1].plot(df.avg_profit, color='green')
    axs[1].set_title("Profit")
    axs[2].plot(df.avg_markup, color='blue')
    axs[2].set_title("Markup")
    axs[3].plot(df.outside_share, color='red')
    axs[3].set_title("Outside Good Share")
    axs[4].plot(df.cost, color='yellow')
    axs[4].set_title("Average Cost in Seed")
    fig.tight_layout()

    fig.savefig("plots/all.png")

if __name__ == "__main__":
    seed = 9
    visualize_marketeq_stability(seed)