import numpy as np

class Consumer:
    def __init__(self, id, alpha, beta):

        self.id = id
        self.alpha = alpha
        self.beta = beta

    def compute_utility(self, x, p, ksi):
        u = x * self.beta - p * self.alpha + ksi + np.random.gumbel()
        print(u)

    def __str__(self) -> str:
        return f"Consumer {self.id}."