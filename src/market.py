import numpy as np
import pandas as pd

from consumer import Consumer
from firm import Firm


class Market:
    # definition of the method also with stype checking 
    def __init__(self, n_firms:int, n_consumers:int):

        self.n_firms = n_firms
        self.n_consumers = n_consumers

        self.firms = []
        self.consumers = []

        # Initialize firms
        for id in range(self.n_firms):
            self.firms.append(Firm(id))

        # Initialize consumers
        for id in range(self.n_consumers):
            # here we call the consumer class because we imported it before
            alpha = 1.
            beta = 0.5
            self.consumers.append(Consumer(id, alpha, beta))

    def say_yeet(self):
        print('yeet')

    def __str__(self) -> str:
        return f"Market with {self.n_firms} firms and {self.n_consumers} consumers."



n_firms = 5
n_consumers = 5
market = Market(n_firms, n_consumers)
market.say_yeet()