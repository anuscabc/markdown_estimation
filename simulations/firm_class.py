class Firm: 
    def __init__(self, price, cost, product_characteristic, share):
        self.price = price
        self.cost = cost
        self.product_characteristic = product_characteristic
        self.share = share
        pass
    
    def profit(price, cost, share): 
        pi = 