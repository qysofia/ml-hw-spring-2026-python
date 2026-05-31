class NumberList:
    def __init__(self):
        self.numbers = []

    def insert(self, num):
        self.numbers.append(num)

    def search(self, x):
        if x in self.numbers:
            return self.numbers.index(x) + 1
        return -1
