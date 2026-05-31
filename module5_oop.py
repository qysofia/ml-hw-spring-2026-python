class NumberList:
    def __init__(self):
        self.numbers = []

    def insert(self, num):
        self.numbers.append(num)

    def search(self, x):
        if x in self.numbers:
            return self.numbers.index(x) + 1
        return -1

n = int(input("Enter a positive integer N: "))

obj = NumberList()

for i in range(n):
    num = int(input(f"Enter number {i+1}: "))
    obj.insert(num)

x = int(input("Enter X to search: "))
print(obj.search(x))
