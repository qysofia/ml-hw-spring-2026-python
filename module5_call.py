from module5_mod import NumberList

obj = NumberList()

n = int(input("Enter a positive integer N: "))

for i in range(n):
    num = int(input(f"Enter number {i+1}: "))
    obj.insert(num)

x = int(input("Enter X to search: "))
print(obj.search(x))
