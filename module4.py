n = int(input("Enter a positive integer N: "))

numbers = []
for i in range(n):
    num = int(input(f"Enter number {i+1}: "))
    numbers.append(num)

x = int(input("Enter X to search: "))

if x in numbers:
    print(numbers.index(x) + 1)
else:
    print(-1)
