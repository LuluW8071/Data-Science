# tuples are unchangeable after creation
tup = (1, 5, 10, 15, 20, 30, 40, 'red', 'green', 'yellow')
print(type(tup), tup)

print(len(tup))
print(tup[:5])
print(tup[4:9])

if 'red' in tup:
    print('red is present')
    if 15 in tup:
        print('15 present')

# Output:
# <class 'tuple'> (1, 5, 10, 15, 20, 30, 40, 'red', 'green', 'yellow')
# 10
# (20, 30, 40, 'red', 'green')