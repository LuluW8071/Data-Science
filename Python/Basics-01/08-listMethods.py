x = [13, 90, 87, 63, 37, 60]

# makes of copy of x list for y list
y = x.copy()
y[0] = 0
print(x)
print(y)

# insert(index, list element)
x.insert(3, 100)
print(x)

# extends the list
lst = ['red', 'green', 'blue', 'yellow']
x.extend(lst)
print(x)

# concatenates the list
print(x + lst)

# Output:
# [13, 90, 87, 63, 37, 60]
# [0, 90, 87, 63, 37, 60]
# [13, 90, 87, 100, 63, 37, 60]
# [13, 90, 87, 100, 63, 37, 60, 'red', 'green', 'blue', 'yellow']
# [13, 90, 87, 100, 63, 37, 60, 'red', 'green', 'blue', 'yellow', 'red', 'green', 'blue', 'yellow']
