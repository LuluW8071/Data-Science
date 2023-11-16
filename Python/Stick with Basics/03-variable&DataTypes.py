# Variables are simply containers that holds data
x = 123  # int
y = 3.1415  # float
z = 'Hello, world!'  # string
comp = 1 + 2j  # complex
comp1 = complex(3, 4)  # complex

print(x)
print(y)
print('Value of x:', x, 'and Value of y:', y)
print(comp+comp1)
print('Boolean value for x > y:', x > y)  # Bool

# Data Types specify types of value a varaible holds
print("Type of x:", type(x))
print("Type of y:", type(y))
print("Type of z:", type(z))
print("Type of z:", type(x+y))
print("Type of comp:", type(comp))

# Output:
# 123
# 3.1415
# Value of x: 123 and Value of y: 3.1415
# (4+6j)
# Boolean value for x > y: True
# Type of x: <class 'int'>
# Type of y: <class 'float'>
# Type of z: <class 'str'>
# Type of z: <class 'float'>
# Type of comp: <class 'complex'>
