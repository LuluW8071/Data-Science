x = "Alice"
y = 25
pi = 3.141592653589793

print("My name is %s and I am %d years old." % (x, y))  # % Operator
print("My name is {} and I am {} years old.".format(x, y))  # .format()
print("My name is {0} and I am {1} years old.".format(x, y))  # .format()
print("My name is {1} and I am {0} years old.".format(y, x))  # .format()
print(f"My name is {x} and I am {y} years old.")  # F-String

# Precision ----> {:.} syntax
print(f'{pi:.2f}')
print('{:.5f}'.format(pi))

# Output:
# My name is Alice and I am 25 years old.
# My name is Alice and I am 25 years old.
# My name is Alice and I am 25 years old.
# My name is Alice and I am 25 years old.
# My name is Alice and I am 25 years old.
# 3.14
# 3.14159