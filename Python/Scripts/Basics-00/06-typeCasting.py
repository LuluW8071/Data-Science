# Typecasting is conversion of one data type into the other data type
# Example:
'''
int(), float(), str(), ord(), hex(), oct(), tuple(), set(), list(), dict(), etc 
'''

# Explict Typecasting is conversion of one data type into other done via developer (manually)
x = 'Python'
y = 2023
z = '1'
print('Explicit Typecasting')
print(x + str(y), '\n', type(x + str(y)))
print(y + int(z), '\n', type(y + int(z)))

# Implicit Typecasting is conversion of one data type into other done via interpreter (automatically)
x = 3.13
y = 12

print('Implicit Typecasting when', x, '+', y,
      'performed the result is', x + y, '\n', type(x + y))
print('Implicit Typecasting when', x, '/', y,
      'performed the result is', x / y, '\n', type(x / y))

# Output:
# Explicit Typecasting
# Python2023
#  <class 'str'>
# 2024
#  <class 'int'>
# Implicit Typecasting when 3.13 + 12 performed the result is 15.129999999999999
#  <class 'float'>
# Implicit Typecasting when 3.13 / 12 performed the result is 0.2608333333333333
#  <class 'float'>
