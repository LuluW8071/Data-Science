''' Default Arguments: '''
# The function assumes a default value,
# even if a value is not provided in the function call for that argument

def name(fname, mname="Jhon", lname="Whatson"):
    print("Hello,", fname, mname, lname)

name("Amy")

# Output:
# Hello, Amy Jhon Whatson
# ----------------------------------------------------------------

''' Keyword Arguments: '''
# Using arguments with key = value, the interpreter recognizes the arguments by the parameter name 
# Order in which the arguments are passed does not matter

def name(fname, mname, lname):
    print("Hello,", fname, mname, lname)

name(mname = "Peter", lname = "Wesker", fname = "Jade")

# Output:
# Hello, Jade Peter Wesker
# ----------------------------------------------------------------

''' Required Arguments: '''
# pass the arguments in the correct positional order and 
# the number of arguments passed should match with actual function definition

# name("Peter", "Quill")

# Output:
# TypeError: name() missing 1 required positional argument: 'lname'

name("Peter", "Ego", "Quill")

# Output:
# Hello, Peter Ego Quill
# ----------------------------------------------------------------

''' Variable-length Arguments: '''
# *parameter defines function to access the args by processing then in form of tuple

def average(*numbers):
    print(type(numbers))
    sum = 0
    for i in numbers:
        sum = sum + i
    avg = sum / len(numbers)
    print(f'Average is {avg}')

average(1234, 12, 999)

# Output:
# <class 'tuple'>
# Average is 748.3333333333334
# ---------------------------------------------------------------

''' Keyword Arbitary Arguments: '''
# **parameter defines function to access the args by processing then in form of dictionary

def name(**name):
    print(type(name))
    print('Name:', name['fname'],
          name['mname'],
          name['lname'])

name(fname='John', lname='Smith', mname='The third')

# Output:
# <class 'dict'>
# Name: John The third Smith
