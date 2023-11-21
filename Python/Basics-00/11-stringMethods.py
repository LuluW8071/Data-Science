# Strings are immutable
a = 'learn!!!String Methods!!!'

print(a)
print(len(a))   # Print length of a string varaible 'a'

print(a.upper())       # uppercase
print(a.lower())       # lowercase

print(a.capitalize())  # Capitalize the first letter of the string

print(a.swapcase())    # Swaps uppercase to lowercase and vice versa

print(a.strip("!"))    # Remove exclamation marks from both ends of string
print(a.lstrip("!"))   # Remove exclamation marks from left side of string
print(a.rstrip("!"))   # Remove exclamation marks from right side of string

print(a.replace('String', 'Python'))  # Replace 'String' with Python

# Split the string into a list of words using space as the delimiter
print(a.split(' '))

# Check if the string ends with '!!!'
print(a.endswith('!!!')) 

# Output:
# learn!!!String Methods!!!
# 25
# LEARN!!!STRING METHODS!!!
# learn!!!string methods!!!
# Learn!!!string methods!!!
# LEARN!!!sTRING mETHODS!!!
# learn!!!String Methods
# learn!!!String Methods!!!
# learn!!!String Methods
# learn!!!Python Methods!!!
# ['learn!!!String', 'Methods!!!']
# True
