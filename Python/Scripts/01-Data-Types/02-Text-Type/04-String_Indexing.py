name = 'Python'
name1 = 'Python is a great language'

print(name[0])
print(name[1])
# print(name[6])  # IndexError: string index out of range
print(name[-1])   # Prints character at last index (name[len(name)-1])
print(name[len(name)-1])

# Looping through the string
print('--------Looping--------')
for characters in name1:
    print(characters)

# Output:
# P
# y
"""IndexError: string index out of range"""
# n
# n
# --------Looping--------
# P
# y
# t
# h
# o
# n
 
# i
# s
 
# a
 
# g
# r
# e
# a
# t
 
# l
# a
# n
# g
# u
# a
# g
# e