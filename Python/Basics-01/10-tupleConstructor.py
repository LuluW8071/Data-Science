# In order to use list comprehesnion on tuples
# First convert tuples to list then convert back to tuple using tuple() constructor
# Example 1: Square of each number in the range [0, 3]
tup = (i*i for i in range(4))
print(tuple(tup))

# Example 2: Square of each even number in the range [0, 9]
tup1 = (i*i for i in range(10) if i % 2 == 0)
print(tuple(tup1))

# Example 3: Filtering strings in a tuple based on certain conditions
names = ('Python', 'Ruby', 'Java', 'JavaScript', 'HTML', 'CSS')

namesWith_y = tuple(item for item in names if 'y' in item)
namesWith_Java = tuple(item for item in names if 'Java' in item)
namesWith_len4 = tuple(item for item in names if len(item) > 4)

print(namesWith_y, namesWith_Java, namesWith_len4, sep='\t')

# Output:
# (0, 1, 4, 9)
# (0, 4, 16, 36, 64)
# ('Python', 'Ruby')	('Java', 'JavaScript')	('Python', 'JavaScript')
