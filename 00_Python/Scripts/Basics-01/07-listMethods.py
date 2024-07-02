x = [13, 27, 90, 87, 63, 37, 60]

# Find the index of 37
print(x.index(37))

# Append 99 and 13 to the list
x.append(99)
x.append(13)

# Sort the list in ascending order
x.sort()
print(x)

# Sort the list in descending order
x.sort(reverse=True)
print(x)

# Reverse the list
x.reverse()
print(x)

# Count the occurrences of 13
print(x.count(13))

# Output:
# 5
# [13, 13, 27, 37, 60, 63, 87, 90, 99]
# [99, 90, 87, 63, 60, 37, 27, 13, 13]
# [13, 13, 27, 37, 60, 63, 87, 90, 99]
# 2