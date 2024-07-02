w = 'lorem ipsum dolor sit amet, consectetur adip'
x = 'introductionToPython01'
y = 'Always Ask Aggressively; Accepting Any Audible Arcane And Amorous Answer'
z = '         '

# Caplitalize first letter of each word in string
print(w.title())

# Count the occurrences of the letter 'a' in the string
print(y.count('A'))

# Find the index of the substring
print(y.find('Ask'))
print(x.index('To'))
print(y.find('hello'))  # Returns -1 if the find_string is not present

# Check if all characters in the string 'y' are alphanumeric (a-z,A-Z,0-9)
# If special symbols are present returns false; else true
print(x.isalnum())
print(y.isalnum())

# Returns true only when string contains whitespaces else false
print(z.isspace())

# Returns true only if first letter of each word in string is capitalized else false
print(y.istitle())

# Output:
# Lorem Ipsum Dolor Sit Amet, Consectetur Adip
# 10
# 7
# 12
# -1
# True
# False
# True
# True
