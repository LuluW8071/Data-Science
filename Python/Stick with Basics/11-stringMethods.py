# Strings are immutable 
a = '!!!String Methods!!!'
x = 'introduction to python!'

print(a)
print(len(a))

print(a.upper())       # uppercase
print(a.lower())       # lowercase

print(a.strip("!"))
print(a.lstrip("!"))
print(a.rstrip("!"))

print(a.replace('String', 'Python'))

print(a.split(' '))
print(x.capitalize()) 