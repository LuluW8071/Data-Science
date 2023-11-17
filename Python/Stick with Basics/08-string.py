# Strings are like a sequence of characters called an array
language1 = 'Python'
language2 = "Ruby"
language3 = 'JavaScript'

print('List of Programming Langauges: ' + language1+' ' + language2+' ' + language3)

'''
Note:
In String, when enclosed within same kind of quote you can't use same quotes 
    quote = 'He said, 'I want to learn python''---------[wrong]

But we can use escape sequence character (\) to use same kind of quote
'''

quote1 = 'He said, "I want to learn python"'
quote2 = 'She said, \'I want to learn ruby\''

print(quote1)
print(quote2)

# Multiline String
print('''
Lorem ipsum dolor sit amet, 
consectetur adip proident 
et inter
''')

# Output:
# List of Programming Langauges: Python Ruby JavaScript
# He said, "I want to learn python"
# She said, 'I want to learn ruby'

# Lorem ipsum dolor sit amet,
# consectetur adip proident
# et inter