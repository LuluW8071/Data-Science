# This is comment

'''
This is comment as well
and 
we are learning about escape sequences
'''

"""This is triple quote comment"""

# '\n' is a escape sequence character for new line
print('This is a print statement and\nyou are reading this statement')

# '\t' is a escape sequence character for adding tab(usually 4 spaces)
print('This is a\nprint statement and\tyou are reading this statement')

# Default separator character sep=' ' (space)
print('Printing...', 1, 2, sep='_', end='YYY')
print('Print', 5, 6, sep='~', end='XXX\n')

print('Escape Character')

# Output:
# This is a print statement and
# you are reading this statement
# This is a
# print statement and     you are reading this statement
# Printing..._1_2YYYPrint~5~6XXX
# Escape Character
