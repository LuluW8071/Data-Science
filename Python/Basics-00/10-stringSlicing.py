name = 'VS_code_is_a_source-code_editor'
name1 = 'PythonLangauge'

print(name[0:5])
print(name1[2:14])
print(name1[6:])
print(name1[:6])
print(name1[:-6])       # Python Interpreter automatically recognizes this as print(name1[0:len(name1)-6]) meaning print(name1[0:8])
# print(name1[0:8])
print(name1[-6:-3])     # Python Interpreter automatically recognizes this as print(name1[len(name1)-6:len(name1)-3]) meaning print(name1[8:11])
# print(name1[8:11])

'''len(variable) stores the length of variable'''
len1 = len(name)
print('There are', len1, 'characters')

# Output:
# VS_co
# thonLangauge
# Langauge
# Python
# PythonLa
# nga
# There are 31 characters
