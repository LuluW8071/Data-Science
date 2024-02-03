# Lists are changeable
list1 = [99, 98, 97, 96, 95, 94, 93, 92, 'Marks', True]
print(list1)
print(type(list1))

print('Printing Index')
for i in range(len(list1)):
    print(f'Index[{i}] : ', list1[i])

print(list1[-3])
# print(list[len(list1) - 3])

if 94 in list1:
    print('94 Present')
    if '93' in list1:
        print('93 Present')
    else:
        print('93 is not string')
else:
    print('Absent')

# [start, end, skip]
print(list1[1:9:2])

# Output:
# [99, 98, 97, 96, 95, 94, 93, 92, 'Marks', True]
# <class 'list'>
# Printing Index
# Index[0] :  99
# Index[1] :  98
# Index[2] :  97
# Index[3] :  96
# Index[4] :  95
# Index[5] :  94
# Index[6] :  93
# Index[7] :  92
# Index[8] :  Marks
# Index[9] :  True
# 92
# 94 Present
# 93 is not string
# [98, 96, 94, 92]
