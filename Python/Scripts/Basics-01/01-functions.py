# A block of code that executes a certain tasks when called is a function
'''User Defined Function'''

def calc_geometricMean(a, b):
    gm = (a*b)/(a+b)
    return gm

def check(a, b):
    if a > b:
        print(a, '>', b)
    elif a < b:
        print(a, '<', b)
    else:
        print(a, '=', b)

def func_later(a, b):
    # This function is a placeholder and its behavior will be implemented later
    pass


a = 10
b = 5
result1 = calc_geometricMean(a, b)
print('geometricMean of', a, 'and', b, 'is', result1)
check(a, b)

c = 100
d = 99
result2 = calc_geometricMean(c, d)
print('geometricMean of', c, 'and', d, 'is', result2)
check(c, d)

# Output Sample:
# geometricMean of 10 and 5 is 3.3333333333333335
# 10 > 5
# geometricMean of 100 and 99 is 49.74874371859296
# 100 > 99
