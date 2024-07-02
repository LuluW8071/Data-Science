# Function Arguments and Return
def average(a, b):
    print('Avg. :', (a+b)/2)


def sum(a=0, b=8):
    print('Sum of', a, 'and', b, '=', a+b)


average(64, 32)
sum()
sum(100, 20)
sum(9)
sum(b=99)

# Output:
# Avg. : 48.0
# Sum of 0 and 8 = 8
# Sum of 100 and 20 = 120
# Sum of 9 and 8 = 17
# Sum of 0 and 99 = 99
