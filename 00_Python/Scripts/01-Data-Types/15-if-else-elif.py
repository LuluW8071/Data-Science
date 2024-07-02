a = int(input('Enter a positive integer: '))

if (a < 0):
    print(a, 'is a -ve integer')
elif (a > 0 and a < 10):
    print(a, 'is a 1 digit +ve integer')
elif (a >= 10 and a < 100):
    print(a, 'is a 2 digit +ve integer')
elif (a >= 100 and a < 1000):
    print(a, 'is a 3 digit +ve integer')
else:
    print(a, 'is', len(str(a)), 'digit +ve integer')    # converted 'a'(int_class) to (string_class) then count

# Output Sample:
# Enter a positive integer: 10000
# 10000 is 5 digit +ve integer