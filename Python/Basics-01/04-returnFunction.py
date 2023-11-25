# Return statement always returns the value of expression back to the calling function
def average(*numbers):
    print(type(numbers))
    sum = 0
    for i in numbers:
        sum = sum + i
    return sum / len(numbers)


a = average(10, 20, 30, 40)
print(a)


def average1(*numbers):
    print(type(numbers))
    sum = 0
    for i in numbers:
        sum = sum + i
    avg1 = sum / len(numbers)
    return 1000
    return avg1


# Always returns statement that comes first
a1 = average1(10, 20, 30, 40)
print(a1)

# Output:
# <class 'tuple'>
# 25.0
# <class 'tuple'>
# 1000
