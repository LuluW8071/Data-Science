x = abs(int(input("Enter a random number: ")))  # abs() : absolute value

match x:
    case 0:
        print(x, 'is zero')
    case 1:
        print(x, 'is one')
    case 2:
        print(x, 'is two')
    # _ is default case
    case _:
        print(x, '> 2')

# Output Sample:
# Enter a random number: -5
# 5 > 2