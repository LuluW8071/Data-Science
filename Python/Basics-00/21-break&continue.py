x = input('Enter a string: ')

for index in range(len(x)):
    character = x[index]
    print('Character:', character, '\tIndex:', index)

    if character in ['a', 'e', 'i', 'o', 'u']:
        print('\tVowel character')
        continue

    if character in ['x', 'y', 'z']:
        print('There was a problem')
        break

print('End of breakAndContinue')

# Output Sample:
# Enter a string: programmingxpro
# Character: p    Index: 0
# Character: r    Index: 1
# Character: o    Index: 2
#         Vowel character
# Character: g    Index: 3
# Character: r    Index: 4
# Character: a    Index: 5
#         Vowel character
# Character: m    Index: 6
# Character: m    Index: 7
# Character: i    Index: 8
#         Vowel character
# Character: n    Index: 9
# Character: g    Index: 10
# Character: x    Index: 11
# There was a problem
# End of breakAndContinue
