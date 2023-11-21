# For loops repeatedly execute a block of code for a specified number of times or over a sequence
name = 'Photosynthesis.'
colors = ['red', 'green', 'blue']

for i in name:
    print(i, end='_')
    if (i == '.'):
        print('\nReached end character\'.\'', name.index('.'))

for color in colors:
    print(color)
    for i in colors[0]:
        print(i)

for x in range(3):
    print(x ,'+ 1 =', x + 1)
    for x in range(1,3):
        print(x, 'x 2 =', x * 2)

# Output:
# P_h_o_t_o_s_y_n_t_h_e_s_i_s_._
# Reached end character'.' 14
# red
# r
# e
# d
# green
# r
# e
# d
# blue
# r
# e
# d
# 0 + 1 = 1
# 1 x 2 = 2
# 2 x 2 = 4
# 1 + 1 = 2
# 1 x 2 = 2
# 2 x 2 = 4
# 2 + 1 = 3
# 1 x 2 = 2
# 2 x 2 = 4