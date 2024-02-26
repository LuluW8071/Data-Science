# Create a decsion maker to greet the user using time zone information
'''Using only things that we learned from 01-16*.py'''

# Hints:
# https://docs.python.org/3/library/time.html#time.strftime

# Morning time zone starts at 04:00:00 to 11:59:59
# Afternoon time zone starts at 12:00:00 to 15:59:59
# Evening time zone starts at 16:00:00 to 19:59:59
# Night time zone starts at 20:00:00 to 24:59:59

from time import strftime

# Get the current time in the "Hour:Minute:Second" format
current_time = strftime("%H:%M:%S")
print(current_time)

name = input('Enter your name: ').title()

current_hour = int(current_time[:2])

# Decision Maker
if 4 <= current_hour < 12:
    greeting = "Good morning! Sir,"
elif 12 <= current_hour < 16:
    greeting = "Good afternoon! Sir,"
elif 16 <= current_hour < 20:
    greeting = "Good evening! Sir,"
else:
    greeting = "Good night! Sir,"

print(greeting, name, end='\n')
print('It is', current_time)

# Output Sample:
# Enter your name: Tom Cruise
# Good evening! Sir, Tom Cruise
# It is 16:56:22
