# Nested-if is using one conditional statement inside another,
# providing a way to make more refined decisions in code

exam_score = int(input('Enter your exam score: '))

if exam_score >= 90:
    print("Excellent! You received an A.")
elif exam_score >= 80:
    print("Good job! You received a B.")
else:
    print("Work harder. You received a grade below B.")
    if exam_score >= 70:
        print("But you passed with a C.")
    elif exam_score >= 60:
        print("You received a D.")
    else:
        print("You need improvement. You received an F.")

# Output Sample:
# Enter your exam score: 100
# Excellent! You received an A.

# Enter your exam score: 70
# Work harder. You received a grade below B.
# But you passed with a C.
