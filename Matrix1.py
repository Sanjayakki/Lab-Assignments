row1 = int(input("Enter the number of rows in Matrix 1: "))
col1 = int(input("Enter the number of columns in Matrix 1: "))

row2 = int(input("Enter the number of rows in Matrix 2: "))
col2 = int(input("Enter the number of columns in Matrix 2: "))

Matrix1 = []
Matrix2 = []

print("Enter the elements of Matrix 1:")
for i in range(row1):
    # CHANGED: used 'current_row' instead of 'row1'
    current_row = list(map(int, input().split()))
    Matrix1.append(current_row)

print("Enter the elements of Matrix 2:")
for i in range(row2):
    # CHANGED: used 'current_row' instead of 'row2'
    current_row = list(map(int, input().split()))
    Matrix2.append(current_row)

print(Matrix1)
print(Matrix2)