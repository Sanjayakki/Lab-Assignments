row1=int(input("enter the number of rows in Matrix 1:"))
col1=int(input("enter the number of columns in Matrix 1:"))

row2=int(input("enter the number of rows in Matrix 2:")) 
col2=int(input("enter the number of columns in Matrix 2:"))

Matrix1=[]
Matrix2=[]

print("Enter the elements of Matrix 1:")

for i in range(row1):
    row1=list(map(int,input().split()))
    Matrix1.append(row1)

print("Enter the elements of Matrix 2:")

for i in range(row2):
    row2=list(map(int,input().split()))
    Matrix2.append(row2)

print(Matrix1)
print(Matrix2)    

