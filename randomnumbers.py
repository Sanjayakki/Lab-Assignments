import random
n=[]
for var in range(100):
    n.append(random.randint(100,151))
mean=sum(n)/len(n)
print("Mean of random numbers:",mean)
median=sorted(n)
if len(n)%2==0:
    median=(median[len(n)//2]+median[len(n)//2-1])/2
else:
    median=median[len(n)//2]
print("Median of random numbers:",median)
mode=max(set(n),key=n.count)
print("Mode of random numbers:",mode)

