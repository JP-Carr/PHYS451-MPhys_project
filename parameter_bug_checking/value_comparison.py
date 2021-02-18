x=[i for i in range(100)]
#y=[9,8,4,5,6]

file=open("values.txt","r")
a=file.read()
#print(a.split(",")[10])
y=[int(a.split(",")[i]) for i in range(0,-1+len(a.split(",")))]

print(len(y))


z=list(set(x) - set(y))
print(z)