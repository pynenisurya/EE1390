#program to calculate area of an equalateraltraiangle given centroid and line equation

import numpy as np
import matplotlib.pyplot as plt
c=2                    #given conditions
O = np.array([[0,0]])
M = np.array([[1,1]])
#plot for the question
plt.plot([0], [0], 'ro',label = "(0,0)centroid")
x1=[1,3,-1]
y1 = [1,-1,3]
plt.plot(x1,y1 ,label = "(1 1)x = 2")
plt.plot([])
plt.axis([-2, 6, -2 ,6])
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.legend() 
plt.show()
#ploting for solution


c=2
O = np.array([[0,0]])
M = np.array([[1,1]])


inv = np.array([[0,1],[-1,0]])
M_inv = np.matmul(M,inv) # finding inverse
S = np.concatenate((M, M_inv))
e = M_inv*O.T
f = e[0,0]
D = np.array([c,f])
X = np.matmul(D,np.linalg.inv(S).T)
d = X - O
v = np.linalg.norm(d)
length = 3*v/(np.sin(np.pi/3))
area = 0.5*length*length*np.sin(np.pi/3)
v = np.linalg.norm(M)
M=M/v
B = X+(length*M_inv)/2
C = X-(length*M_inv)/2
A = 3*O - 2*X
Q = np.array([A[0,0],A[0,1]])
A = Q
Q = np.array([B[0,0],B[0,1]])
B = Q
Q = np.array([C[0,0],C[0,1]])
C = Q
print(area)


len =10

lam_1 = np.linspace(0,1,len)

x_AB = np.zeros((2,len))
x_BC = np.zeros((2,len))
x_CA = np.zeros((2,len))
for i in range(len):
    temp1 = A + lam_1[i]*(B-A)
    x_AB[:,i]= temp1.T
    temp2 = B + lam_1[i]*(C-B)
    x_BC[:,i]= temp2.T
    temp3 = C + lam_1[i]*(A-C)
    x_CA[:,i]= temp3.T
#print(x_AB[0,:],x_AB[1,:])
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

plt.plot(A[0], A[1], 'o')
plt.text(A[0] * (1 + 0.1), A[1] * (1 - 0.1) , 'A')
plt.plot(B[0], B[1], 'o')
plt.text(B[0] * (1 - 0.2), B[1] * (1) , 'B')
plt.plot(C[0], C[1], 'o')
plt.text(C[0] * (1 + 0.03), C[1] * (1 - 0.1) ,'C')

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() #minor

#else
plt.show()
