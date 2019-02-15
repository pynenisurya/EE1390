import numpy as np
import math
Q = np.array([[0,0]])
P = np.array([[-1,0]])
R = np.array([[3,3*math.sqrt(3)]])
r = np.linalg.norm(R)
R = R/r




#metod 1
p = np.linalg.norm(P)
P = P/p
QP = P-Q
QR = R-Q
angular_bisector = QP+QR
C = angular_bisector*Q.T
print(angular_bisector)
print(C)


#method 2
QP2 = np.array([[-1,0],[0,-1]])
Angle = R*np.linalg.inv(QP2)
t = math.acos(Angle[0][0])
t = t/2
rotator = np.array([[math.cos(t),math.sin(t)],[math.sin(-t),math.cos(t)]])
slope = P*rotator
print(slope)
constant = Q*slope.T
print(constant)


