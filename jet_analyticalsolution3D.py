import numpy as np
import time as tm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

c=100
nu=1

r=np.linspace(1,20,100)
phi=np.linspace(0,2*np.pi,5)
theta=np.linspace(0,np.pi,8)

RR,PHIPHI,THETATHETA=np.meshgrid(r,phi,theta)
XX,YY,ZZ=RR*np.sin(THETATHETA)*np.cos(PHIPHI),RR*np.sin(THETATHETA)*np.sin(PHIPHI),RR*np.cos(THETATHETA)

#U=(nu*(4*np.cos(THETATHETA)*(1+c-np.cos(THETATHETA))-2*(np.sin(THETATHETA))**2)/(RR*(1+c-np.cos(THETATHETA))**2))
#V=(-nu*(2*np.sin(THETATHETA))/(RR*(1+c-np.cos(THETATHETA))))
U=2*np.cos(THETATHETA)/RR
V=-1*np.sin(THETATHETA)/RR

VX=U*np.sin(THETATHETA)*np.cos(PHIPHI)+V*np.cos(THETATHETA)*np.cos(PHIPHI)
VY=U*np.sin(THETATHETA)*np.sin(PHIPHI)+V*np.cos(THETATHETA)*np.sin(PHIPHI)
VZ=U*np.cos(THETATHETA)-V*np.sin(THETATHETA)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-5, 5)

ax.set_title(f"Jet assiale", fontsize=12)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.quiver(XX,YY,ZZ,VX,VY,VZ)

plt.show()