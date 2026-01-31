import numpy as np
import time as tm
import matplotlib.pyplot as plt
plt.rcParams['figure.facecolor'] = "#ffffff"
plt.rcParams['axes.facecolor'] = '#f0f0f0'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#d0d0d0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.7
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 20
c=0.5
nu=1

r=np.linspace(1,20,100)
phi=0
theta=np.linspace(0,np.pi,40)

RR,THETATHETA=np.meshgrid(r,theta)
PHIPHI=phi
XX,YY,ZZ=RR*np.sin(THETATHETA)*np.cos(PHIPHI),RR*np.sin(THETATHETA)*np.sin(PHIPHI),RR*np.cos(THETATHETA)

U=(nu*(4*np.cos(THETATHETA)*(1+c-np.cos(THETATHETA))-2*(np.sin(THETATHETA))**2)/(RR*(1+c-np.cos(THETATHETA))**2))
V=(-nu*(2*np.sin(THETATHETA))/(RR*(1+c-np.cos(THETATHETA))))


VX=U*np.sin(THETATHETA)*np.cos(PHIPHI)+V*np.cos(THETATHETA)*np.cos(PHIPHI)
VY=U*np.sin(THETATHETA)*np.sin(PHIPHI)+V*np.cos(THETATHETA)*np.sin(PHIPHI)
VZ=U*np.cos(THETATHETA)-V*np.sin(THETATHETA)


plt.figure(figsize=(10,10))
plt.title('Jet puntiforme - c=0.5')
plt.quiver(XX, ZZ, VX, VZ, color='blue', scale=20)
plt.xlabel('X')
plt.ylabel('Z')
plt.xlim([-0.1,2])
plt.ylim([-10,10])
plt.grid(True)

plt.savefig("./relation/immagini_presentazione/anal_get2D.pdf")
plt.show()