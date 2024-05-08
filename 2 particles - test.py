#LIBRARIES

import numpy as np
import math as mt
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import itertools
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams["text.usetex"]=True


#VALUE DEFINITION

s=3         #dimension of lattice
n=2         #number of particles
a=5         #initial position
mu=1        #value of mu
tau=1       #value of tau
U=0         #value of U

t=0         #initial time
t_f=20      #final time
nt=100      #number of iterations

d=mt.comb(s+n-1,n) #dimension of the Hilbert space

#OPERATOR MATRICES

#Hilbert space basis

def generate_vector_basis(dimension,particles):
    basis=[]
    for combo in itertools.combinations_with_replacement(range(dimension),particles):
        vector=[0]*dimension
        for index in combo:
            vector[index]+=1
        basis.append(vector)
    return basis

basis=np.array(generate_vector_basis(s,n))
file_path="basis.txt"
np.savetxt(file_path,basis,fmt='%.0f')

#Number operator matrices

Nbig=np.zeros((d,d))
n_op={}
for i in range(s):
    name=f"n_{i+1}"
    matrix=np.zeros((d,d))
    for j in range(d):
        vector=basis[j]
        matrix[j,j]=(vector[i])
    n_op[name]=matrix
    Nbig+=matrix

file_path="n_op.txt"
np.savetxt(file_path,n_op["n_1"],fmt='%.4f')

#Hamiltonian operator matrix

H_int=np.zeros((d,d))                       #interaction terms
for vector in basis:
    index=np.where((basis==vector).all(axis=1))
    for value in vector:
        if value > 1:
            H_int[index[0],index[0]]+=value*(value-1)

H_hop=np.zeros((d,d))                       #hopping terms
shift_v1=np.zeros(s)
shift_v2=np.zeros(s)
for v1 in basis:
    index1=np.where((basis==v1).all(axis=1))
    for v2 in basis:
        if (v1==v2).all():
            continue
        index2=np.where((basis==v2).all(axis=1))
        for i in range(s):
            shift_v1[:]=v1[:]
            shift_v1[i]-=1
            for j in range(s):
                if (j!=i+1) or (j!=i+1):
                    continue
                shift_v2[:]=v2[:]
                shift_v2[j]-=1
                if (shift_v1==shift_v2).all():
                    H_hop[index1,index2]=np.sqrt(np.dot(basis[index1],np.transpose(basis[index2])))
                    H_hop[index2,index1]=np.sqrt(np.dot(basis[index1],np.transpose(basis[index2])))

H=tau*H_hop+U/2*H_int                       #Hamiltonian
file_path="H.txt"
np.savetxt(file_path,H,fmt='%.3f')

#Data storage matrices

psi_t=np.empty((nt+1,d),dtype=complex)
time=np.empty((nt+1,1))
prob=np.empty((nt+1,s))
test=np.empty((nt+1,3),dtype=complex)

#INITIAL STATE

psi_0_old=np.zeros(s)
psi_0_old[1]=2
psi_0_old[0]=0

psi_0=np.zeros((d,1))
index=np.where((basis==psi_0_old).all(axis=1))
psi_0[index[0]]=1

#COMPUTATION

#Time evolution of \Psi

for k in range(nt+1):
    psi_tt=(np.dot(expm(-1j*H*t),psi_0))
    psi_t[k,:d]=np.transpose(psi_tt)
    time[k,0]=t
    t=(k+1)*t_f/nt

#Expected value <n>:

    for j in range(s):
        prob[k,j]=np.real(np.vdot(psi_tt,np.dot(n_op[f"n_{j+1}"],psi_tt)))
    
    test[k,0]=np.vdot(psi_tt,psi_tt)                #norma=1
    test[k,1]=np.vdot(psi_tt,np.dot(H,psi_tt))      #energia
    test[k,2]=np.vdot(psi_tt,np.dot(Nbig,psi_tt))   #N total

#Data storage

file_path="test.txt"
np.savetxt(file_path,test,fmt='%.8f')

data=np.concatenate((prob,time),axis=1)
file_path="data.txt"
np.savetxt(file_path,data,fmt='%.4f')