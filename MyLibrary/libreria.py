#libreria.py 

import math as m
import numpy as np

def Posneg (dati,q=0): #sottrae le righe a due a due 
  s = np.shape(dati)
  z = s[0:q]+((int(s[q]/2)) ,)+s[q+1:]
  m = s[0:q]+s[q+1:]
  Result = np.zeros(m,dtype=float) 
  Result = np.expand_dims(Result, axis=q) 
  for _i in range (0,int(s[q]/2)):
        M =  np.take(dati, 2*_i, axis=q) -  np.take(dati, 2*_i+1, axis=q)
        M = np.expand_dims(M, axis=q)
        Result = np.concatenate((Result,M),axis = q)
  Result = np.delete(Result,0, axis=q)
  return (Result)


def Binning (dati, n, q = 0, normalize = 0 ):
  s = np.shape(dati)
  sqn = int(s[q]/n)
  z = s[0:q] + ((int(s[q]/n)) ,) + s[q+1:]
  m = s[0:q] + s[q+1:]
  Result = np.zeros(m,dtype = float)
  Result = np.expand_dims(Result, axis = q)
  for _i in range(0,int(s[q]/n)):
    M =  np.zeros(m,dtype = float)
    for _j in range (0,n):
      M = M + np.take(dati, n*_i+_j, axis = q)
    M = np.expand_dims(M,axis = q)
    Result = np.concatenate((Result,M),axis = q)
  Result = np.delete(Result,0, axis=q)
  if normalize == 0:
    return(Result/n)
  return(Result)


def Not (array):
    x2 = np.array(array,dtype=int)
    x2[x2==0]=2
    x2[x2==1]=3
    x2[x2==2]=1
    x2[x2==3]=0
    return x2


Matrix = np.array([[0, 0],[0, 1]])
def Hadamard (Iter, Seed = Matrix): #dim_base = 2**(Iter+1)
    if (Iter == 1):
        Seed[Seed==0]=-1
        return Seed
    SeedI = Not(Seed)
    newMatrix1 = np.append(Seed,Seed,axis=1)
    newMatrix2 = np.append(Seed,SeedI,axis=1)
    newMatrix = np.append (newMatrix1,newMatrix2,axis=0)
    return Hadamard(Iter-1,newMatrix)






def Index_ (n):
    if (n == 1):
        return (np.array([[1]],dtype = int))
    I = np.ones(2**(n-1),dtype = int)
    l = np.zeros(2**(n-2),dtype = int)
    l = np.append (l,Not(l))
    I = np.append(I,l)
    if (n==2):
        I = np.reshape(I,(n,-1))
        return I

    for _i in range (1,n-1):
        end  = np.zeros(2**(n-_i-2),dtype = int)
        line = np.append (end,Not(end))
        line  = np.append(line,Not(line))
        for _j in range (0,_i-1):
            line  = np.append(line,line)
        I = np.append(I,line)
    I = np.reshape(I,(n,-1))
    return I



def Sequency (N): #dim_base = 2**(N)
    if (N==1):
        Base = Index_ (2)
    M = np.array([],dtype=int)
    zeri = np.zeros(2**(N-1),dtype=int)
    line = np.array([],dtype = int)
    M = np.zeros(2**N,dtype = int)
    M = np.append(M,zeri)
    M = np.append(M,Not(zeri))
    for _x in range (2,N+1):
        Ind = Index_(_x).T
        for  _y in range (0,2**(_x-1)):
            v = Ind [2**(_x-1)-1-_y]
            line = np.zeros(2**(N-_x),dtype = int)
            for _z in range (0,_x):
                if (v[_z]):
                    line = np.append(line,Not(line))                    
                else:
                    line = np.append(line,line)
            M = np.append(M,line)        
    Base= np.reshape (M,[-1,2**N])
    Base[Base==1] = -1
    Base[Base==0] = 1
    return Base

def MSE (original, compressed):
  mse = np.mean((original-compressed)**2)
  return mse
def PSNR(original, compressed):
  original = np.array(original)
  compressed = np.array (compressed)
  min_I_original = np.min(original)
  max_I_original = np.max(original)
  mse = MSE (original,compressed)
  if(mse == 0): # se l'MSE è zero vuol dire che non cè rumore sul segnale .
                  # quindi è inutile calcolare la PSNR.
    return 100
  PSNR_val = 10*m.log10((max_I_original - min_I_original)**2 / mse)
  return PSNR_val

