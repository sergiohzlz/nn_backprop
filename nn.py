from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from six.moves import zip
import numpy as np
import sys

xor = np.array( [[0,0,1,0],[0,1,1,1],[1,0,1,1],[1,1,1,0]] )
X = xor[:,:3]
Y = xor[:,-1].reshape(-1,1)

alfa, iters = (0.5, 50000)
en, hn, sn = ( X.shape[1], 4, Y.shape[1] )
facth, facts = 'sigmoide', 'identidad'

syn_0 = 2*np.random.random( (en,hn) ) - 1
syn_1 = 2*np.random.random( (hn,sn) ) - 1

def fact(f='sigmoide'):
    if f=='sigmoide':
        return [lambda x: 1./(1+np.exp(-x)), lambda y: y*(1-y)]
    elif f=='tanh':
        return [np.tanh, lambda y: 1 - y**2]
    elif f=='ident':
        return [lambda x: x, lambda y: 1]

def entrena(iters,alfa,pesos,iterr=1000):
    syn_0, syn_1 = pesos
    for j in range(iters):
        I = X.copy()
        H = fh(np.dot(I,syn_0))
        S = fs(np.dot(H,syn_1))
        #error de la capa de salida
        S_err = S - Y
        #delta capa salida
        S_d = S_err*dfs(S)
        #error capa escondida
        H_err = S_d.dot(syn_1.T)
        #delta capa escondida
        H_d = H_err*dfh(H)
        #actualizamos pesos
        syn_1 -= alfa * (H.T.dot(S_d))
        syn_0 -= alfa * (I.T.dot(H_d))
        if((iterr>0) and (j%iterr==0)):
            print(np.mean(np.abs(S_err)))
    return [syn_0,syn_1]

def ff(estimulo,pesos):
    syn_0, syn_1 = pesos
    I = estimulo.copy()
    H = fh(np.dot(I,syn_0))
    S = fs(np.dot(H,syn_1))
    return S

if __name__ == '__main__':
    tipocapa1="sigmoide"
    tipocapa2="ident"
    f1, f2 = fact(tipocapa1), fact(tipocapa2)
    fh, dfh = f1
    fs, dfs = f2
    print("alfa: {0}".format(alfa))
    print("iteraciones: {0}".format(iters))
    print("arquitectura: {0} {1} {2}".format(en,hn,sn))
    print("funciones de activacion: {0} {1}".format(tipocapa1,tipocapa2))
    print("pesos primera capa \n{0}".format(syn_0))
    print("pesos segunda capa \n{0}".format(syn_1))
    print("Datos \n{0}".format(X))
    print("Previo \n{0}".format(Y))
    pesos_t = entrena(iters,alfa,[syn_0,syn_1])
    print("Entranada \n{0}".format(ff(X,pesos_t)))
