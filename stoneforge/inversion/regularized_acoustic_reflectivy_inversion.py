#Programa que implementa computacionalmente Inversao de traco sismico por deconvolucao sparse spike usando o algoritmo de variacao total e minimos quadrados

import numpy as np 

def IRLS_Inv(trace, G, alphaL1, alphaL2,thresold,niter):
    residuo_iter = np.zeros(10)
    dim = len(trace)
    R = np.eye(dim)
    I = np.eye(dim)
    G_T = G.transpose()
    count = 0 
    #Tikhonov ordem zero:
    M1_L2 = np.linalg.inv( np.matmul(G_T, G) + (alphaL2*I) )
    M2_L2 = np.matmul(G_T, trace)
    RegL2 = np.matmul(M1_L2, M2_L2)

    #Inversao do traco sismico utilizando regularizacao de tikhonov a partir da norma1 nos residuos dos dados
    while (count < niter):
            Gt_R = np.matmul(G_T, R)
            M1_L1 = np.linalg.inv( np.matmul(Gt_R,G) + (alphaL1*I) )
            M2_L1 = np.matmul(Gt_R, trace)
            RegL1 = np.matmul(M1_L1, M2_L1)
            dcalc = np.matmul(G, RegL1)
            residuo = dcalc[:] - trace[:]
         
            for i in range( (dim//2)+2 ):
                if abs(residuo[i]) >= thresold:
                    R[i,i] = 1/abs(residuo[i])
                else:
                    R[i,i] = 1/thresold
            for i in range(dim):
                if abs(RegL1[i]) >= thresold:
                    I[i,i] = 1/abs(RegL1[i])
                else:
                    I[i,i] = 1/thresold
            residuo_iter[count] = residuo[145]
            count = count + 1
    return RegL2, RegL1, residuo_iter