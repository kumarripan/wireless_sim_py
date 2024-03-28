import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl

SNRdB = np.arange(-10,10,1); # Signal-to-noise power ratio in dB
SNR = 10**(SNRdB/10); # Signal-to-noise power ratio
numBlocks = 100; # Number of blocks
Capacity_OPT = np.zeros(len(SNRdB)); # Optimal capacity of MIMO channel 
Capacity_EQ = np.zeros(len(SNRdB));# Capacity with equal power allocation
r = 4; # Number of receive antennas
t = 4; # Number of transmit antennas


def OPT_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = 0;
    No = 1;
    P = SNR;
    while not CAP:
        onebylam = (P + sum(No/S[0:t]**2))/t;
        if  onebylam - No/S[t-1]**2 >= 0:
            optP = onebylam - No/S[0:t]**2;
            CAP = sum(np.log2(1+ S[0:t]**2 * optP/No));
        elif onebylam - No/S[t-1]**2 < 0:
            t = t-1;            
    return CAP

def EQ_CAP_MIMO(Heff,SNR):
    U, S, V = nl.svd(Heff, full_matrices=False)
    t = len(S);
    CAP = sum(np.log2(1+ S[0:t]**2 * SNR/t));
    return CAP


for L in range(numBlocks): # Looping over blocks
    # rxt MIMO channel
    H = (nr.normal(0.0, 1.0, (r,t)) + 1j*nr.normal(0.0, 1.0, (r,t)))/np.sqrt(2);
    for kx in range(len(SNRdB)):
        Capacity_OPT[kx] = Capacity_OPT[kx] + OPT_CAP_MIMO(H, SNR[kx]);
        Capacity_EQ[kx] = Capacity_EQ[kx] + EQ_CAP_MIMO(H, SNR[kx]);

                                                           
#Averaging capacity values over the blocks
Capacity_OPT = Capacity_OPT/numBlocks;
Capacity_EQ = Capacity_EQ/numBlocks;

# Plotting capacities for equal and optimal power allocation obtained via simulation
plt.plot(SNRdB,Capacity_OPT,'b-s');
plt.plot(SNRdB,Capacity_EQ,'r-.o');
plt.grid(1,which='both')
plt.legend(["OPT","Equal"], loc ="upper left");
plt.suptitle('MIMO Capacity vs SNR(dB)')
plt.xlabel('SNR (dB)')
plt.ylabel('Capacity (b/s/Hz)') 

