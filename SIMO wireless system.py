# -*- coding: utf-8 -*-
"""
Multiple antenna Wireless system

@author: ripradha
"""

import numpy as np
import numpy.random as nr
#import numpy.linalg as nl
import matplotlib.pyplot as plt

# combinatorial to produce nCr = n!/r!(n-r)!
from scipy.special import comb

blocklength = 1000;
nBlocks = 10000;
L=2; # Number of Rx ant Or Diversity order

EbdB = np.arange(1.0, 10.1, 1.5); # Energy per Bit in dB
Eb  = 10 ** (EbdB/10);
No = 1;
SNR = 2*Eb/No; #2* for QPSK 2 bits per symbol

SNRdB = 10*np.log10(SNR);
BER = np.zeros(len(EbdB));
BERt = np.zeros(len(EbdB));

for blk in range(nBlocks):
    #Rayleigh fading channel with STD = 1/sqrt(2) (Variance = 1/2) Mean = 0 with avg power unity
    h = nr.normal(0.0, 1.0/np.sqrt(2), (L,1)) + 1j*nr.normal(0.0, 1.0/np.sqrt(2), (L,1));   
    #Complex Gaussian noise with mean=0 and STD = sqrt(No/2) => variance = No/2
    noise = nr.normal(0.0, np.sqrt(No/2), (L,blocklength)) + 1j*nr.normal(0.0, np.sqrt(No/2), (L,blocklength));
    
    BitsI = nr.randint(2, size = blocklength);
    BitsQ = nr.randint(2, size = blocklength);
    sym = (2*BitsI -1) + 1j*(2*BitsQ -1); # Complex QPSK symbol
    
    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K])*sym; #Power scaling of Sym         
        RxSym = h*TxSym + noise; #output Rx
        
        MRCout = np.sum(np.conj(h) * RxSym, axis=0);
        #MRCout = np.sum(nl.pinv(h) * RxSym, axis=0);
        
        DecI = (np.real(MRCout)>0);
        DecQ = (np.imag(MRCout)>0);
        
        BER[K]  =BER[K] + np.sum(DecI != BitsI) + np.sum(DecQ != BitsQ);
        
BER = BER/blocklength/nBlocks/2; #/2 coz of QPSK
BERt = comb(2*L-1, L)/2**L/SNR**L; #BER in Theory

plt.yscale('log')
plt.plot(SNRdB, BER,'g--');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for MRC')
plt.legend(["Simulation", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER')  
        
        
        
        
    
    





























