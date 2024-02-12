

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.linalg as nl
from scipy.special import comb


# Hermitian of matrix
def CT(G):
    return np.conj(np.transpose(G));

# Calculates the A^H * A
def AHA(A):
    return np.matmul(CT(A), A);

blocklength = 1000;
nBlocks = 10000;
r = 2;
t = 2;

EbdB = np.arange(-10.0, 33.1, 4.0);
Eb = 10 ** (EbdB/10);
No = 1;
Es = 2*Eb;
SNR = Es/No;
SNRdB =  10 * np.log10(SNR);

# Bit error rate
BER_ZF = np.zeros(len(EbdB)); 
BER_LMMSE = np.zeros(len(EbdB));
BERt  = np.zeros(len(EbdB));  # Theory

for blk in range(nBlocks): #BLOCK fading Channel => 1 channel per BLOCK
    # MIMO channel
    H = nr.normal(0.0, 1.0/np.sqrt(2), (r,t)) + 1j*nr.normal(0.0, 1.0/np.sqrt(2), (r,t));
    
    noise = nr.normal(0.0, np.sqrt(No/2), (r, blocklength)) + \
                      1j* nr.normal(0.0, np.sqrt(No/2), (r, blocklength)); # IID Noise
                     
    BitsI = nr.randint(2, size=(t, blocklength));
    BitsQ = nr.randint(2, size=(t, blocklength));
    Sym = (2*BitsI-1) + 1j*(2*BitsQ-1); #Complex QPSK Symbol
    
    for K in range(len(SNRdB)): #for each SNR value generate the TxSymb
        TxSym = np.sqrt(Eb[K]) *Sym; #constellation is +-sqrt(Eb[K]) & +-j*sqrt(Eb[K])
        RxSym = np.matmul(H, TxSym) +noise;
        
        #ZF Receiver
        ZFRx = nl.pinv(H);
        ZFout = np.matmul(ZFRx, RxSym);
        
        DecBitsI_ZF = (np.real(ZFout)>0);
        DecBitsQ_ZF = (np.imag(ZFout)>0);
        
        #BER in ZF receiver
        BER_ZF[K] = BER_ZF[K] + np.sum(DecBitsI_ZF != BitsI) + np.sum(DecBitsQ_ZF != BitsQ);
        
        ##LMMSE Receiver
        LMMSERx = np.matmul(nl.inv(AHA(H) + 1/SNR[K]*np.identity(t)), CT(H));
        LMMSEout = np.matmul(LMMSERx, RxSym);
        
        DecBitsI_LMMSE = (np.real(LMMSEout)>0);
        DecBitsQ_LMMSE = (np.imag(LMMSEout)>0);
        
        #BER in LMMSE receiver
        BER_LMMSE[K] = BER_LMMSE[K] + np.sum(DecBitsI_LMMSE != BitsI) + np.sum(DecBitsQ_LMMSE != BitsQ);
        
#BER_ZF = BER_ZF/(blocklength*nBlocks*2*t); 
#BER_LMMSE = BER_LMMSE/(blocklength*nBlocks*2*t);
BER_ZF = BER_ZF/blocklength/nBlocks/2/t; 
BER_LMMSE = BER_LMMSE/blocklength/nBlocks/2/t;
L = r-t+1;
#BERt = comb(2*L-1, L)* (1/2**L) *(1/SNR**L); #Theorical BER expression
BERt = comb(2*L-1, L)/2**L/SNR**L;
              
plt.yscale('log');
plt.plot(SNRdB, BER_ZF, 'g-');
plt.plot(SNRdB, BER_LMMSE, 'b-.s');
plt.plot(SNRdB, BERt, 'ro');

plt.grid(1, which='both');
plt.suptitle("BER of MIMO Channel QPSK");
plt.legend(["ZF", "LMMSE","ZF_Theory"], loc = "lower left");
plt.xlabel('SNR (dB)');
plt.ylabel('BER')
    
    
    
                      
    
                      
        
    

































