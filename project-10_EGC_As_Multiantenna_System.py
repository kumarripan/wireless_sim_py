import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
from scipy.special import comb

blockLength = 1000;
nBlocks = 10000;
L = 2;

EbdB = np.arange(1.0,23.1,2.0);
Eb = 10**(EbdB/10);
No = 1; 
SNR = 2*Eb/No;
SNRdB = 10*np.log10(Eb/No);
BEREGC = np.zeros(len(EbdB));
BERAS = np.zeros(len(EbdB)); 
BERMRC = np.zeros(len(EbdB));


for blk in range(nBlocks):
    h = (nr.normal(0.0, 1.0, (L,1)) + 1j * nr.normal(0.0, 1.0, (L,1)))/np.sqrt(2);
    noise = nr.normal(0.0, np.sqrt(No/2), (L, blockLength)) + 1j * nr.normal(0.0, np.sqrt(No/2), (L, blockLength));
    BitsI = nr.randint(2, size = blockLength);
    BitsQ = nr.randint(2, size = blockLength);
    Sym = (2*BitsI -1) + 1j * (2*BitsQ -1);
    
    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K]) * Sym;
        RxSym = h * TxSym + noise;
        EGCout = np.sum(np.conj(h/abs(h)) * RxSym, axis=0);
        DecBitsI_EGC = (np.real(EGCout) > 0);
        DecBitsQ_EGC = (np.imag(EGCout) > 0);
        BEREGC[K] = BEREGC[K] + np.sum(DecBitsI_EGC != BitsI) + np.sum(DecBitsQ_EGC != BitsQ);
        
        ASel = np.argmax(abs(h));
        ASout = np.conj(h[ASel])*RxSym[ASel,:];
        DecBitsI_AS = (np.real(ASout) > 0);
        DecBitsQ_AS = (np.imag(ASout) > 0);
        BERAS[K] = BERAS[K] + np.sum(DecBitsI_AS != BitsI) + np.sum(DecBitsQ_AS != BitsQ);  
    
    
BEREGC = BEREGC/blockLength/nBlocks/2;
BERAS = BERAS/blockLength/nBlocks/2;
BERMRC = comb(2*L-1, L)/2**L/SNR**L;
plt.yscale('log')
plt.plot(SNRdB, BEREGC,'g-o');
plt.plot(SNRdB, BERAS,'b-.s');
plt.plot(SNRdB, BERMRC,'r:');
plt.grid(1,which='both')
plt.suptitle('BER for EGC and AS')
plt.legend(["EGC","AS", "MRC"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 