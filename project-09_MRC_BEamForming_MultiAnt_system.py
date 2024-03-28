import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
import numpy.random as nr
from scipy.special import comb


blockLength = 1000;
nBlocks = 10000;
L = 2;
EbdB = np.arange(1.0,18.1,1.5);
Eb = 10**(EbdB/10);
No = 1;
SNR = 2*Eb/No;
SNRdB = 10*np.log10(2*Eb/No);
BER = np.zeros(len(EbdB)); 
BERt = np.zeros(len(EbdB));


for blk in range(nBlocks):
    h = (nr.normal(0.0, 1.0, (L,1)) + 1j * nr.normal(0.0, 1.0, (L,1)))/np.sqrt(2);
    noise = nr.normal(0.0, np.sqrt(No/2), (L, blockLength)) + 1j * nr.normal(0.0, np.sqrt(No/2), (L, blockLength));
    BitsI = nr.randint(2, size = blockLength);
    BitsQ = nr.randint(2, size = blockLength);
    Sym = (2*BitsI -1) + 1j * (2*BitsQ -1);
    
    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K]) * Sym;
        RxSym = h * TxSym + noise;
        MRCout = np.sum(np.conj(h) * RxSym, axis=0);
        #MRCout = np.sum((h/nl.norm(h)) * RxSym, axis=0);
        DecBitsI = (np.real(MRCout) > 0);
        DecBitsQ = (np.imag(MRCout) > 0);
        BER[K] = BER[K] + np.sum(DecBitsI != BitsI) + np.sum(DecBitsQ != BitsQ);

    
BER = BER/blockLength/nBlocks/2; 
BERt = comb(2*L-1, L)/2**L/SNR**L;
plt.yscale('log')
plt.plot(SNRdB, BER,'g-');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for MRC')
plt.legend(["Simulation", "Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 
