import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import MIMO

blockLength = 1000;
nBlocks=10000;
EbdB = np.arange(1.0, 10.1);

Eb = 10**(EbdB/10);
No = 1;
SNR  = 2*Eb/No;
SNRdB = 10*np.log10(SNR);
SER = np.zeros(len(EbdB));
BER = np.zeros(len(EbdB));

for blk in range(nBlocks):
        BitsI = nr.randint(2, size=blockLength);
        BitsQ = nr.randint(2, size=blockLength);
        sym = (2*BitsI-1) + 1j*(2*BitsQ-1);
        noise = nr.normal(0.0, np.sqrt(No/2), blockLength) + 1j*nr.normal(0.0, np.sqrt(No/2), blockLength);
        
        for K in range (len(EbdB)):
            TxSym = np.sqrt(Eb[K])*sym;
            RxSym =  TxSym + noise;            
            DecBitsI = (np.real(RxSym) > 0);
            DecBitsQ = (np.imag(RxSym) > 0);            
            SER[K] = SER[K] + np.sum(np.logical_or(DecBitsI != BitsI, DecBitsQ != BitsQ));
            BER[K] = BER[K] + np.sum(DecBitsI != BitsI) + np.sum(DecBitsQ != BitsQ);      
        
        
SER = SER/blockLength/nBlocks;
BER = BER/blockLength/nBlocks/2;
        
plt.yscale('log');
plt.plot(SNRdB, SER, 'g-');
plt.plot(SNRdB, 2*MIMO.Q(np.sqrt(SNR)), 'ro');
plt.plot(SNRdB, BER, 'b-.');
plt.plot(SNRdB, MIMO.Q(np.sqrt(SNR)),'ms');
plt.grid(1,which='both');
plt.suptitle('BER and SER for AWGN Channel QPSK')
plt.legend(["SER", "SER Theory", "BER", "BER Theory"], loc="lower left");
plt.xlabel('SNR(db)')
plt.ylabel('SER')