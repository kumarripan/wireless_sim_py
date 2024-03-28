import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr

blockLength = 1000;
nBlocks = 50000;
EbdB = np.arange(1.0,45.1,4.0);
Eb = 10**(EbdB/10);
No = 1;
SNR = 2*Eb/No;
SNRdB = 10*np.log10(2*Eb/No);
BER = np.zeros(len(EbdB));
BERt = np.zeros(len(EbdB));


for blk in range(nBlocks):    
    h = (nr.normal(0.0, 1.0, 1) + 1j*nr.normal(0.0, 1.0, 1))/np.sqrt(2);
    #h = (nr.normal(0.0, np.sqrt(2), 1) + 1j*nr.normal(0.0, np.sqrt(2), 1));
    noise = nr.normal(0.0, np.sqrt(No/2), blockLength) + 1j*nr.normal(0.0, np.sqrt(No/2), blockLength);
    
    Bitsl = nr.randint(2, size=blockLength);
    BitsQ = nr.randint(2, size=blockLength);
    Sym = (2*Bitsl-1) + 1j*(2*BitsQ-1); #QPSK Symbol

    for K in range(len(SNRdB)):
        TxSym = np.sqrt(Eb[K])*Sym;
        RxSym = h*TxSym + noise;
        EqSym = RxSym/h;
        
        DecBitsI = (np.real(EqSym) > 0);
        DecBitsQ = (np.imag(EqSym) > 0);
        
        BER[K] = BER[K] + np.sum(DecBitsI != Bitsl) + np.sum(DecBitsQ != BitsQ);

BER = BER/blockLength/nBlocks/2; 
BERt = 1/2/SNR;
#plt.plot(RxSym, )
plt.yscale('log')
plt.plot(SNRdB, BER,'g-');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for Rayleigh Fading Channel QPSK')
plt.legend(["BER","BER Theory"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 