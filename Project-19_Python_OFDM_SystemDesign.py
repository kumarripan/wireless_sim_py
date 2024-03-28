import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import numpy.fft as nf


Nsub = 256;
Ncp = int(Nsub/4);
nBlocks = 10000;
L = 2;
EbdB = np.arange(1.0,55.1,4.0);
Eb = 10**(EbdB/10);
No = 1;
SNR = 2*Eb/No;
SNRdB = 10*np.log10(2*Eb/No);
BER = np.zeros(len(EbdB));
BERt = np.zeros(len(EbdB));


for blk in range(nBlocks):
    noise = nr.normal(0.0, np.sqrt(No/2), L+Nsub+Ncp-1) + 1j* nr.normal(0.0, np.sqrt(No/2), L+Nsub+Ncp-1);
    BitsI = nr.randint(2, size=Nsub);
    BitsQ = nr.randint(2, size = Nsub);
    Sym = (2*BitsI-1) + 1j* (2*BitsQ-1);
    h = (nr.normal(0.0, 1.0, L) + 1j * nr.normal(0.0, 1.0, L))/np.sqrt(2);
    hFFT = nf.fft(np.concatenate((h, np.zeros(Nsub-L))));
    
    for K in range(SNRdB.size):
        LoadedSym = np.sqrt(Eb[K]) * Sym;
        TxSamples = nf.ifft(LoadedSym);        
        TxSamCP = np.concatenate((TxSamples[Nsub-Ncp:Nsub], TxSamples));
        
        # Liner convolution due to addition of CP
        RxSamCP = np.convolve(h, TxSamCP) + noise; # out put length = L + Ncp_Msub-1
        RxSamples = RxSamCP[Ncp:Ncp+Nsub]; # CP removal
        RxSym = nf.fft(RxSamples); # time domain to Freq domain
        ZFout = RxSym/hFFT;
        DecBitsI = (np.real(ZFout) >0);
        DecBitsQ = (np.imag(ZFout) >0);
        BER[K] = BER[K]  + np.sum(DecBitsI != BitsI) + np.sum(DecBitsQ != BitsQ);        
    
BER = BER/nBlocks/Nsub/2;
SNReff = SNR*L/Nsub;
BERt = 1/2/SNReff;
plt.yscale('log')
plt.plot(SNRdB, BER,'g--');
plt.plot(TxSamples, TxSamples,'b-');
plt.plot(SNRdB, BERt,'ro');
plt.grid(1,which='both')
plt.suptitle('BER for OFDM Channel')
plt.legend(["Simulation", "TxSamples", "Theory"], loc ="lower left");
#plt.legend(["Simulation"], loc ="lower left");
plt.xlabel('SNR (dB)')
plt.ylabel('BER') 