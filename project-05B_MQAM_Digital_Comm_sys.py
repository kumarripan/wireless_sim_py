import numpy as np
import matplotlib.pyplot as plt
import numpy.random as nr
import MIMO

blockLength = 1000;
nBlocks=10000;
M = 16; #QAM order
sqM = np.int(np.sqrt(M)); #Total numOf sym in each PAM
EbdB = np.arange(1.0, 12.1, 1.0);
Eb = 10 ** (EbdB/10);

n=np.log2(M);
Es = n*Eb;  #Avg symol power
No = 1;
SNR = Es/No; # SNR
SNRdB = 10* log10(SNR);

SER = np.zeros(len(EbdB));

for blk in range(nBlocks)


