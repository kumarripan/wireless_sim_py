import numpy as np
import matplotlib.pyplot as plt

blockLength = 10000000;
nbins = 1000;
h = (np.random.normal(0.0, 1.0, blockLength)+1j*np.random.normal(0.0, 1.0, blockLength))/np.sqrt(2);
amp = np.abs(h)
phi = np.angle(h)

plt.figure(1)
plt.hist(amp,bins=nbins,density=True);
plt.suptitle('Rayleigh PDF')
plt.xlabel('x')
plt.ylabel('$f_A$(a)')

plt.figure(2)
plt.hist(phi,bins=nbins,density=True);
plt.suptitle('Phase PDF')
plt.xlabel('x')
plt.ylabel('$f_\Phi(\phi)$')