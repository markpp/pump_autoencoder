import numpy as np
import pickle
import matplotlib.pyplot as plt

# Read pickle object. The CWT_group is a 3D numpy arrays with
# [frekvensrespons, time and pump] on the three axes respectively.
# [frekvens(1/2-1/24 s/h), tid (1-24 h), pumpe (1-35)]
# Hver pkl fil representerer et sample og indeholder frekvensdata fra et
# døgn for alle pumpestationerne. Døgnet går fra kl. 00-23.

# Fremkomst af data:
    # For hver pumpestation er der lavet en Continous Wavelet Transformation (CWT)
    # over 3 dage. Herefter er der skåret et døgn væk i hver side så resultatet
    # ikke bliver biased af kanter. Herefter er CWT'erne fra hver pumpestation
    # blevet lagt ovenpå hinanden, således at der fremkommer et tredimintionelt
    # array svarende til et sample

    # Hvert sample er blevet normaliseret ift. den højeste værdi i samplet.
    # Dette er problematisk ift. output, og jeg arbejder pt. på at finde en bedre
    # måde at normalisere data på.

CWT_group = pickle.load(open('data/val/1.pkl', 'rb'))

n_freq, n_time, n_pump = CWT_group.shape

fig, ax = plt.subplots(1,n_pump,figsize=(4*n_pump,4))

print(CWT_group[:,:,0])
# Plot for hver pumpe
for pump in range(n_pump):
    ax[pump].imshow(CWT_group[:,:,pump])
    ax[pump].set_title('Pump ' + str(pump))
    ax[pump].set_xlabel('Hour')
    ax[pump].set_ylabel('Period')

fig.tight_layout()
plt.show()
