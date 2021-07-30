import numpy as np
import sys

data = open(sys.argv[1], 'r').read().split('\n')[1:]

for line in data:
    filename = line.split(',')[0]
    vector = np.array([float(n) for n in line.split(',')[1:]])
    np.save(sys.argv[2]+filename.replace('.wav', ''), vector)
