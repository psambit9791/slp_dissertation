import torch
import numpy as np
import os
import sys

wbf = sys.argv[1]
outf = sys.argv[2]

for file in os.listdir(wbf):
    print (file)
    word_lengths = []
    c = 0
    wbs = torch.load(wbf+file)
    for i in range(0, len(wbs)):
        if i != 1 and wbs[i] == 1:
            word_lengths.append(c)
            c = 1
        else:
            c += 1
    word_lengths.append(c)

    np.save(outf+file.replace('.pt', '.npy'), np.array(word_lengths))
    assert wbs.shape[0] == np.array(word_lengths).sum()

    print(torch.split(wbs, word_lengths))
