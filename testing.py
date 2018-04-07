import os
import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from skimage import io, filters, color, morphology
from scipy.io import loadmat

import structured_forests
from structured_forests import StructuredRandomForrest, BSDS500

bsds = BSDS500(dirpath='./BSR')
srf = structured_forests.load_model('./results/Sat_Apr_7_2018/Sat_Apr_7_2018_v1.pkl')
srf.threshold = 0.1
for i in range(10):
    start = time.time()
    edge_map = srf.predict_edge_map(bsds.read_image(bsds.test_ids[i]),
                        groundTruth=bsds.get_edge_map(bsds.test_ids[i])[0],
                        #  imshow=False)
                        imsave=True,
                        fn=bsds.test_ids[i]+'_floating_threshold')
    end = time.time()
    print('latency: {:.4f}s'.format(end - start))
