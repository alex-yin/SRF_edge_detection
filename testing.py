import os
import sys
import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from skimage import io, filters, color, morphology, feature
from scipy.io import loadmat

import structured_forests
from structured_forests import StructuredRandomForrest, BSDS500

bsds = BSDS500(dirpath='./BSR')

sobel = filters.sobel
scharr = filters.scharr
prewitt = filters.prewitt
canny = feature.canny

img = bsds.read_image('train/247085')
img = color.rgb2gray(img)
sobel_result = sobel(img)
scharr_result = scharr(img)
prewitt_result = prewitt(img)
canny_result = canny(img)

for _str,result in [('sobel',sobel_result),('scharr',scharr_result),('prewitt',prewitt_result),('canny',canny_result)]:
    plt.imshow(result,cmap='Greys')
    plt.title(_str+' result')
    plt.show()

sys.exit()
srf = structured_forests.load_model('./results/Sat_Apr_7_2018/Sat_Apr_7_2018_v1.pkl')
edge_map = srf.predict_edge_map(bsds.read_image('train/236017'),
                    groundTruth=bsds.get_edge_map('train/236017')[0],
                    imshow=False)
                    #  imsave=True,
                    #  fn=bsds.test_ids[i]+'_floating_threshold')
plt.imshow(edge_map,cmap='Greys')
plt.title('StructuredRandomForrest result')
plt.savefig('./figs/srf_result.png', dpi=400, bbox_inches='tight')

sys.exit()
for i in range(10):
    start = time.time()
    edge_map = srf.predict_edge_map(bsds.read_image(bsds.test_ids[i]),
                        groundTruth=bsds.get_edge_map(bsds.test_ids[i])[0],
                        #  imshow=False)
                        imsave=True,
                        fn=bsds.test_ids[i]+'_floating_threshold')
    end = time.time()
    print('latency: {:.4f}s'.format(end - start))
