import os
import sys
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

#  sobel = filters.sobel
#  scharr = filters.scharr
#  prewitt = filters.prewitt

#  img = bsds.read_image('train/236017')
#  img = color.rgb2gray(img)
#  sobel_result = sobel(img)
#  scharr_result = scharr(img)
#  prewitt_result = prewitt(img)

#  for _str,result in [('sobel',sobel_result),('scharr',scharr_result),('prewitt',prewitt_result)]:
#      plt.imshow(sobel_result,cmap='Greys')
#      plt.title(_str+' result')
#      plt.savefig('./figs/'+_str+'_result.png', dpi=400, bbox_inches='tight')

srf = structured_forests.load_model('./results/Sun_Apr_8_2018/Sun_Apr_8_2018_v2.pkl')
#  edge_map = srf.predict_edge_map(bsds.read_image('train/236017'),
#                      groundTruth=bsds.get_edge_map('train/236017')[0],
#                      imshow=True)
#                      #  imsave=True,
#                      #  fn=bsds.test_ids[i]+'_floating_threshold')
#  plt.imshow(edge_map,cmap='Greys')
#  plt.title('StructuredRandomForrest result')
#  plt.savefig('./figs/srf_result.png', dpi=400, bbox_inches='tight')

#  sys.exit()
for i in range(10):
    start = time.time()
    edge_map = srf.predict_edge_map(bsds.read_image(bsds.test_ids[i]),
                        groundTruth=bsds.get_edge_map(bsds.test_ids[i])[0],
                        #  imshow=True)
                        imsave=True,
                        fn=bsds.test_ids[i]+'_floating_threshold')
    end = time.time()
    print('latency: {:.4f}s'.format(end - start))
