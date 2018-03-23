import sys
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skimage import io, filters 
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt

class BSDS500(object):
	"""docstring for BSDS500"""
	def __init__(self, dirpath):
		super(BSDS500, self).__init__()
		self.dirpath = dirpath
		self.data_path = os.path.join(self.dirpath,'BSDS500','data')
		self.image_path = os.path.join(self.data_path,'images')
		self.gt_path = os.path.join(self.data_path,'groundTruth')

		self.train_ids = self._list_ids(self.image_path,'train')
		self.test_ids = self._list_ids(self.image_path,'test')
		self.val_ids = self._list_ids(self.image_path,'val')
		self.penguin_ids = ['train/106020','train/106025']

	@staticmethod
	def _list_ids(dir,set_name):
		ids = []
		files = os.listdir(os.path.join(dir,set_name))
		for fn in files:
			if(fn[-4:]=='.jpg'):
				ids.append(os.path.join(set_name, fn[:-4]))
		return ids

	def read_image(self, id):
		return io.imread(os.path.join(self.image_path,id+'.jpg'))

	def get_edge_map(self, id):
		mat = loadmat(os.path.join(self.gt_path,id+'.mat'))
		gt = mat['groundTruth']
		num_gts = gt.shape[1]
		return np.asarray([gt[0,i]['Boundaries'][0,0] for i in range(num_gts)],dtype=np.float32)

	def gen_edge_dataset(self,ids,patch_shape=32,label_shape=16,sample_stride=2):
		"""generate edge mapping dataset with list of image ids"""
		imgs = []
		gts = []
		for id in ids:
			imgs.append(self.read_image(id))
			gts.append(self.get_edge_map(id))
		indices = np.arange(len(imgs))
		X = []
		Y = []
		for i in indices:
			print('processing image {}'.format(ids[i]))
			_img = imgs[i]
			_gt = gts[i]
			_patches, _labels = self._get_img_patch_and_label(img=_img,gt=_gt,patch_shape=patch_shape,
													label_shape=label_shape,sample_stride=sample_stride)
			X += _patches
			Y += _labels
		X_np = np.stack(X,axis=0)
		Y_np = np.stack(Y,axis=0)
		return X_np, Y_np

	@staticmethod
	def _get_img_patch_and_label(img,gt,patch_shape,label_shape,sample_stride):
		_p = []
		_l = []
		(_gt_c, _gt_x, _gt_y) = gt.shape
		(_img_x,_img_y,_img_c) = img.shape
		pad_width = int(np.ceil((patch_shape-label_shape)/2))
		padded_img = np.pad(img,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
		for x in np.arange(start=0,stop=_gt_x-label_shape+1,step=sample_stride):
			for y in np.arange(start=0,stop=_gt_y-label_shape+1,step=sample_stride):
				# x,y is the top left pixel of label box 
				for i in np.arange(start=0,stop=_gt_c,step=1):
					_iter_l = np.array(gt[i,x:x+label_shape,y:y+label_shape])
					_iter_img = np.array(padded_img[x:x+patch_shape,y:y+patch_shape,:])
					_l.append(_iter_l)
					_p.append(_iter_img)
		return _p, _l

class StructuredRandomForrest(object):
	"""
	Structured Random Forrest
	input x and output y are structured (matrix form)
	"""
	def __init__(self,n_estimators,max_features,max_depth,verbose=1000,n_jobs=3):
		super(StructuredRandomForrest, self).__init__()
		self.rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,
										max_depth=max_depth,verbose=verbose,n_jobs=n_jobs)
		self.x_shape = None
		self.y_shape = None

	def fit(self,X,Y):
		if self.x_shape == None and self.y_shape == None:
			self.x_shape = X[0].shape
			self.y_shape = Y[0].shape
		flat_X = X.reshape((len(X),X[0].size))
		flat_Y = Y.reshape((len(Y),Y[0].size))
		return self.rf.fit(flat_X,flat_Y)

	def predict(self,X):
		if self.x_shape == None or self.y_shape == None:
			raise(AttributeError('shape of input and output unknown.'))
		flat_X = X.reshape((len(X),X[0].size))
		flat_Y = np.array(self.rf.predict(flat_X))
		Y = flat_Y.reshape((len(flat_Y),)+self.y_shape)
		return Y

	def score(self,X,Y):
		"""per pixel accuracy"""
		pred_Y = self.predict(X)
		score = []
		label_size = Y[0].size
		for i in range(len(Y)):
			score.append(np.sum(Y[i]==pred_Y[i])/label_size)
		return np.mean(score)


def main():
	bsds = BSDS500(dirpath='./BSR')
	X, Y = bsds.gen_edge_dataset(ids=bsds.penguin_ids,sample_stride=8)
	srf = StructuredRandomForrest(n_estimators=10,max_features=100,max_depth=100,verbose=1000,n_jobs=3)
	srf.fit(X[:5000],Y[:5000])
	srf.predict(X[-5000:])
	srf.score(X[-5000:],Y[-5000:])
	return

if __name__ == '__main__':
	main()