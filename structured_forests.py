import sys
import os
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomForestClassifier
from skimage import io, filters, color, morphology
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

	def get_list_of_data(self, ids):
		imgs = []
		gts = []
		for id in ids:
			imgs.append(self.read_image(id))
			gts.append(self.get_edge_map(id))
		return imgs, gts


class StructuredRandomForrest(object):
	"""
	Structured Random Forrest
	input x and output y are structured (matrix form)
	"""
	def __init__(self,n_estimators,max_features,max_depth,verbose=100,n_jobs=3,feature='gradient',**kwargs):
		super(StructuredRandomForrest, self).__init__()
		self.rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,
										max_depth=max_depth,verbose=verbose,n_jobs=n_jobs,\
										oob_score=True)
		self.feature = feature
		self.patch_width = kwargs['patch_width'] if 'patch_width' in kwargs.keys() else 32
		self.label_width = kwargs['label_width'] if 'label_width' in kwargs.keys() else 16
		self.sample_stride = kwargs['sample_stride'] if 'sample_stride' in kwargs.keys() else 8
		self.prediction_stride = kwargs['prediction_stride'] if 'prediction_stride' in kwargs.keys() else 2
		if self.feature=='gradient':
			self.dataset_generator = self._gradient_feature_label_generator
			self.gradient_window = kwargs['gradient_window'] if 'gradient_window' in kwargs.keys() else morphology.square
			self.gradient_window_size = kwargs['gradient_window_size'] if 'gradient_window_size' in kwargs.keys() else 4
			self.feature_generator = self._gradient_feature_generator
		elif self.feature=='rgb':
			self.dataset_generator = self._rgb_feature_label_generator
			self.feature_generator = self._rgb_feature_generator
		self.empty_label_sampling_factor = kwargs['empty_label_sampling_factor'] \
										if 'empty_label_sampling_factor' in kwargs.keys() else 0.1
		self.regular_label_sampling_factor = kwargs['regular_label_sampling_factor'] \
										if 'regular_label_sampling_factor' in kwargs.keys() else 1
		if self.sample_stride>self.label_width:
			raise(RuntimeWarning('sample stride {} is bigger than label width {}'.format(self.sample_stride,self.label_width)))

	def gen_dataset(self, imgs, gts):
		if len(imgs)!=len(gts):
			raise(ValueError('image and ground truth has different dimension'))
		indices = np.arange(len(imgs))
		X = []
		Y = []
		args = {'patch_width':self.patch_width, 
				'label_width':self.label_width,
				'sample_stride':self.sample_stride,
				'empty_label_sampling_factor':self.empty_label_sampling_factor, 
				'regular_label_sampling_factor':self.regular_label_sampling_factor}
		for i in indices:
			print('processing image {}/{}'.format(i+1,len(indices)))
			img = imgs[i]
			gt = gts[i]
			args.update([('img',img),('gt',gt)])
			if self.feature == 'gradient':
				args.update([('gradient_window',self.gradient_window),('gradient_window_size',self.gradient_window_size)])
			features, labels = self.dataset_generator(**args)
			X += features
			Y += labels
		X_np = np.stack(X, axis=0)
		Y_np = np.stack(Y, axis=0)
		return X_np, Y_np

	def fit(self,X,Y):
		return self.rf.fit(X,Y)

	def predict(self,X):
		return self.rf.predict(X)

	def raw_score(self,X,Y):
		"""accuracy by simply comparing labels"""
		return self.rf.score(X,Y)

	def pixel_score(self,X,Y):
		"""accuracy based on pixel prediction"""
		pred_Y = self.predict(X)
		score = []
		label_size = self.label_width**2
		for i in range(len(Y)):
			score.append(np.sum(Y[i]==pred_Y[i])/label_size)
		mean_score = np.mean(score)
		return mean_score

	def predict_edge_map(self,img,imshow=False):
		args = {'img':img,
				'patch_width':self.patch_width,
				'label_width':self.label_width,
				'sample_stride':self.prediction_stride}
		if self.feature == 'gradient':
			args.update([('gradient_window',self.gradient_window),
						('gradient_window_size',self.gradient_window_size)])
		features = self.feature_generator(**args)
		pred_labels = np.array(self.predict(features))
		struct_labels = pred_labels.reshape(len(pred_labels),self.label_width,self.label_width)
		pad_width = int(np.ceil((self.patch_width-self.label_width)/2))
		edge_map = np.zeros((img.shape[0],img.shape[1]))
		num_prediction = np.zeros((img.shape[0],img.shape[1]))
		(_img_x,_img_y,_img_c) = img.shape
		label_index = 0
		for x in np.arange(start=0,stop=_img_x-self.label_width+1,step=self.prediction_stride):
			for y in np.arange(start=0,stop=_img_y-self.label_width+1,step=self.prediction_stride):
				edge_map[x:x+self.label_width,y:y+self.label_width] += struct_labels[label_index]
				num_prediction[x:x+self.label_width,y:y+self.label_width] += np.ones((self.label_width,self.label_width))
				label_index += 1
		num_prediction[num_prediction==0] = 1
		if imshow:
			self._imshow_edge_map(edge_map, img)
		return edge_map

	@staticmethod
	def _gradient_feature_label_generator(img, gt, patch_width, label_width, 
											sample_stride, gradient_window, 
											gradient_window_size,
											empty_label_sampling_factor,
											regular_label_sampling_factor
											):
		gradient = filters.rank.gradient(color.rgb2grey(img),gradient_window(gradient_window_size))
		_f = []
		_l = []
		(_gt_c, _gt_x, _gt_y) = gt.shape
		(_grad_x,_grad_y) = gradient.shape
		pad_width = int(np.ceil((patch_width-label_width)/2))
		padded_gradient = np.pad(gradient,pad_width=((pad_width,pad_width),(pad_width,pad_width)),mode='reflect')
		for x in np.arange(start=0,stop=_gt_x-label_width+1,step=sample_stride):
			for y in np.arange(start=0,stop=_gt_y-label_width+1,step=sample_stride):
				for i in np.arange(start=0,stop=_gt_c,step=1):
					# x,y is the top left pixel of label box 
					_iter_f = np.array(padded_gradient[x:x+patch_width,y:y+patch_width]).flatten()
					_iter_l = np.array(gt[i,x:x+label_width,y:y+label_width]).flatten()
					rand = np.random.rand()
					if np.any(_iter_l):
						if rand < regular_label_sampling_factor:
							_f.append(_iter_f)
							_l.append(_iter_l)
					else:
						if rand < empty_label_sampling_factor:
							_f.append(_iter_f)
							_l.append(_iter_l)
		return _f, _l

	@staticmethod
	def _gradient_feature_generator(img, patch_width, label_width, sample_stride, 
									gradient_window, gradient_window_size):
		gradient = filters.rank.gradient(color.rgb2grey(img),gradient_window(gradient_window_size))
		_f = []
		(_grad_x,_grad_y) = gradient.shape
		pad_width = int(np.ceil((patch_width-label_width)/2))
		padded_gradient = np.pad(gradient,pad_width=((pad_width,pad_width),(pad_width,pad_width)),mode='reflect')
		for x in np.arange(start=0, stop=_grad_x-label_width+1, step=sample_stride):
			for y in np.arange(start=0, stop=_grad_y-label_width+1, step=sample_stride):
				_iter_f = np.array(padded_gradient[x:x+patch_width,y:y+patch_width]).flatten()
				_f.append(_iter_f)
		return _f

	@staticmethod
	def _rgb_feature_label_generator(img, gt, patch_width, label_width,
									sample_stride, empty_label_sampling_factor,
									regular_label_sampling_factor):
		_p = []
		_l = []
		(_gt_c, _gt_x, _gt_y) = gt.shape
		(_img_x,_img_y,_img_c) = img.shape
		pad_width = int(np.ceil((patch_width-label_width)/2))
		padded_img = np.pad(img,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
		for x in np.arange(start=0,stop=_gt_x-label_width+1,step=sample_stride):
			for y in np.arange(start=0,stop=_gt_y-label_width+1,step=sample_stride):
				for i in np.arange(start=0,stop=_gt_c,step=1):
					# x,y is the top left pixel of label box 
					_iter_l = np.array(gt[i,x:x+label_width,y:y+label_width]).flatten()
					_iter_p = np.array(padded_img[x:x+patch_width,y:y+patch_width,:]).flatten()
					
					rand = np.random.rand()
					if np.any(_l):
						if rand < regular_label_sampling_factor:
							_l.append(_iter_l)
							_p.append(_iter_p)
					else:
						if rand < empty_label_sampling_factor:
							_l.append(_iter_l)
							_p.append(_iter_p)
		return _p, _l

	@staticmethod
	def _rgb_feature_generator(img,patch_width,label_width,sample_stride):
		_p = []
		(_img_x,_img_y,_img_c) = img.shape
		pad_width = int(np.ceil((patch_width-label_width)/2))
		padded_img = np.pad(img,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
		for x in np.arange(start=0,stop=_img_x-label_width+1,step=sample_stride):
			for y in np.arange(start=0,stop=_img_y-label_width+1,step=sample_stride):
				_iter_p = np.array(padded_img[x:x+patch_width,y:y+patch_width,:]).flatten()
				_p.append(_iter_p)
		return _p

	@staticmethod
	def _imshow_edge_map(edge_map, img):
		fig, (ax1,ax2) = plt.subplots(2)
		ax1.imshow(edge_map)
		ax2.imshow(img)
		plt.show()
		return

def save_model(SRF, filename):
	with open(filename,"wb") as handler:
		pkl.dump(SRF,handler,protocol=pkl.HIGHEST_PROTOCOL)
	return

def load_model(filename):
	with open(filename,"rb") as handler:
		return pkl.load(handler)

def main():
	bsds = BSDS500(dirpath='./BSR')
	srf = StructuredRandomForrest(n_estimators=100,
								max_features='auto',
								max_depth=None,
								verbose=1000,
								n_jobs=3,
								feature='gradient',
								patch_width=8,
								label_width=4,
								sample_stride=4,
								empty_label_sampling_factor=0.01)
	imgs, gts = bsds.get_list_of_data(bsds.train_ids[0:100])
	X,Y = srf.gen_dataset(imgs, gts)
	print(X.shape)
	print(Y.shape)
	print(np.all(Y==0,axis=1).shape)
	print(np.sum(np.all(Y==0,axis=1))/len(Y))
	srf.fit(X,Y)
	print(srf.raw_score(X,Y))
	print(bsds.train_ids[2])
	edge = srf.predict_edge_map(bsds.read_image(bsds.train_ids[10]),True)
	return

if __name__ == '__main__':
	main()