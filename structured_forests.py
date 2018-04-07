import os
import time
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA
from skimage import io, filters, color, morphology
from scipy.io import loadmat


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

    @staticmethod
    def _list_ids(dir,set_name):
        ids = []
        files = os.listdir(os.path.join(dir,set_name))
        for fn in files:
            if(fn[-4:]=='.jpg'):
                ids.append(os.path.join(set_name, fn[:-4]))
        return ids


class StructuredRandomForrest(object):
    """
    Structured Random Forrest
    input x and output y are structured (matrix form)
    """
    def __init__(self, n_estimators, max_features, max_depth, verbose=100, n_jobs=3,\
                feature='gradient', use_PCA=False, intermediate_mapping=False, **kwargs):
        super(StructuredRandomForrest, self).__init__()
        if use_PCA:
            self.rf = RandomForestRegressor(n_estimators=n_estimators,max_features=max_features,
                                        max_depth=max_depth,verbose=verbose,n_jobs=n_jobs,\
                                        oob_score=True)
        else:
            self.rf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,
                                        max_depth=max_depth,verbose=verbose,n_jobs=n_jobs,\
                                        oob_score=True)
        self.feature = feature
        self.patch_width = kwargs['patch_width'] if 'patch_width' in kwargs.keys() else 32
        self.label_width = kwargs['label_width'] if 'label_width' in kwargs.keys() else 16
        self.sample_stride = kwargs['sample_stride'] if 'sample_stride' in kwargs.keys() else 8
        self.prediction_stride = kwargs['prediction_stride'] if 'prediction_stride' in kwargs.keys() else 2
        self.threshold = kwargs['threshold'] if 'threshold' in kwargs.keys() else 0.2
        self.epsilon = kwargs['epsilon'] if 'epsilon' in kwargs.keys() else 1e-30
        if self.feature=='gradient':
            self.dataset_generator = self.gradient_feature_label_generator
            self.feature_generator = self.gradient_feature_generator
            self.gradient_window = kwargs['gradient_window']\
                    if 'gradient_window' in kwargs.keys() else morphology.square
            self.gradient_window_size = kwargs['gradient_window_size']\
                    if 'gradient_window_size' in kwargs.keys() else 4
        elif self.feature=='rgb':
            self.dataset_generator = self.rgb_feature_label_generator
            self.feature_generator = self.rgb_feature_generator
        elif self.feature=='default':
            self.dataset_generator = self.default_feature_label_generator
            self.feature_generator = self.default_feature_generator
            self.gradient_window = kwargs['gradient_window']\
                    if 'gradient_window' in kwargs.keys() else morphology.square
            self.gradient_window_size = kwargs['gradient_window_size']\
                    if 'gradient_window_size' in kwargs.keys() else 4
        self.empty_label_sampling_factor = kwargs['empty_label_sampling_factor'] \
                    if 'empty_label_sampling_factor' in kwargs.keys() else 0.1
        self.regular_label_sampling_factor = kwargs['regular_label_sampling_factor'] \
                    if 'regular_label_sampling_factor' in kwargs.keys() else 1
        if self.sample_stride>self.label_width:
            raise(RuntimeWarning('sample stride {} is bigger than label width {}'.format(\
                    self.sample_stride,self.label_width)))
        self.kwargs = kwargs
        self.use_PCA = use_PCA
        if self.use_PCA:
            self.init_PCA()
        self.intermediate_mapping = intermediate_mapping

    def init_PCA(self):
        self.X_n_components = self.kwargs['X_n_components']\
                if 'X_n_components' in self.kwargs.keys() else None
        self.X_PCA = PCA(n_components = self.X_n_components)
        self.Y_n_components = self.kwargs['Y_n_components']\
                if 'Y_n_components' in self.kwargs.keys() else None
        self.Y_PCA = PCA(n_components = self.Y_n_components)
        return

    def gen_dataset(self, imgs, gts):
        if len(imgs)!=len(gts):
            raise(ValueError('image and ground truth has different dimension'))
        indices = np.arange(len(imgs))
        X = []
        Y = []
        for i in indices:
            print('processing image {}/{}'.format(i+1,len(indices)))
            img = imgs[i]
            gt = gts[i]
            features, labels = self.dataset_generator(img, gt)
            X += features
            Y += labels
        if self.intermediate_mapping:
            Y = self.y_to_z_mapping(Y)
            print(len(Y[0]))
        if self.use_PCA:
            print('performing principle component analysis. X compoenents:{}, Y components:{}'\
                                                .format(self.X_n_components, self.Y_n_components))
            X = self.X_PCA.fit_transform(X)
            Y = self.Y_PCA.fit_transform(Y)
        X = np.stack(X, axis=0)
        Y = np.stack(Y, axis=0)
        return X, Y

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

    def predict_edge_map(self,img,groundTruth=None,imshow=False,imsave=False,fn='no_name_given.png'):
        features = self.feature_generator(img)
        # apply PCA to X
        if self.use_PCA:
            features = self.X_PCA.transform(features)
        # predict
        pred_labels = np.array(self.predict(features))
        # apply PCA to Y
        if self.use_PCA:
            pred_labels = self.Y_PCA.inverse_transform(pred_labels)
        # z to y mapping if using intermediate mapping
        if self.intermediate_mapping:
            pred_labels = pred_labels[:,:self.label_width**2]
        # recontruct structured labels
        struct_labels = pred_labels.reshape(len(pred_labels),self.label_width,self.label_width)
        edge_map = np.zeros((img.shape[0],img.shape[1]))
        num_prediction = np.zeros((img.shape[0],img.shape[1]))
        (_img_x,_img_y,_img_c) = img.shape
        label_index = 0
        for x in np.arange(start=0,stop=_img_x-self.label_width+1,step=self.prediction_stride):
            for y in np.arange(start=0,stop=_img_y-self.label_width+1,step=self.prediction_stride):
                edge_map[x:x+self.label_width,y:y+self.label_width] += struct_labels[label_index]
                num_prediction[x:x+self.label_width,y:y+self.label_width] += \
                        np.ones((self.label_width,self.label_width))
                label_index += 1
        # process raw prediction
        processed_edge_map = self.edge_map_processing(edge_map, num_prediction)
        if imshow:
            self._imshow_edge_map(processed_edge_map, img, groundTruth)
        if imsave:
            if fn[:-4] != '.png':
                fn += '_unprocessed.png'
            self.imsave_edge_map(fn, edge_map, img, groundTruth)
        return processed_edge_map

    def edge_map_processing(self, edge_map, num_prediction):
        num_prediction[num_prediction==0] = 1
        normalized_edge_map = edge_map / num_prediction
        _max = normalized_edge_map.max()
        _min = normalized_edge_map.min()
        normalized_edge_map = (normalized_edge_map-_min) / (_max-_min)
        ll_numerator = normalized_edge_map * np.log(1- normalized_edge_map + self.epsilon)
        ll_denominator = ll_numerator + (1-normalized_edge_map) * np.log(normalized_edge_map + self.epsilon)
        ll_edge_map = ll_numerator / ll_denominator
        clean_edge_map = np.zeros(shape=ll_edge_map.shape)
        indices = np.where(ll_edge_map>self.threshold)
        clean_edge_map[indices] = 1
        return clean_edge_map

    def y_to_z_mapping(self, Y):
        """create mapping from strucutured label Y to intermeidate space Z"""
        if len(Y[0])!=self.label_width**2:
            print('input labels have different dimension')
        Z = []
        for label in Y:
            z_label = np.array(label)
            for i in range(self.label_width**2):
                z_label = np.concatenate((z_label, (label[i+1:]==label[i]).astype(int)))
            Z.append(z_label)
        return Z

    def default_feature_label_generator(self, img, gt):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride
        gradient_window = self.gradient_window
        gradient_window_size  = self.gradient_window_size
        empty_label_sampling_factor = self.empty_label_sampling_factor
        regular_label_sampling_factor = self.regular_label_sampling_factor

        gradient = filters.rank.gradient(color.rgb2grey(img),gradient_window(gradient_window_size))
        feat = np.concatenate((img,gradient.reshape(gradient.shape[0],gradient.shape[1],1)),axis=-1)
        _f = []
        _l = []
        (_gt_c, _gt_x, _gt_y) = gt.shape
        (_feat_x,_feat_y,_feat_c) = feat.shape
        pad_width = int(np.ceil((patch_width-label_width)/2))
        padded_feat = np.pad(feat,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
        for x in np.arange(start=0,stop=_gt_x-label_width+1,step=sample_stride):
            for y in np.arange(start=0,stop=_gt_y-label_width+1,step=sample_stride):
                for i in np.arange(start=0,stop=1,step=1):
                    # x,y is the top left pixel of label box
                    _iter_f = np.array(padded_feat[x:x+patch_width,y:y+patch_width,:]).flatten()
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

    def default_feature_generator(self, img):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride
        gradient_window = self.gradient_window
        gradient_window_size  = self.gradient_window_size

        gradient = filters.rank.gradient(color.rgb2grey(img),gradient_window(gradient_window_size))
        feat = np.concatenate((img,gradient.reshape(gradient.shape[0],gradient.shape[1],1)),axis=-1)
        _f = []
        (_feat_x,_feat_y,_feat_c) = feat.shape
        pad_width = int(np.ceil((patch_width-label_width)/2))
        padded_feat = np.pad(feat,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
        for x in np.arange(start=0, stop=_feat_x-label_width+1, step=sample_stride):
            for y in np.arange(start=0, stop=_feat_y-label_width+1, step=sample_stride):
                _iter_f = np.array(padded_feat[x:x+patch_width,y:y+patch_width,:]).flatten()
                _f.append(_iter_f)
        return _f

    def gradient_feature_label_generator(self, img, gt):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride
        gradient_window = self.gradient_window
        gradient_window_size  = self.gradient_window_size
        empty_label_sampling_factor = self.empty_label_sampling_factor
        regular_label_sampling_factor = self.regular_label_sampling_factor

        gradient = filters.rank.gradient(color.rgb2grey(img),gradient_window(gradient_window_size))
        _f = []
        _l = []
        (_gt_c, _gt_x, _gt_y) = gt.shape
        (_grad_x,_grad_y) = gradient.shape
        pad_width = int(np.ceil((patch_width-label_width)/2))
        padded_gradient = np.pad(gradient,pad_width=((pad_width,pad_width),(pad_width,pad_width)),mode='reflect')
        for x in np.arange(start=0,stop=_gt_x-label_width+1,step=sample_stride):
            for y in np.arange(start=0,stop=_gt_y-label_width+1,step=sample_stride):
                for i in np.arange(start=0,stop=1,step=1):
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

    def gradient_feature_generator(self, img):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride
        gradient_window = self.gradient_window
        gradient_window_size  = self.gradient_window_size

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

    def rgb_feature_label_generator(self,img, gt):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride
        empty_label_sampling_factor = self.empty_label_sampling_factor
        regular_label_sampling_factor = self.regular_label_sampling_factor

        _p = []
        _l = []
        (_gt_c, _gt_x, _gt_y) = gt.shape
        (_img_x,_img_y,_img_c) = img.shape
        pad_width = int(np.ceil((patch_width-label_width)/2))
        padded_img = np.pad(img,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
        for x in np.arange(start=0,stop=_gt_x-label_width+1,step=sample_stride):
            for y in np.arange(start=0,stop=_gt_y-label_width+1,step=sample_stride):
                for i in np.arange(start=0,stop=1,step=1):
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

    def rgb_feature_generator(self, img):
        patch_width = self.patch_width
        label_width = self.label_width
        sample_stride = self.sample_stride

        _p = []
        (_img_x,_img_y,_img_c) = img.shape
        pad_width = int(np.ceil((patch_width-label_width)/2))
        padded_img = np.pad(img,pad_width=((pad_width,pad_width),(pad_width,pad_width),(0,0)),mode='reflect')
        for x in np.arange(start=0,stop=_img_x-label_width+1,step=sample_stride):
            for y in np.arange(start=0,stop=_img_y-label_width+1,step=sample_stride):
                _iter_p = np.array(padded_img[x:x+patch_width,y:y+patch_width,:]).flatten()
                _p.append(_iter_p)
        return _p

    def set_working_dir(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.working_dir = path
        return

    def imsave_edge_map(self, filename, edge_map, img, gt):
        filename = filename.replace('/','_')
        path = os.path.join(self.working_dir,filename)
        if gt is not None:
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            ax3.imshow(gt,cmap='Greys')
            ax3.set_title('ground truth')
        else:
            fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax1.set_title('original image')
        ax2.imshow(edge_map,cmap='Greys')
        ax2.set_title('SRF result')
        plt.savefig(path, dpi=400, bbox_inches='tight')
        return

    def save_model(self, filename):
        filename = filename.replace('/','_')
        if self.working_dir is None:
            raise(RuntimeWarning('please set working dir'))
            path = os.path.join(os.getcwd(),filename)
        else:
            path = os.path.join(self.working_dir,filename)
        if path[-4:]!='.pkl':
            path+='.pkl'
        with open(path,"wb") as handler:
            pkl.dump(self, handler, protocol=pkl.HIGHEST_PROTOCOL)
        return

    @staticmethod
    def _imshow_edge_map(edge_map, img, gt):
        if gt is not None:
            fig, (ax1,ax2,ax3) = plt.subplots(1,3)
            ax3.imshow(gt,cmap='Greys')
            ax3.set_title('ground truth')
        else:
            fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(img)
        ax1.set_title('original image')
        ax2.imshow(edge_map,cmap='Greys')
        ax2.set_title('SRF result')
        plt.show()
        return

def load_model(filename):
    with open(filename,"rb") as handler:
        return pkl.load(handler)

def save_log(srf,start_time,train_duration,raw_score,num_patches,empty_labels):
    # logging
    logfile = open(os.path.join(srf.working_dir,'session_info.log'),'a')
    logfile.write('Start Time:\t\t {}\n'.format(time.ctime(start_time)))
    logfile.write('End Time:\t\t {}\n'.format(time.ctime(start_time)))
    logfile.write('Train Duration:\t\t {}\n'.format(train_duration))
    logfile.write('n_estimators:\t\t {}\n'.format(srf.rf.n_estimators))
    logfile.write('max_features:\t\t {}\n'.format(srf.rf.max_features))
    logfile.write('feature:\t\t {}\n'.format(srf.feature))
    logfile.write('use_PCA:\t\t {}\n'.format(srf.use_PCA))
    logfile.write('threshold:\t\t {}\n'.format(srf.threshold))
    logfile.write('intermediate_mapping:\t {}\n'.format(srf.intermediate_mapping))
    logfile.write('X_n_components:\t\t {}\n'.format(srf.X_PCA.n_components))
    logfile.write('Y_n_components:\t\t {}\n'.format(srf.Y_PCA.n_components))
    logfile.write('patch_width:\t\t {}\n'.format(srf.patch_width))
    logfile.write('label_width:\t\t {}\n'.format(srf.label_width))
    logfile.write('sample_stride:\t\t {}\n'.format(srf.sample_stride))
    logfile.write('prediction_stride:\t {}\n'.format(srf.prediction_stride))
    logfile.write('raw_score:\t\t {}\n'.format(raw_score))
    logfile.write('number of patches:\t {}\n'.format(num_patches))
    logfile.write('empty labels:\t\t {}\n'.format(empty_labels))
    logfile.close()
    return

def main():
    bsds = BSDS500(dirpath='./BSR')
    srf = StructuredRandomForrest(n_estimators=10,
                        max_features='auto',
                        max_depth=None,
                        verbose=100,
                        n_jobs=25,
                        feature='default',
                        use_PCA=True,
                        intermediate_mapping=True,
                        X_n_components=64,
                        Y_n_components=16,
                        patch_width=8,
                        label_width=4,
                        sample_stride=2,
                        prediction_stride=2,
                        threshold=0.5,
                        empty_label_sampling_factor=0.3)
    imgs, gts = bsds.get_list_of_data(bsds.train_ids)

    X,Y = srf.gen_dataset(imgs, gts)

    print(X.shape)
    print(Y.shape)

    start_time = time.time()
    srf.fit(X,Y)
    train_end = time.time()
    train_duration = train_end - start_time

    raw_score = srf.raw_score(X,Y)
    asctime = time.asctime().replace('  ',' ').replace(' ','_').replace(':','%')
    ascdate = asctime[:10]+asctime[-4:]
    save_dir = os.path.join('./results',ascdate)
    srf.set_working_dir(save_dir)
    srf.predict_edge_map(bsds.read_image(bsds.test_ids[0]),
                        groundTruth=bsds.get_edge_map(bsds.test_ids[0])[0],
                        #  imshow=True)
                        imsave=True,
                        fn=bsds.test_ids[10])
    save_log(srf=srf,
            start_time=start_time,
            train_duration=train_duration,
            raw_score=raw_score,
            num_patches=len(X),
            empty_labels=np.sum(np.all(Y==0,axis=1))/len(Y))
    srf.save_model(ascdate+'_v1')
    return

if __name__ == '__main__':
    main()
