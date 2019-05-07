# dataloader.py
import math
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from random import shuffle
from tensorflow.python.keras.utils import to_categorical, Sequence

class GetDataset():
    def __init__(self, x, y, num_classes=10, preproc_fn=None, augment_fn=None, do_shuffle=True):
        """Get single data
        
        Args:
          x:
          y:
          num_classes:
          preproc_fn:
          augment_fn:
        Reuturns:
          img:
          target:
        """
        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.preproc_fn = preproc_fn
        self.augment_fn = augment_fn
        self.do_shuffle = do_shuffle
        
        # -- init -- #
        self.counter = 0
        if do_shuffle:
            self._shuffle_item()
            
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        
        img, target = self.x[index], self.y[index]
        if self.augment_fn is not None:
            img = self.augment_fn.augment_image(img)
        
        img = img.astype('float32')
        if self.preproc_fn is not None:
            img = self.preproc_fn(img)
            
        target = to_categorical(target, self.num_classes)
        
        self.counter = (self.counter + 1) % len(self.x)
        if self.counter % len(self.x) == 0:
            self._shuffle_item()
            
        if len(img.shape) != 3:
            img = img[:,:,np.newaxis]
        return img, target
    
    def __next__(self):
        return self.__getitem__(self.counter)
    
    def _shuffle_item(self):
        temp_item = list(zip(self.x, self.y))
        shuffle(temp_item)
        self.x, self.y = zip(*temp_item)

class DataLoader(Sequence):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index+1) * self.batch_size
        end_idx = end_idx if end_idx <= len(self.dataset) else len(self.dataset)
        
        imgs, targets = [], []
        for i in range(start_idx, end_idx):
            img, target = next(self.dataset)
            imgs.append(img)
            targets.append(target)
        return np.array(imgs), np.array(targets)
    
    
    
class TrainingAugmentation(object):
    sometimes = lambda aug: iaa.Sometimes(0.75, aug)
    
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        sometimes(iaa.Affine(scale=(0.8, 1.2),
                             rotate=(-45, 45),
                             translate_percent=(-0.2, 0.2),
                             mode="constant")
                 ),
    ])
    
class TestingAugmentation(object):
    sometimes = lambda aug: iaa.Sometimes(0.75, aug)
    
    augmentation = iaa.Sequential([
        sometimes(iaa.Affine(scale=(0.9, 1.1),
                             rotate=(-60, 60),
                             shear=(-30,30),
                             mode="constant")),
        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
    ])