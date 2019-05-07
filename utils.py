import cv2
import glob
import os
import numpy as np
from tensorflow.python.keras import datasets
from tqdm import tqdm
from pathlib import Path


class Fetch_dataset():
    def __init__(self, dataset_name="mnist"):
        self.dataset_map = DATASET_MAP
        self.dataset_name = dataset_name
        
    def load_data(self):
        assert self._check_dataset_in_map, "{} not found in datamap, check you have make dataset available".format(self.dataset_name)
        fetch_fn = self.dataset_map[self.dataset_name]
        
        return fetch_fn.load_data()
    
    def _check_dataset_in_map(self):
        return self.dataset_name in list(self.dataset_map.keys())
    
class DatasetCatdog():
    def __init__(self, train_path=[], test_path=[], num_train_images=5000, imsize=(256,256)):
        self.train_path = train_path
        self.test_path = test_path
        self.num_train_images = num_train_images
        self.imsize = imsize
        
    def load_data(self):
        train_path = self.read_path(self.train_path)
        n_take = self.num_train_images if len(train_path) >= self.num_train_images else len(train_path)
        train_path = np.random.choice(train_path, n_take)
        train_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(i), self.imsize), cv2.COLOR_BGR2RGB) for i in tqdm(train_path)])
        train_targets = [self.parse_target(i) for i in train_path]
        
        test_path = self.read_path(self.test_path)
        test_images = np.array([cv2.cvtColor(cv2.resize(cv2.imread(i), self.imsize), cv2.COLOR_BGR2RGB) for i in tqdm(test_path)])
        test_targets = [self.parse_target(i) for i in test_path]
        
        return (train_images, np.array(train_targets, dtype="int8") ), (test_images, np.array(test_targets,dtype="int8") )
        
    def read_path(self, path):
        all_path = [glob.glob(i+"*.jpg") for i in path]
        all_path = [item for i in all_path for item in i]
        return all_path
    
    def parse_target(self, path):
        return int("dog" in os.path.basename(path))

class DatasetPCAM():
    def __init__(self, train_path=[], test_path=[], num_train_images=5000):
        self.train_path = train_path
        self.test_path = test_path
        self.num_train_images = num_train_images
        
    def load_data(self):
        train_path = self.read_path(self.train_path)
        n_take = self.num_train_images if len(train_path) >= self.num_train_images else len(train_path)
        train_path = np.random.choice(train_path, n_take)
        train_images = np.array([cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in tqdm(train_path)])
        train_targets = [self.parse_target(i) for i in train_path]
        
        test_path = self.read_path(self.test_path)
        test_images = np.array([cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB) for i in tqdm(test_path)])
        test_targets = [self.parse_target(i) for i in test_path]
        
        return (train_images, np.array(train_targets, dtype="int8") ), (test_images, np.array(test_targets,dtype="int8") )
        
    def read_path(self, path):
        all_path = [glob.glob(i+"*.tif") for i in path]
        all_path = [item for i in all_path for item in i]
        return all_path
    
    def parse_target(self, path):
        p = Path(path)
        return p.parts[-2]
    
DATASET_MAP = {
    "mnist":datasets.mnist,
    "fashion-mnist":datasets.fashion_mnist,
    "cifar10":datasets.cifar10,
    "cifar100":datasets.cifar100,
    "cat-dog":DatasetCatdog(train_path=["/mnt/extension/experiment/cat_dog/train/training/"],
                            test_path=["/mnt/extension/experiment/cat_dog/train/valid/"], 
                            num_train_images=5000),
    "pcam": DatasetPCAM(
        train_path=["/mnt/extension/experiment/pcam/base_dir/train_dir/0/",
                    "/mnt/extension/experiment/pcam/base_dir/train_dir/1/"],
        test_path=["/mnt/extension/experiment/pcam/base_dir/val_dir/0/",
                   "/mnt/extension/experiment/pcam/base_dir/val_dir/1/"], 
        num_train_images=5000)
        }