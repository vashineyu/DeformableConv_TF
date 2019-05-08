# train.py / main
import argparse
import glob
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import get_cfg_defaults
from DeformableConv_TF.model import preproc_fn, build_model, build_resnet_model
from dataloader import GetDataset, DataLoader, TrainingAugmentation, TestingAugmentation
from sklearn.model_selection import train_test_split
from utils import Fetch_dataset

parser = argparse.ArgumentParser(description="Cats/Dogs playground parameters")
parser.add_argument(
    "--config-file",
    default=None,
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)
args = parser.parse_args()

cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
if args.opts is not None:
    cfg.merge_from_list(args.opts)
cfg.freeze()
print(cfg)

devices = ",".join(str(i) for i in cfg.SYSTEM.DEVICES)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = devices

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = str(int(cfg.SYSTEM.MIX_PRECISION))

dset = Fetch_dataset(dataset_name=cfg.DATASET.SET)
trainset, testset = dset.load_data()
x_train, y_train = trainset
x_test, y_test = testset

idx = np.arange(len(x_train))
idx_train, idx_valid = train_test_split(idx, test_size=0.1)
x_train, x_valid = x_train[idx_train], x_train[idx_valid]
y_train, y_valid = y_train[idx_train], y_train[idx_valid]

dataset_train = GetDataset(x=x_train, y=y_train, num_classes=max(y_test)+1, 
                           preproc_fn=preproc_fn, augment_fn=TrainingAugmentation.augmentation)
dataset_valid = GetDataset(x=x_valid, y=y_valid, num_classes=max(y_test)+1, 
                           preproc_fn=preproc_fn, augment_fn=TestingAugmentation.augmentation)

dataloader = DataLoader(dataset=dataset_train, batch_size=cfg.MODEL.BATCH_SIZE)

""" Pre-occupy the gpu, prevent after loading data but found gpu has already been taken"""
if cfg.MODEL.BACKBONE == "":
    model = build_model(input_shape=(256,256,3), 
                        output_num=2, 
                        use_deformable=cfg.MODEL.USE_DEFORMABLE_CONV,
                        num_deform_group=cfg.MODEL.NUM_DEFORM_GROUP)
else:
    model = build_resnet_model(input_shape=(256,256,3), 
                               output_num=2, 
                               use_deformable=cfg.MODEL.USE_DEFORMABLE_CONV,
                               num_deform_group=cfg.MODEL.NUM_DEFORM_GROUP,
                               backbone=cfg.MODEL.BACKBONE)

optim = tf.keras.optimizers.SGD(lr=cfg.MODEL.LEARNING_RATE, nesterov=True, momentum=0.95)
#optim = tf.keras.optimizers.Adam(lr=cfg.MODEL.LEARNING_RATE)
model.compile(loss="categorical_crossentropy", metrics=["acc"], optimizer=optim)
model.summary()

x_valid, y_valid = next(iter(DataLoader(dataset_valid, batch_size=len(dataset_valid))))
print(x_valid.shape)

model.fit_generator(dataloader, 
                    epochs=cfg.MODEL.EPOCHS, 
                    steps_per_epoch=len(dataloader), 
                    validation_data=(x_valid, y_valid))

## -- ##
dataset_test = GetDataset(x=x_test, y=y_test, num_classes=max(y_test)+1, 
                          preproc_fn=preproc_fn, augment_fn=TestingAugmentation.augmentation)
x_test, y_test = next(iter(DataLoader(dataset_test, batch_size=len(dataset_valid))))
print(x_test.shape)
tst_pred = model.predict(x_test, verbose=1, batch_size=cfg.MODEL.INFERENCE_SIZE)

testset_accuracy = np.sum(tst_pred.argmax(axis=1) == y_test.argmax(axis=1)) / len(y_test)
print("Test accuracy: {:.4f}".format(testset_accuracy))

model.save("./model.h5")
