# config.py
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICES = []
_C.SYSTEM.MIX_PRECISION = False

_C.MODEL = CN()
_C.MODEL.BACKBONE = ""
_C.MODEL.LEARNING_RATE = 0.001
_C.MODEL.OPTIMIZER = "SGD"
_C.MODEL.EPOCHS = 100
_C.MODEL.BATCH_SIZE = 32
_C.MODEL.USE_DEFORMABLE_CONV = False
_C.MODEL.NUM_DEFORM_GROUP = 0
_C.MODEL.INFERENCE_SIZE = 128


_C.DATASET = CN()
_C.DATASET.SET = "mnist" # mnist / fasion-mnist / cifar10 / cat-dog

def get_cfg_defaults():
    return _C.clone()
