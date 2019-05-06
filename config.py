# config.py
from yacs.config import CfgNode as CN

_C = CN()
_C.SYSTEM = CN()
_C.SYSTEM.DEVICES = []

_C.MODEL = CN()
_C.MODEL.BACKBONE = "resnet50v2"
_C.MODEL.LEARNING_RATE = 0.001
_C.MODEL.OPTIMIZER = "SGD"
_C.MODEL.EPOCHS = 100
_C.MODEL.BATCH_SIZE = 32
_C.MODEL.USE_DEFORMABLE_CONV = False
_C.MODEL.NUM_DEFORM_GROUP = 0

def get_cfg_defaults():
    return _C.clone()