from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.PRETRAIN = 'E:/models/resnet18-5c106cde.pth'
_C.MODEL.CHECKPOINT = 'classification/checkpoints/color'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 4
_C.TRAIN.EPOCHS = 5
_C.TRAIN.DATASET = "E:/test/color_small/train"

_C.TEST = CN()
_C.TEST.DATASET = "E:/test/color_small/test"