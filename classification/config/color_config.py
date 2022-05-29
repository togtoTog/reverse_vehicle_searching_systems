from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
# _C.MODEL.PRETRAIN = 'D:/Workspace/PyCharm/vehicle_classification/checkpoints/resnet34-333f7ec4.pth'
_C.MODEL.PRETRAIN = 'classification/checkpoints/color/model_29.pth'
_C.MODEL.CHECKPOINT = 'classification/checkpoints/color'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.EPOCHS = 10
_C.TRAIN.DATASET = "D:/Workspace/PyCharm/vehicle_classification/color_small/train"

_C.TEST = CN()
_C.TEST.DATASET = "D:/Workspace/PyCharm/vehicle_classification/color_small/test"

_C.METRIC = CN()
_C.METRIC.LOSS_PIC = 'classification/pic/loss_curve.png'
_C.METRIC.ACCURACY_PIC = 'classification/pic/accuracy_curve.png'