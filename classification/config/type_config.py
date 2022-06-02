from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda:0"
# _C.MODEL.PRETRAIN = '/root/project/car_classifier/resnet50-19c8e357.pth'
_C.MODEL.PRETRAIN = 'classification/checkpoints/type/model_29.pth'
_C.MODEL.CHECKPOINT = 'classification/checkpoints/type'

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.EPOCHS = 30
_C.TRAIN.DATASET = "/root/project/car_classifier/type/train"

_C.TEST = CN()
_C.TEST.DATASET = "/root/project/car_classifier/type/test"

_C.METRIC = CN()
_C.METRIC.LOSS_PIC = 'classification/pic/type/loss_curve.png'
_C.METRIC.ACCURACY_PIC = 'classification/pic/type/accuracy_curve.png'