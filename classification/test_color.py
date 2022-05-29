import os
import time

from PIL import Image
import torch
from torchvision import transforms, models
import matplotlib.pyplot as plt

from config import color_cfg

plt.ion()  # interactive mode

# 模型存储路径
model_save_path = 'classification/checkpoints/color/model_23.pth'

# ------------------------ 加载数据 --------------------------- #
# Data augmentation and normalization for training
# Just normalization for validation
# 定义预训练变换
preprocess_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#设置测试集
test_transform = transforms.Compose([
    transforms.Resize((224, 224)), #resize到224x224大小
    transforms.ToTensor(), #转化成Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) #正则化
])

# class_names = ['0', '180', '270', '90']  # 这个顺序很重要，要和训练时候的类名顺序一致
# class_names = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9']  # 这个顺序很重要，要和训练时候的类名顺序一致
class_names = ['yellow', 'black', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown']  # 这个顺序很重要，要和训练时候的类名顺序一致
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------ 载入模型并且训练 --------------------------- #
# model = torch.load(model_save_path)
model = models.resnet34(pretrained=False).to('cuda:0')
model.load_state_dict(torch.load(model_save_path))
model.eval()
# print(model)

# image_PIL = Image.open(os.path.join(color_cfg.TEST.DATASET, '1/0002_c011_00083575_0.jpg'))
image_PIL = Image.open(os.path.join(color_cfg.TEST.DATASET, '8/0237_c016_00065285_0.jpg'))
# image_PIL = Image.open('color_small/test/1/0404_c012_00015760_0.jpg')
# image_PIL = Image.open('color_small/test/2/0005_c007_00078465_0.jpg')
# image_PIL = Image.open('color_small/test/2/0006_c014_00022345_0.jpg')
# image_PIL = Image.open('color_small/test/2/0005_c003_00077690_0.jpg')
# image_PIL = Image.open('color_small/test/3/0102_c001_00008215_0.jpg')
# image_PIL = Image.open('color_small/test/3/0090_c003_00078465_1.jpg')
# image_PIL = Image.open('color_small/test/8/0237_c016_00065285_0.jpg')
# image_PIL = Image.open('monkey/validation/n1/n100.jpg')
#
# image_tensor = preprocess_transform(image_PIL)
image_tensor = test_transform(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)
start_time = time.time()
out = model(image_tensor)
end_time = time.time()
print(end_time-start_time)
# 得到预测结果，并且从大到小排序
_, indices = torch.sort(out, descending=True)
# 返回每个预测值的百分数
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:5]])
print([(class_names[idx], percentage[idx].item()) for idx in indices[0][:1]])
