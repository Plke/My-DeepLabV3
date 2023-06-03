# My-DeepLabV3

————VOC
     | ————data
     |       |————imgs
     |       |————labels
     |       |————list
     |       
     |————log
     |
     |————model
     |
     |————dataset.py 
     |
     |————get_segmentation.ipynb
     |
     |————helpfun.py
     |
     |————requirements.txt
     |
     |————train.ipynb
     |
     |————transforms.py
     |
     |————
     |
     |————
data中存放数据 imgs为原图，labels为分割标签mask，list中为划分训练测试集的图片id

log存放训练相关参数以及结果

model存放训练得到的模型

dataset.py 为自定义的dataset用于加载数据

get_segmentation.ipynb 为观察图像分割效果，得到miou，注意，测试时是每张图像保持原图大小进行分割的

helpfun.py 包含了一些辅助函数

requirements.txt 包含了python需要有的库，注意，没有标注版本，可能会有冲突或报错

train.ipynb 为修改训练参数以及训练的地方

transforms.py 定义了一些用于图像分割的数据增强方法，
"# My-DeepLabV3" 
