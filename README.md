# My-DeepLabV3

## 项目目录结构

My-DeepLabV3
├── data
│   ├── imgs
│   ├── labels
│   └── list
├── log
├── model
├── dataset.py
├── get_segmentation.ipynb
├── helpfun.py
├── requirements.txt
├── train.ipynb
└── transforms.py
     

### 详细说明

- `data` 目录
  - `imgs`：存放原始图像文件。
  - `labels`：存放图像分割的标签（mask）。
  - `list`：包含用于划分训练集和测试集的图片ID列表。

- `log` 目录
  - 用于存储训练过程中的日志和结果。

- `model` 目录
  - 存放训练完成后得到的模型文件。

- `dataset.py`
  - 定义了自定义的数据集加载类。

- `get_segmentation.ipynb`
  - Jupyter Notebook，用于观察图像分割效果并计算mIoU。

- `helpfun.py`
  - 包含一些辅助函数。

- `requirements.txt`
  - 列出了Python环境中所需的库。

- `train.ipynb`
  - Jupyter Notebook，用于修改训练参数和执行模型训练。

- `transforms.py`
  - 定义了一些用于图像分割的数据增强方法。
