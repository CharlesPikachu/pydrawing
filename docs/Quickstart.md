# 快速开始


## 已经支持的算法

#### 图像卡通化

#### 铅笔素描画

#### 卡通GAN

#### 快速风格迁移

#### 抖音特效

#### 视频转字符画

#### 拼马赛克图片


## 随机运行一个小程序
写如下代码，保存并运行即可：
```python
import random
from pydrawing import pydrawing

filepath = 'asserts/input.jpg'
config = {
    "savedir": "outputs",
    "savename": "output"
}
drawing_client = pydrawing()
drawing_client.execute(filepath, random.choice(drawing_client.getallsupports()))
```