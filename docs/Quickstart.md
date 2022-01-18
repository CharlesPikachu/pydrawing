# 快速开始


## 已经支持的算法

### 图像卡通化
#### 相关论文
暂无
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/efwNQl0JVJt6_x_evdL41A)
#### 调用示例
```python
from pydrawing import pydrawing

config = {'mode': ['rgb', 'hsv'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoonise', config=config)
```

### 铅笔素描画
#### 相关论文
[Paper](https://jiaya.me/archive/projects/pencilsketch/npar12_pencil.pdf)
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/K_2lGGlLKHIIm4iSg0xCUw)
#### 调用示例
```python
from pydrawing import pydrawing

config = {'mode': ['gray', 'color'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'pencildrawing', config=config)
```

### 卡通GAN
#### 相关论文
[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf)
#### 公众号文章介绍
[Introduction]()
#### 调用示例
```python
from pydrawing import pydrawing

config = {'style': ['Hayao', 'Hosoda', 'Paprika', 'Shinkai'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoongan', config=config)
```

### 快速风格迁移
#### 相关论文
[Paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/Ed-1fWOIhI52G-Ugrv7n9Q)
#### 调用示例
```python
from pydrawing import pydrawing

config = {'style': ['starrynight', 'cuphead', 'mosaic​'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'fastneuralstyletransfer', config=config)
```

### 抖音特效
#### 相关论文
暂无
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/RRnrO2H84pvtUdDsAYD9Qg)
#### 调用示例
```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'douyineffect')
```

### 视频转字符画
#### 相关论文
暂无
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/yaNQJyeUeisOenEeoVsgDg)
#### 调用示例
```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'characterize')
```

### 拼马赛克图片
#### 相关论文
暂无
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/BG1VW3jx0LUazhhifBapVw)
#### 调用示例
```python
from pydrawing import pydrawing
​
config = {'src_images_dir': 'images', 'block_size': 15}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'photomosaic', config=config)
```

### 信号故障特效
#### 相关论文
暂无
#### 公众号文章介绍
[Introduction](https://mp.weixin.qq.com/s/Yv0uPLsTGwVnj_PKqYCmAw)
#### 调用示例
```python
from pydrawing import pydrawing

filepath = 'input.mp4'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'glitch')
```


## 随机运行一个小程序
写如下代码，保存并运行即可：
```python
import random
from pydrawing import pydrawing

filepath = 'asserts/dog.jpg'
config = {
    "savedir": "outputs",
    "savename": "output"
}
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, random.choice(drawing_client.getallsupports()), config=config)
```