# 快速开始


## 已经支持的算法

#### 图像卡通化

1.相关论文

暂无

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/efwNQl0JVJt6_x_evdL41A)

3.调用示例

```python
from pydrawing import pydrawing

config = {'mode': ['rgb', 'hsv'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoonise', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- mode: 卡通化时所用的颜色空间, 支持"rgb"和"hsv"模式, 默认值为"rgb"。

#### 人脸卡通化

1.相关论文

[Paper](https://arxiv.org/pdf/1907.10830.pdf)

2.公众号文章介绍

[Introduction]()

3.调用示例

```python
from pydrawing import pydrawing

config = {'use_face_segmentor': False}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoonizeface', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- use_cuda: 模型是否使用cuda加速, 默认值为"False";
- use_face_segmentor: 是否使用人脸分割器进一步去除人脸背景, 默认值为"False"。

#### 铅笔素描画

1.相关论文

[Paper](https://jiaya.me/archive/projects/pencilsketch/npar12_pencil.pdf)

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/K_2lGGlLKHIIm4iSg0xCUw)

3.调用示例

```python
from pydrawing import pydrawing

config = {'mode': ['gray', 'color'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'pencildrawing', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- mode: 生成的图片是灰色图还是彩色图, 支持"gray"和"color", 默认值为"gray";
- kernel_size_scale: 铅笔笔画相关参数, 默认值为"1/40";
- stroke_width: 铅笔笔画相关参数, 默认值为"1";
- color_depth: 铅笔色调相关参数, 默认值为"1";
- weights_color: 铅笔色调相关参数, 默认值为"[62, 30, 5]";
- weights_gray: 铅笔色调相关参数, 默认值为"[76, 22, 2]";
- texture_path: 纹理图片路径, 默认使用库里提供的"default.jpg"文件。

#### 卡通GAN

1.相关论文

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf)

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/18fUOO5fH1PVUzTMNNCWwQ)

3.调用示例
```python
from pydrawing import pydrawing

config = {'style': ['Hayao', 'Hosoda', 'Paprika', 'Shinkai'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoongan', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- style: 卡通画的风格类型, 支持"Hayao", "Hosoda", "Paprika"和"Shinkai", 默认值为"Hosoda";
- use_cuda: 模型是否使用cuda加速, 默认值为"True"。

#### 快速风格迁移

1.相关论文

[Paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/Ed-1fWOIhI52G-Ugrv7n9Q)

3.调用示例

```python
from pydrawing import pydrawing

config = {'style': ['starrynight', 'cuphead', 'mosaic​'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'fastneuralstyletransfer', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- style: 迁移的画的风格类型, 支持"starrynight", "cuphead"和"mosaic", 默认值为"starrynight";
- use_cuda: 模型是否使用cuda加速, 默认值为"True"。

#### 抖音特效

1.相关论文

暂无

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/RRnrO2H84pvtUdDsAYD9Qg)

3.调用示例

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'douyineffect')
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False"。

#### 视频转字符画

1.相关论文

暂无

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/yaNQJyeUeisOenEeoVsgDg)

3.调用示例

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'characterize')
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False"。

#### 拼马赛克图片

1.相关论文

暂无

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/BG1VW3jx0LUazhhifBapVw)

3.调用示例

```python
from pydrawing import pydrawing
​
config = {'src_images_dir': 'images', 'block_size': 15}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'photomosaic', config=config)
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- block_size: 马赛克block大小, 默认值为"15";
- src_images_dir: 使用的图片路径, 请保证该文件夹中存在大量色彩各异的图片以实现较好的拼图效果。

#### 信号故障特效

1.相关论文

暂无

2.公众号文章介绍

[Introduction](https://mp.weixin.qq.com/s/Yv0uPLsTGwVnj_PKqYCmAw)

3.调用示例

```python
from pydrawing import pydrawing

filepath = 'input.mp4'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'glitch')
```

4.config选项

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- header_size: 文件头部大小, 一般不需要改, 默认值为"200";
- intensity: 随机扰动相关的参数, 默认值为"0.1";
- block_size: 一次读取文件的大小, 默认值为"100"。


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