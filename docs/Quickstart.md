# 快速开始


## 已经支持的算法

#### 图像卡通化

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/efwNQl0JVJt6_x_evdL41A)

**3.调用示例**

```python
from pydrawing import pydrawing

config = {'mode': ['rgb', 'hsv'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoonise', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- mode: 卡通化时所用的颜色空间, 支持"rgb"和"hsv"模式, 默认值为"rgb"。

#### 人脸卡通化

**1.相关论文**

[Paper](https://arxiv.org/pdf/1907.10830.pdf)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/L0z1ZO1Qztk0EF1KAMfmbA)

**3.调用示例**

```python
from pydrawing import pydrawing

config = {'use_face_segmentor': False}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoonizeface', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- use_cuda: 模型是否使用cuda加速, 默认值为"False";
- use_face_segmentor: 是否使用人脸分割器进一步去除人脸背景, 默认值为"False"。

#### 铅笔素描画

**1.相关论文**

[Paper](https://jiaya.me/archive/projects/pencilsketch/npar12_pencil.pdf)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/K_2lGGlLKHIIm4iSg0xCUw)

**3.调用示例**

```python
from pydrawing import pydrawing

config = {'mode': ['gray', 'color'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'pencildrawing', config=config)
```

**4.config选项**

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

**1.相关论文**

[Paper](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2205.pdf)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/18fUOO5fH1PVUzTMNNCWwQ)

**3.调用示例**
```python
from pydrawing import pydrawing

config = {'style': ['Hayao', 'Hosoda', 'Paprika', 'Shinkai'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'cartoongan', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- style: 卡通画的风格类型, 支持"Hayao", "Hosoda", "Paprika"和"Shinkai", 默认值为"Hosoda";
- use_cuda: 模型是否使用cuda加速, 默认值为"True"。

#### 快速风格迁移

**1.相关论文**

[Paper](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/Ed-1fWOIhI52G-Ugrv7n9Q)

**3.调用示例**

```python
from pydrawing import pydrawing

config = {'style': ['starrynight', 'cuphead', 'mosaic​'][0]}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'fastneuralstyletransfer', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- style: 迁移的画的风格类型, 支持"starrynight", "cuphead"和"mosaic", 默认值为"starrynight";
- use_cuda: 模型是否使用cuda加速, 默认值为"True"。

#### 抖音特效

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/RRnrO2H84pvtUdDsAYD9Qg)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'douyineffect')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False"。

#### 视频转字符画

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/yaNQJyeUeisOenEeoVsgDg)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'characterize')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False"。

#### 拼马赛克图片

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/BG1VW3jx0LUazhhifBapVw)

**3.调用示例**

```python
from pydrawing import pydrawing
​
config = {'src_images_dir': 'images', 'block_size': 15}
filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'photomosaic', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- block_size: 马赛克block大小, 默认值为"15";
- src_images_dir: 使用的图片路径, 请保证该文件夹中存在大量色彩各异的图片以实现较好的拼图效果。

#### 信号故障特效

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/Yv0uPLsTGwVnj_PKqYCmAw)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.mp4'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'glitch')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- header_size: 文件头部大小, 一般不需要改, 默认值为"200";
- intensity: 随机扰动相关的参数, 默认值为"0.1";
- block_size: 一次读取文件的大小, 默认值为"100"。

#### 贝塞尔曲线画画

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/SWpaTPw9tOLs5h1EgP30Vw)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'beziercurve')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- num_samples: 采样点, 默认值为"15";
- width: 坐标变换宽度, 默认值为"600";
- height: 坐标变换高度, 默认值为"600";
- num_colors: 使用的颜色数量, 默认值为"32"。

#### 遗传算法拟合图像-圆形

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/L0z1ZO1Qztk0EF1KAMfmbA)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'geneticfittingcircle')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- cache_dir: 中间结果保存的文件夹, 默认值为"cache";
- save_cache: 是否保存中间结果, 默认值为"True";
- init_cfg: 算法初始化参数, 默认值为如下:
```python
init_cfg = {
	'num_populations': 10,
	'init_num_circles': 1,
	'num_generations': 1e5,
	'print_interval': 1,
	'mutation_rate': 0.1,
	'selection_rate': 0.5,
	'crossover_rate': 0.5,
	'circle_cfg': {'radius_range': 50, 'radius_shift_range': 50, 'center_shift_range': 50, 'color_shift_range': 50},
}
```

#### 遗传算法拟合图像-多边形

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/L0z1ZO1Qztk0EF1KAMfmbA)

**3.调用示例**

```python
from pydrawing import pydrawing

filepath = 'input.jpg'
drawing_client = pydrawing.pydrawing()
drawing_client.execute(filepath, 'geneticfittingpolygon')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- cache_dir: 中间结果保存的文件夹, 默认值为"cache";
- save_cache: 是否保存中间结果, 默认值为"True";
- init_cfg: 算法初始化参数, 默认值为如下:
```python
init_cfg = {
	'num_populations': 10,
	'num_points_list': list(range(3, 40)),
	'init_num_polygons': 1,
	'num_generations': 1e5,
	'print_interval': 1,
	'mutation_rate': 0.1,
	'selection_rate': 0.5,
	'crossover_rate': 0.5,
	'polygon_cfg': {'size': 50, 'shift_range': 50, 'point_range': 50, 'color_range': 50},
}
```

#### 照片怀旧风格

**1.相关论文**

暂无

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/yRCt69u_gzPI85-vOrb_sQ)

**3.调用示例**

```python
from pydrawing import pydrawing
​
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'nostalgicstyle')
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False"。

#### 手写笔记处理

**1.相关论文**

[Paper](https://mzucker.github.io/2016/09/20/noteshrink.html)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/yRCt69u_gzPI85-vOrb_sQ)

**3.调用示例**

```python
from pydrawing import pydrawing
​
config = {
    'sat_threshold': 0.20, 
    'value_threshold': 0.25, 
    'num_colors': 8, 
    'sample_fraction': 0.05,
    'white_bg': False,
    'saturate': True,
}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'noteprocessor', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- sat_threshold: 背景饱和度阈值, 默认值为"0.2";
- value_threshold: 背景的阈值, 默认值为"0.25";
- num_colors: 输出颜色的数量, 默认值为"8";
- sample_fraction: 采样的像素占比, 默认值为"0.05";
- white_bg: 使背景为白色, 默认值为"False";
- saturate: 使颜色不饱和, 默认值为"True"。

#### 照片油画化

**1.相关论文**

[Paper](https://github.com/cyshih73/Faster-OilPainting/blob/master/Report.pdf)

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/yRCt69u_gzPI85-vOrb_sQ)

**3.调用示例**

```python
from pydrawing import pydrawing
​
config = {
    'edge_operator': 'sobel', 
    'palette': 0, 
    'brush_width': 5, 
}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'oilpainting​', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- brush_width: 画笔大小, 默认值为"5";
- palette: 调色板颜色, 默认为"0", 代表使用原图的实际颜色;
- edge_operator: 使用的边缘检测算子, 支持"sobel", "prewitt", "scharr"和"roberts", 默认值为"sobel"。

#### 简单的照片矫正

**1.相关论文**

暂无。

**2.公众号文章介绍**

[Introduction](https://mp.weixin.qq.com/s/yRCt69u_gzPI85-vOrb_sQ)

**3.调用示例**

```python
from pydrawing import pydrawing
​
config = {
    'epsilon_factor': 0.08, 
    'canny_boundaries': [100, 200], 
    'use_preprocess': False, 
}
filepath = 'input.jpg'
drawing_client = pydrawing()
drawing_client.execute(filepath, 'photocorrection', config=config)
```

**4.config选项**

- savename: 保存结果时用的文件名, 默认值为"output";
- savedir: 保存结果时用的文件夹, 默认值为"outputs";
- merge_audio: 处理视频时, 是否把原视频中的音频合成到生成的视频中, 默认值为"False";
- epsilon_factor: 多边形估计时的超参数, 默认为"0.08";
- canny_boundaries: canny边缘检测算子的两个边界值, 默认为"[100, 200]";
- use_preprocess: 是否在边缘检测前对图像进行预处理, 默认值为"False"。


## 随机运行一个小程序

写如下代码，保存并运行即可:

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

部分测试效果如下:

<div align="center">
  <img src="https://github.com/CharlesPikachu/pydrawing/raw/master/docs/screenshot_characterize.gif" width="600"/>
</div>
<br />
<div align="center">
  <img src="https://github.com/CharlesPikachu/pydrawing/raw/master/docs/screenshot_fastneuralstyletransfer.gif" width="600"/>
</div>
<br />
<div align="center">
  <img src="https://github.com/CharlesPikachu/pydrawing/raw/master/docs/screenshot_photomosaic.png" width="600"/>
</div>
<br />
<div align="center">
  <img src="https://github.com/CharlesPikachu/pydrawing/raw/master/docs/screeshot_noteprocessor.png" width="600"/>
</div>
<br />