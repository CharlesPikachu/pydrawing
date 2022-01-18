# 安装Pydrawing


## 环境配置
- 操作系统: Linux or macOS or Windows
- Python版本: Python3.6+
- ffmpeg: 若输入视频中含有音频, 需要借助[ffmpeg](https://ffmpeg.org/)解码, 因此需要保证电脑中存在ffmpeg并在环境变量中。
- Pytorch: 若需要使用CartoonGan等算法, 需要安装Pytorch>=1.0.0和配置对应的环境, 详见[官方文档](https://pytorch.org/get-started/locally/)。


## PIP安装(推荐)
在终端运行如下命令即可(请保证python在环境变量中):
```sh
pip install pydrawing --upgrade
```


## 源代码安装

#### 在线安装
运行如下命令即可在线安装:
```sh
pip install git+https://github.com/CharlesPikachu/pydrawing.git@master
```

#### 离线安装
利用如下命令下载pydrawing源代码到本地:
```sh
git clone https://github.com/CharlesPikachu/pydrawing.git
```
接着, 切到pydrawing目录下:
```sh
cd pydrawing
```
最后运行如下命令进行安装:
```sh
python setup.py install
```