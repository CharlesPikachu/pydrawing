'''
Function:
    用Python美化你的照片或视频
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import warnings
if __name__ == '__main__':
    from modules import *
else:
    from .modules import *
warnings.filterwarnings('ignore')


'''用Python美化你的照片或视频'''
class pydrawing():
    def __init__(self, **kwargs):
        for key, value in kwargs.items(): setattr(self, key, value)
        self.supported_beautifiers = self.initializebeautifiers()
        self.logger_handle = Logger(kwargs.get('logfilepath', 'pydrawing.log'))
    '''执行对应的算法'''
    def execute(self, filepath, beautifier_type=None, config={}):
        assert beautifier_type in self.supported_beautifiers, 'unsupport beautifier_type %s' % beautifier_type
        if 'savedir' not in config: config['savedir'] = 'outputs'
        if 'savename' not in config: config['savename'] = 'output'
        if 'logger_handle' not in config: config['logger_handle'] = self.logger_handle
        beautifier = self.supported_beautifiers[beautifier_type](**config)
        beautifier.process(filepath)
    '''获得所有支持的美化器'''
    def getallsupports(self):
        return list(self.supported_beautifiers.keys())
    '''初始化美化器'''
    def initializebeautifiers(self):
        supported_beautifiers = {
            'cartoonise': CartooniseBeautifier,
            'pencildrawing': PencilDrawingBeautifier,
        }
        return supported_beautifiers


'''run'''
if __name__ == '__main__':
    import random
    drawing_client = pydrawing()
    drawing_client.execute('asserts/input.jpg', random.choice(drawing_client.getallsupports()))