'''
Function:
    利用遗传算法画画
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import copy
import random
import numpy as np
from ...utils import checkdir
from PIL import Image, ImageDraw
from ..base import BaseBeautifier


'''多边形'''
class Polygon():
    def __init__(self, num_points=3, size=50, shift_range=50, point_range=50, color_range=50, target_image=None, **kwargs):
        # set attrs
        self.size = size
        self.num_points = num_points
        self.target_image = target_image
        self.shift_range = shift_range
        self.point_range = point_range
        self.color_range = color_range
        # 点
        image_width, image_height = target_image.size[:2]
        x, y = random.randint(0, int(image_width)), random.randint(0, int(image_height))
        self.points = []
        for _ in range(self.num_points): 
            self.points.append(((y + random.randint(-size, size), x + random.randint(-size, size))))
        # 颜色
        point = random.choice(self.points)
        r, g, b = np.asarray(target_image)[min(point[0], image_height-1), min(point[1], image_width-1)]
        self.color = (int(r), int(g), int(b), random.randint(0, 256))
    '''变异'''
    def mutate(self):
        mutations = ['shift', 'point', 'color', 'reset']
        mutation_type = random.choice(mutations)
        # 整体偏移
        if mutation_type == 'shift':
            x_shift = int(random.randint(-self.shift_range, self.shift_range) * random.random())
            y_shift = int(random.randint(-self.shift_range, self.shift_range) * random.random())
            self.points = [(x + x_shift, y + y_shift) for x, y in self.points]
        # 随机改变一个点
        elif mutation_type == 'point':
            index = random.choice(list(range(len(self.points))))
            self.points[index] = (
                self.points[index][0] + int(random.randint(-self.point_range, self.point_range) * random.random()),
                self.points[index][1] + int(random.randint(-self.point_range, self.point_range) * random.random()),
            )
        # 随机改变颜色
        elif mutation_type == 'color':
            self.color = tuple(c + int(random.randint(-self.color_range, self.color_range) * random.random()) for c in self.color)
            self.color = tuple(min(max(c, 0), 255) for c in self.color)
        # 重置
        else:
            new_polygon = Polygon(
                num_points=max(self.num_points + random.choice([-1, 0, 1]), 3), 
                size=self.size, 
                shift_range=self.shift_range, 
                point_range=self.point_range, 
                color_range=self.color_range, 
                target_image=self.target_image
            )
            self.points = new_polygon.points
            self.color = new_polygon.color


'''利用遗传算法画画'''
class GeneticFittingPolygonBeautifier(BaseBeautifier):
    def __init__(self, init_cfg=None, cache_dir='cache', save_cache=True, **kwargs):
        super(GeneticFittingPolygonBeautifier, self).__init__(**kwargs)
        if init_cfg is None:
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
        self.init_cfg = init_cfg
        self.cache_dir = cache_dir
        self.save_cache = save_cache
    '''迭代图片'''
    def iterimage(self, image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # 初始化
        populations = []
        for _ in range(self.init_cfg['num_populations']):
            population = []
            for _ in range(self.init_cfg['init_num_polygons']):
                population.append(Polygon(
                    target_image=image, 
                    num_points=random.choice(self.init_cfg['num_points_list']),
                    **self.init_cfg['polygon_cfg']
                ))
            populations.append(population)
        # 迭代
        mutation_rate = self.init_cfg['mutation_rate']
        for g in range(1, int(self.init_cfg['num_generations']+1)):
            fitnesses = []
            for idx, population in enumerate(copy.deepcopy(populations)):
                fitness_ori = self.calcfitnesses([population], image)[0]
                fitness = 0
                while fitness_ori > fitness:
                    population_new = population + [Polygon(target_image=image, num_points=random.choice(self.init_cfg['num_points_list']), **self.init_cfg['polygon_cfg'])]
                    fitness = self.calcfitnesses([population_new], image)[0]
                populations[idx] = population_new
                fitnesses.append(fitness)
            if g % self.init_cfg['print_interval'] == 0:
                if self.save_cache:
                    population = populations[np.argmax(fitnesses)]
                    output_image = self.draw(population, image)
                    checkdir(self.cache_dir)
                    output_image.save(os.path.join(self.cache_dir, f'cache_g{g}.png'))
                self.logger_handle.info(f'Generation: {g}, FITNESS: {max(fitnesses)}')
            num_populations = len(populations)
            # --自然选择
            populations = self.select(image, fitnesses, populations)
            # --交叉
            populations = self.crossover(populations, num_populations)
            # --变异
            populations = self.mutate(image, populations, mutation_rate)
        # 选择最优解返回
        population = populations[np.argmax(fitnesses)]
        output_image = self.draw(population, image)
        return cv2.cvtColor(np.asarray(output_image), cv2.COLOR_RGB2BGR)
    '''自然选择'''
    def select(self, image, fitnesses, populations):
        sorted_idx = np.argsort(fitnesses)[::-1]
        selected_populations = []
        selection_rate = self.init_cfg['selection_rate']
        for idx in range(int(len(populations) * selection_rate)):
            selected_idx = int(sorted_idx[idx])
            selected_populations.append(populations[selected_idx])
        return selected_populations
    '''交叉'''
    def crossover(self, populations, num_populations):
        indices = list(range(len(populations)))
        while len(populations) < num_populations:
            idx1 = random.choice(indices)
            idx2 = random.choice(indices)
            population1 = copy.deepcopy(populations[idx1])
            population2 = copy.deepcopy(populations[idx2])
            for polygon_idx in range(len(population1)):
                if self.init_cfg['crossover_rate'] > random.random():
                    population1[polygon_idx] = population2[polygon_idx]
            populations.append(population1)
        return populations
    '''变异'''
    def mutate(self, target_image, populations, mutation_rate):
        populations_new = copy.deepcopy(populations)
        for idx, population in enumerate(populations):
            fitness_ori = self.calcfitnesses([population], target_image)[0]
            for polygon in population:
                if mutation_rate > random.random():
                    polygon.mutate()
            fitness = self.calcfitnesses([population], target_image)[0]
            if fitness > fitness_ori: populations_new[idx] = population
        return populations_new
    '''计算适应度'''
    def calcfitnesses(self, populations, target_image):
        fitnesses = []
        for idx in range(len(populations)):
            image = self.draw(populations[idx], target_image)
            fitnesses.append(self.calcsimilarity(image, target_image))
        return fitnesses
    '''画图'''
    def draw(self, population, target_image):
        image = Image.new('RGB', target_image.size, "#FFFFFF")
        for polygon in population:
            item = Image.new('RGB', target_image.size, "#000000")
            mask = Image.new('L', target_image.size, 255)
            draw = ImageDraw.Draw(item)
            draw.polygon(polygon.points, polygon.color)
            draw = ImageDraw.Draw(mask)
            draw.polygon(polygon.points, fill=128)
            image = Image.composite(image, item, mask)
        return image
    '''计算两幅图之间的相似度'''
    def calcsimilarity(self, image, target_image):
        image = np.asarray(image) / 255.
        target_image = np.asarray(target_image) / 255.
        similarity = 1 - abs(image - target_image).mean()
        return similarity