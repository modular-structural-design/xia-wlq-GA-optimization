import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.patches as patches
case1 = {
    'zone1':{
        'story':1,
        'direction':'x',
        'modular':[3000,3200,4000,4000],
        'width':12600,
        'location':[0,0]
    },
    'zone2': {
        'story': 1,
        'direction': 'y',
        'modular': [3000, 3200, 4000, 4000],
        'width': 12000,
        'location': [0, 5000]
    },
    'zone3': {
        'story': 2,
        'direction': 'x',
        'modular': [3000, 3200, 4000, 4000],
        'width': 12600,
        'location': [0, 0]
    },
    'zone4': {
        'story': 2,
        'direction': 'y',
        'modular': [3000, 3200, 4000, 4000],
        'width': 12600,
        'location': [0, 5000]
    }
}
# data center
case1 ={'zone1': {'story': 1, 'direction': 'y', 'modular': [3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500], 'width': 6000, 'location': [83147, 112422]}, 'zone2': {'story': 1, 'direction': 'y', 'modular': [3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500, 3500], 'width': 6000, 'location': [120547, 112422]}, 'zone3': {'story': 1, 'direction': 'y', 'modular': [3000, 3000, 3500, 3500, 3500, 3500], 'width': 12200, 'location': [92647, 144922]}, 'zone4': {'story': 1, 'direction': 'y', 'modular': [3000, 3000, 3500, 3500, 3500, 3500], 'width': 12200, 'location': [104847, 144922]}}

# Rongsheng
case2 ={'zone1': {'story': 2, 'direction': 'x', 'modular': [4200, 4200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 13175, 'location': [52593, 46107]}, 'zone2': {'story': 3, 'direction': 'x', 'modular': [4200, 4200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 13175, 'location': [52593, 46107]}, 'zone3': {'story': 4, 'direction': 'x', 'modular': [4200, 4200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 13175, 'location': [52593, 46107]}, 'zone4': {'story': 5, 'direction': 'x', 'modular': [4200, 4200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 13175, 'location': [52593, 46107]}, 'zone5': {'story': 6, 'direction': 'x', 'modular': [4200, 4200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 13175, 'location': [52593, 46107]}}

# test_01
case3 = {'zone1': {'story': 1, 'direction': 'x', 'modular': [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], 'width': 12600, 'location': [101897, 27000]}, 'zone2': {'story': 2, 'direction': 'x', 'modular': [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], 'width': 12600, 'location': [101897, 27000]}, 'zone3': {'story': 1, 'direction': 'y', 'modular': [3200, 3000, 3000, 3000, 3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], 'width': 12600, 'location': [149897, 27000]}, 'zone4': {'story': 2, 'direction': 'y', 'modular': [3200, 3000, 3000, 3000, 3000, 3000, 3000, 4000, 4000, 4000, 4000, 4000, 4000, 4000], 'width': 12600, 'location': [149897, 27000]}, 'zone5': {'story': 1, 'direction': 'x', 'modular': [4000, 4000, 4000, 4000, 4000, 4000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 12600, 'location': [101897, 63600]}, 'zone6': {'story': 2, 'direction': 'x', 'modular': [4000, 4000, 4000, 4000, 4000, 4000, 3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000], 'width': 12600, 'location': [101897, 63600]}}

# hotel
case4 = {'zone1': {'story': 1, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone2': {'story': 2, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone3': {'story': 3, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone4': {'story': 4, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone5': {'story': 5, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone6': {'story': 6, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone7': {'story': 7, 'direction': 'x', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 103500]}, 'zone8': {'story': 1, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone9': {'story': 2, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone10': {'story': 3, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone11': {'story': 4, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone12': {'story': 5, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone13': {'story': 6, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone14': {'story': 7, 'direction': 'y', 'modular': [3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11185, 'location': [78385, 103500]}, 'zone15': {'story': 1, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone16': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone17': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone18': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone19': {'story': 5, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone20': {'story': 6, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}, 'zone21': {'story': 7, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3600, 3600, 3600, 3600, 3600, 3600], 'width': 11200, 'location': [27985, 124700]}}

# Lixinhu
case5 = {'zone1': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [176223, 162607]}, 'zone2': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 162607]}, 'zone3': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 162607]}, 'zone4': {'story': 5, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 162607]}, 'zone5': {'story': 6, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [176223, 162607]}, 'zone6': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 193307]}, 'zone7': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 193307]}, 'zone8': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 193307]}, 'zone9': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 4000, 3200, 3200, 4000, 4000, 4000, 3600, 3600, 4000, 4000, 4000], 'width': 12600, 'location': [166623, 229507]}, 'zone10': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 4000, 3200, 3200, 4000, 4000, 4000, 3600, 3600, 4000, 4000, 4000], 'width': 12600, 'location': [166623, 229507]}, 'zone11': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 4000, 3200, 3200, 4000, 4000, 4000, 3600, 3600, 4000, 4000, 4000], 'width': 12600, 'location': [166623, 229507]}, 'zone12': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 260507]}, 'zone13': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 260507]}, 'zone14': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 260507]}, 'zone15': {'story': 5, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [166623, 260507]}, 'zone16': {'story': 6, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 12600, 'location': [176223, 260507]}}


case6 = {'zone1': {'story': 1, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [176223, 175207]}, 'zone2': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 175207]}, 'zone3': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 175207]}, 'zone4': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 175207]}, 'zone5': {'story': 5, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [176223, 175207]}, 'zone6': {'story': 1, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 205907]}, 'zone7': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 205907]}, 'zone8': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 205907]}, 'zone9': {'story': 1, 'direction': 'x', 'modular': [4000, 4000, 3200, 3200, 3200, 3200, 4000, 4000, 4000, 4000, 3200, 4000, 4000], 'width': 9300, 'location': [166623, 229507]}, 'zone10': {'story': 2, 'direction': 'x', 'modular': [4000, 4000, 3200, 3200, 3200, 3200, 4000, 4000, 4000, 4000, 3200, 4000, 4000], 'width': 9300, 'location': [166623, 229507]}, 'zone11': {'story': 3, 'direction': 'x', 'modular': [4000, 4000, 3200, 3200, 3200, 3200, 4000, 4000, 4000, 4000, 3200, 4000, 4000], 'width': 9300, 'location': [166623, 229507]}, 'zone12': {'story': 1, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 260507]}, 'zone13': {'story': 2, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 260507]}, 'zone14': {'story': 3, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 260507]}, 'zone15': {'story': 4, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [166623, 260507]}, 'zone16': {'story': 5, 'direction': 'x', 'modular': [3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200, 3200], 'width': 9300, 'location': [176223, 260507]}}
def get_story_num(case):
    case1 = case
    story_num = []
    for i in range(len(case1)):
        story_num.append(case1[f'zone{i+1}']['story'])
    story_id = list(set(story_num))
    return story_id

def get_modular_type(case):
    case1 = case
    modular_type_all = []
    for j in range(len(case1)):
        for z in range(len(case1[f'zone{j + 1}']['modular'])):
            modular_type_all.append([case1[f'zone{j + 1}']['modular'][z],case1[f'zone{j + 1}']['width']])
    modular = []
    for i in range(len(modular_type_all)):
        if modular_type_all[i] not in modular:
            modular.append(modular_type_all[i])
    colors = ['cyan','yellow','blueviolet', 'green', 'blue', 'burlywood', 'steelblue',
                         'grey', 'wheat', 'beige', 'salmon','purple', 'tan', 'red','grey']
    modular_color = []
    for i in range(len(modular)):
        size = modular[i]
        color = colors[i]
        modular_color.append({'size':size,'color':color})

    return modular_type_all,len(modular),modular_color

def draw_picture(case,modular_color,story_id):
    case1 = case
    modular_lo = []
    for i in story_id:
        fig = plt.figure(figsize=(7, 5), dpi=100)
        ax = fig.add_subplot(111)
        for j in range(len(case1)):

            if case1[f'zone{j + 1}']['story'] == i:

                modular_x = [0]
                for z in range(len(case1[f'zone{j + 1}']['modular']) - 1):
                    modular_x.append(modular_x[z] + case1[f'zone{j + 1}']['modular'][z])
                if case1[f'zone{j + 1}']['direction'] == 'x':
                    rectangles = []
                    # 使用循环添加矩形信息
                    for x_loc in range(len(modular_x)):
                        x = case1[f'zone{j + 1}']['location'][0] + modular_x[x_loc]  # 依次增加x坐标
                        y = case1[f'zone{j + 1}']['location'][1]  # 依次增加y坐标
                        width = case1[f'zone{j + 1}']['modular'][x_loc]
                        height = case1[f'zone{j + 1}']['width']
                        modular_lo.append([x,y])
                        rectangles.append({'x': x, 'y': y, 'width': width, 'height': height})
                    # 创建一个图形

                    # 循环绘制每个矩形
                    for rect_info in rectangles:
                        x = rect_info['x']
                        y = rect_info['y']
                        width = rect_info['width']
                        height = rect_info['height']
                        for mo_color in range(len(modular_color)):
                            if [width, height] == modular_color[mo_color]['size']:
                                color = modular_color[mo_color]['color']
                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor=color)
                        ax.add_patch(rect)

                if case1[f'zone{j + 1}']['direction'] == 'y':
                    rectangles = []
                    # 使用循环添加矩形信息
                    for x_loc in range(len(modular_x)):
                        x = case1[f'zone{j + 1}']['location'][0]  # 依次增加x坐标
                        y = case1[f'zone{j + 1}']['location'][1] + modular_x[x_loc]  # 依次增加y坐标
                        width = case1[f'zone{j + 1}']['width']
                        height = case1[f'zone{j + 1}']['modular'][x_loc]
                        modular_lo.append([x, y])
                        rectangles.append({'x': x, 'y': y, 'width': width, 'height': height})

                    # 循环绘制每个矩形
                    for rect_info in rectangles:
                        x = rect_info['x']
                        y = rect_info['y']
                        width = rect_info['width']
                        height = rect_info['height']
                        for mo_color in range(len(modular_color)):
                            if [height, width] == modular_color[mo_color]['size']:
                                color = modular_color[mo_color]['color']
                        rect = patches.Rectangle((x, y), width, height, linewidth=1, edgecolor='r', facecolor=color)
                        ax.add_patch(rect)
        #寻找最大、最小范围
        x_scope = []
        y_scope = []
        for room_num in range(len(modular_lo)):
            x_scope.append(modular_lo[room_num][0])
            y_scope.append(modular_lo[room_num][1])
        plt.xlim(min(x_scope)-5000, max(x_scope)+10000)
        plt.ylim(min(y_scope)-5000, max(y_scope)+20000)
        plt.axis('equal')
        APIPath = os.path.join(os.getcwd(), f'draw')

        SpecifyPath = True
        if not os.path.exists(APIPath):
            try:
                os.makedirs(APIPath)
            except OSError:
                pass
        path1 = os.path.join(APIPath, f'story{i}')
        plt.savefig(path1, dpi=300)
        plt.close()
        plt.show()

def draw_case(case_num):
    story_id = get_story_num(case_num)
    # 获得楼层编号
    modular_type_all, modular_num, modular_color = get_modular_type(case_num)

    draw_picture(case_num, modular_color, story_id)


# case2 =case6
# #获得楼层编号
# story_id = get_story_num(case2)
# #获得楼层编号
# modular_type_all,modular_num,modular_color = get_modular_type(case2)
#
# draw_picture(case2,modular_color,story_id)

draw_case(case2)