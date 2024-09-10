import copy
import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA
import os
import MULIT_FEM as MF
import FEM_Index_calculation as FC
import random
import FEM_parser as FEA
import gc
import matplotlib
import pyvista as pv
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches



plt.rc('font', family='Times New Roman')


def draw_fitness(path):
    with open(path, 'r') as file:
        ga_data = json.load(file)
    fitness_dict = {key: item["fitness"] for key, item in ga_data.items()}
    fitness = list(fitness_dict.values())

    fig2 = plt.figure(num=1, figsize=(23, 30))
    ax2 = fig2.add_subplot(111)
    ax2.tick_params(labelsize=40)
    ax2.set_xlabel("Iteration", fontsize=50)  # 添加x轴坐_标标签，后面看来没必要会删除它，这里只是为了演示一下。
    ax2.set_ylabel('Fitness', fontsize=50)  # 添加y轴标签，设置字体大小为16，这里也可以设字体样式与颜色
    ax2.spines['bottom'].set_linewidth(3);  ###设置底部坐标轴的粗细
    ax2.spines['left'].set_linewidth(3)
    ax2.spines['right'].set_color('none')
    ax2.spines['top'].set_color('none')

    bbb = np.arange(0, len(fitness))
    ccc = fitness
    # ax2.plot(bbb, ccc, linewidth=3, color=color[i])
    ax2.plot(bbb, ccc, linewidth=3, color='blue')
    # ax2.plot(bbb, ccc, linewidth=3, color=co)
    # legend_handles = [plt.Line2D([0], [0], color='red', lw=2,linewidth=20),
    #                   plt.Line2D([0], [0], color='black', lw=2,linewidth=20),
    #                   plt.Line2D([0], [0], color='blue', lw=2, linewidth=20)
    #                   ]
    # legend_labels = ['200*200', '50*50', '100*100']
    # plt.legend(fontsize=30)
    ax2.tick_params(labelsize=30, which='major', length=10, width=1)
    ax2.tick_params(axis='both', direction='in')
    ax2.set(xlim=(0, 105), ylim=(0, 450),
            xticks=np.arange(20, 105, 20),
            yticks=np.arange(0, 450, 100))
    # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    plt.show()


def select_2(pop, fitness):  # nature selection wrt pop's fitness

    fit_ini = copy.deepcopy(fitness)
    luyi = copy.deepcopy(fitness)
    luyi.sort(reverse=True)
    sort_num = []
    for i in range(len(fit_ini)):
        sort_num.append(luyi.index(fit_ini[i]))
    # print(sort_num)
    # print(f'{len(sort_num)}_{len(pop)}')
    for i in range(len(sort_num)):
        if sort_num[i] == 0:
            sort_num[i] += 0.01
    # pop_last.append(pop)

    # for i in range(len(list_new)):
    #     list_new[i] = m.e ** (list_new[i] * 1.5)
    idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True,
                           p=np.array(sort_num) / (sum(sort_num)))
    pop2 = np.zeros((len(pop), len(pop[0])))
    for i in range(len(pop2)):
        pop2[i] = pop[int(idx[i])]
    return pop2


def crossover_and_mutation(pop2, CROSSOVER_RATE, MUTATION_RATE, section_info):
    pop = pop2

    new_pop = np.zeros((len(pop), len(pop[0])))
    for i in range(len(pop)):
        father = pop[i]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(len(pop2))]
            cross_points1 = np.random.randint(low=0, high=len(pop[0]))
            cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            while cross_points2 == cross_points1:
                cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            exchan = []
            exchan.append(cross_points2)
            exchan.append(cross_points1)
            for j in range(min(exchan), max(exchan)):
                child[j] = mother[j]
        mutation(child, MUTATION_RATE, section_info)
        new_pop[i] = child

    return new_pop


def mutation(child, MUTATION_RATE, section_info):
    for i in range(len(child)):
        if np.random.rand() < MUTATION_RATE:
            child[i] = random.randint(0, len(section_info) - 1)


def calute(File_Path, ModelPath, mySapObject, SapModel, pop2, mic_FEM_data, FEM_sematics, modular_num, FEA_info2, u):
    # 染色体解码
    merged_list = [pop2[i:i + 3] for i in range(0, len(pop2), 3)]
    modular_FEM = {}

    # 用 for 循环生成字典
    for i in range(0, modular_num):  # 假设你需要生成键 1 到 2
        modular_FEM[i + 1] = {"sections": merged_list[i]}

    FEA.parsing_to_sap2000_mulit(FEA_info2, FEM_sematics, modular_FEM, File_Path, SapModel, mySapObject, ModelPath)

    FC.output_index(modular_FEM, File_Path, File_Path, mic_FEM_data)
    with open(os.path.join(File_Path, 'max_values.json'), 'r') as file:
        index_data = json.load(file)
    value_dict = {key: item["value"] for key, item in index_data.items()}
    values_list = list(value_dict.values())
    value_id = []
    for j in range(len(values_list) - 1):
        if values_list[j] < 0:
            value_id.append(0)
        else:
            value_id.append(values_list[j])
    weight = values_list[-1]
    fit = weight + u * (sum(value_id))
    return fit


def GA_structure(SapModel_name, mySapObject_name, ModelPath_name, File_Path,sorted_elements):
    all_min_fit = []
    all_min_weight = []
    all_min_pop = []
    pop2 = MF.generate_chromosome(modular_num, section_info, pop_size)
    for run_time in range(n_iteration):

        pop_data = {}
        for i in range(len(pop2)):
            pop_data[f'chro{i}'] = {
                'code': pop2[i]
            }

        pop_data = MF.thread_sap(File_Path, ModelPath_name, mySapObject_name, SapModel_name, num_thread, pop2,
                                 mic_FEM_data,
                                 FEM_sematics, modular_num, FEA_info2, pop_data,sorted_elements)
        fitness_dict = {key: item["fitness"] for key, item in pop_data.items()}
        fitness = list(fitness_dict.values())

        min_fitness = min(fitness)
        min_index = fitness.index(min(fitness))
        min_chro = pop_data[f'chro{min_index}']['code']
        all_min_fit.append(min_fitness)
        all_min_weight.append(pop_data[f'chro{min_index}']['weight'])
        all_min_pop.append(pop_data[f'chro{min_index}']['code'])

        pop2 = np.array(pop2)
        pop2 = select_2(pop2, fitness)
        pop2 = crossover_and_mutation(pop2, CROSSOVER_RATE, MUTATION_RATE, section_info)
        pop2.tolist()
        pop2 = [[int(num) for num in sublist] for sublist in pop2]
        if run_time % 10 == 0:
            print(f'运行{run_time}次')
        if min_fitness <= calute(File_Path[0], ModelPath_name[0], mySapObject_name[0], SapModel_name[0], pop2[0],
                                 mic_FEM_data, FEM_sematics, modular_num, FEA_info2, 10000):
            pop2[0] = min_chro
    ga_data = {}

    for i in range(len(all_min_fit)):
        ga_data[f'chro{i}'] = {
            'code': all_min_pop[i],
            'weight': all_min_weight[i],
            'fitness': all_min_fit[i]
        }
    os.path.join(os.getcwd(), "FEM_sap2000")
    with open(os.path.join(os.getcwd(), f'Structural_GA_data\GA_data_case{case_number}.json'), 'w') as json_file:
        json.dump(ga_data, json_file, indent=4)

    for i in range(len(mySapObject_name)):
        ret = mySapObject_name[i].ApplicationExit(False)
        SapModel_name[i] = None
        mySapObject_name[i] = None
    return all_min_fit, all_min_weight, all_min_pop

def draw_outsapce(out):
    # 给定的字典数据
    data = out

    # 创建图形和坐标轴
    fig, ax = plt.subplots()

    # 遍历字典，绘制每个矩形
    for key, points in data.items():
        # 取出矩形的四个角点
        rect_points = points

        # 矩形左下角的坐标和宽高
        x_min = min(p[0] for p in rect_points)  # 找到最小的 x 坐标
        y_min = min(p[1] for p in rect_points)  # 找到最小的 y 坐标
        width = max(p[0] for p in rect_points) - x_min  # 矩形的宽度
        height = max(p[1] for p in rect_points) - y_min  # 矩形的高度

        # 创建矩形对象
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')

        # 将矩形添加到轴上
        ax.add_patch(rect)

    # 设置坐标轴范围
    ax.set_xlim(0, 70000)
    ax.set_ylim(0, 40000)
    ax.set_aspect('equal')

    # 显示图形
    plt.show()
def draw_point(point):
    # 给定的字典数据，表示三维坐标
    coordinates = point

    # 提取坐标点并转换为 NumPy 数组
    points = np.array(list(coordinates.values()))

    # 创建 pyvista 的 PolyData 对象来存储点
    point_cloud = pv.PolyData(points)

    # 创建一个绘图对象
    plotter = pv.Plotter()

    # 添加点到绘图对象
    plotter.add_mesh(point_cloud, color='blue', point_size=10, render_points_as_spheres=True)

    # 设置绘图范围
    plotter.set_background('white')
    plotter.show_grid()

    # 显示三维点图
    plotter.show()


# region Reading data
with open('config.json', 'r') as f:
    file_data = json.load(f)

with open(os.path.join(file_data["file_paths"]["BuildingData"], file_data["file_names"]["building_data"]), 'r') as f:
    building_data = json.load(f)

out_space_num = len(building_data['outer_space_config'])
out_space_info = building_data["outer_space_per_building"]
out_space_cfg = building_data["outer_space_config"]
inner_space_info = building_data["outer_space_has_inner_space"]
inner_space_cfg = building_data["inner_space_config"]
out_space_relationship = building_data["outer_space_relationship"]

FEM_basic_data = os.path.join(file_data["file_paths"]["FEMData"], file_data["file_names"]["FEM_basic_data"])
FEM_mic_data_ori = os.path.join(file_data["file_paths"]["FEMData"],
                                file_data["file_names"]["mic_structure_data"])
FEM_mic_data_ref = os.path.join(file_data["file_paths"]["FEMData"],
                                file_data["file_names"]["mic_structure_data2"])
mic_FEM_data = os.path.join(file_data["file_paths"]["FEMData"],
                            file_data["file_names"]["mic_FEM_data"])
mic_results = os.path.join(file_data["file_paths"]["FEMData"],
                           file_data["file_names"]["mic_results"])

FEM_loading = os.path.join(file_data["file_paths"]["FEMData_prescribed"],
                           file_data["file_names"]["FEA_loading"])
FEM_sematics = os.path.join(file_data["file_paths"]["FEMData_prescribed"],
                            file_data["file_names"]["FEA_semantics"])
SAP_path = file_data["file_paths"]["sap_dirpath"]

if not os.path.exists(file_data["file_paths"]["FEMData"]):
    os.makedirs(file_data["file_paths"]["FEMData"])

# endregion

# region preprocess
story_height = {"0": 3000, "1": 3000, "2": 3000}
modular_type_case = 'case2' ###### 这个要改
case_number = 3
case_name = 'layout' + str(case_number) + '.json'
with open(os.path.join(file_data["file_paths"]["Layout_Resulst"], case_name), 'r') as f:
    modular_plan = json.load(f)
modular_plan = {int(key): value for key, value in modular_plan.items()}

# 收集所有元素并去重
all_elements = set()

# 遍历每个键对应的列表
for values in modular_plan.values():
    all_elements.update(values)  # 将列表中的元素添加到集合中

# 将集合转换为列表并排序
sorted_elements = sorted(all_elements)


with open(os.path.join(file_data["file_paths"]["BuildingData"], file_data["file_names"]["mic_types"]), 'r') as f:
    tp = json.load(f)
modular_type = tp[modular_type_case]
modular_type = {int(key): value for key, value in modular_type.items()}

# endregion


#  region FEM modelling and analysis ------------
reload(ut)
project_info = ut.output_structured_data(building_data, modular_plan, modular_type, story_height, FEM_basic_data)
MiC_info = ut.implement_modular_structure_data(FEM_basic_data, FEM_mic_data_ori)
nodes, edges, planes = ut.transform_mic_data(MiC_info)

# draw_point(MiC_info['spaces']['nodes'])


MiC_info2 = ut.modify_mic_geo(FEM_mic_data_ori, FEM_mic_data_ref, contraction=200)
nodes, edges, planes = ut.transform_mic_data2(MiC_info2)
FEA_info2 = ut.implement_FEA_info_enrichment(FEM_mic_data_ref, FEM_loading, mic_FEM_data)



modular_num = len(sorted_elements)  # 模块种类数
num_thread = 1  # 线程数
pop_size = 6  # 种群数量
n_iteration = 1
CROSSOVER_RATE = 0.6
MUTATION_RATE = 0.15
modular_FEM = {
    1: {"sections": [6, 8, 12]},
    4: {"sections": [2, 7, 17]}
}

section_info = FC.extract_section_info()
SapModel_name, mySapObject_name, ModelPath_name, File_Path = MF.mulit_sap(num_thread)
FEA.parsing_to_sap2000_mulit(FEA_info2, FEM_sematics, modular_FEM, File_Path[0],SapModel_name[0], mySapObject_name[0],ModelPath_name[0])
# all_min_fit, all_min_weight, all_min_pop = GA_structure(SapModel_name, mySapObject_name, ModelPath_name, File_Path,sorted_elements)
gc.collect()

# path=os.path.join(os.getcwd(), f'Structural_GA_data\GA_data_case{1}.json')

# SapModel_name, mySapObject_name, ModelPath_name, File_Path = MF.mulit_sap(num_thread)
#
# chromo = [3,7,12,14,9,12]

# fit,weight = calute(File_Path[0], ModelPath_name[0], mySapObject_name[0], SapModel_name[0],chromo,mic_FEM_data,FEM_sematics,modular_num,FEA_info2,1000)
# # draw_fitness(path)
# print(fit)
# print(weight)
