import copy

import FEM_parser as FEA
import utils as ut
import numpy as np
import math as m
import json
import FEM_Index_calculation as FC
import random
from random import randint
import pyvista as pv
import os
import queue
import threading
import configparser
import comtypes.client
import sys


def out_put_reaction(SapModel,frames):
    name_re = []
    frame_reactions = []
    frame_reactions_all = []
    for edge_indx in range(len(frames)):
        result = []
        P_na = []
        mm1 = np.zeros((7, 3))
        mm2 = []
        Obj, ObjSta, P, V2, V3, T, M2, M3 = get_frame_reactions("frame"+str(edge_indx), SapModel)
        if len(P) != 0:
            # result.append(Obj)
            result.append(ObjSta)
            result.append(P)
            result.append(V2)
            result.append(V3)
            result.append(T)
            result.append(M2)
            result.append(M3)
            num_fra = len(Obj)
            mid_num = int(0.5 * (num_fra))
            name_re.append(Obj[0])

            for i in range(len(result)):
                mm1[i][0] = result[i][0]
                mm1[i][1] = result[i][mid_num]
                mm1[i][2] = result[i][num_fra - 1]
            frame_reactions.append(mm1)
            frame_reactions_all.append(result)
    mm = ["ObjSta", "P", "V2", "V3", "T", "M2", "M3"]
    frame_weight = []


    return frame_reactions

def get_frame_reactions(frames,SapModel):
    result = []
    Object11 = 0
    Obj = []
    ObjSta = []
    Elm = []
    ElmSta = []
    LoadCase = []
    StepType = []
    StepNum = []
    NumberResults = 0
    P = []
    V2 = []
    V3 = []
    T = []
    M2 = []
    M3 = []
    [NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2, M3,
     ret] = SapModel.Results.FrameForce(frames, Object11, NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType, StepNum, P, V2, V3, T, M2,M3)
    return Obj, ObjSta,P, V2, V3, T, M2,M3

def out_put_displacement(Nodes, SapModel):
    displacements = []
    displacements_hor = []
    name_all_nodes = []
    for i in range(len(Nodes)):
        result = []
        Obj,U1, U2, U3, R1, R2, R3 = get_point_displacement("nodes"+str(i), SapModel)
        # if len(U1) != 0:
        name_all_nodes.append(Obj[0])
        result.append(U1[0])
        result.append(U2[0])
        result.append(U3[0])
        # result.append(R1[0])
        # result.append(R2[0])
        # result.append(R3[0])
        displacements.append(result)
        displacements_hor.append(m.sqrt(U1[0]**2+U2[0]**2))
    displacements = np.array(displacements)

    return displacements

def get_point_displacement(nodes,SapModel):
    displacements = []
    ObjectElm = 0
    NumberResults = 0
    m001 = []
    result = []
    Obj = []
    Elm = []
    ACase = []
    StepType = []
    StepNum = []
    U1 = []
    U2 = []
    U3 = []
    R1 = []
    R2 = []
    R3 = []
    ObjectElm = 0
    [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3,ret] = SapModel.Results.JointDispl(nodes, ObjectElm, NumberResults, Obj,Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    return Obj,U1, U2, U3, R1, R2, R3

def calculate_g(Frame_section_property,frame_reactions,frame_area,frame_length):
    # 柱强度验算
    rx = 1
    ry = 1
    f = 355
    wnx = 906908.8
    wny = 101172.04
    faix = 0.8
    faiy = 0.8
    bmx = 0.9
    btx = 0.9
    bmy = 0.9
    bty = 0.9
    Nex = 5
    Ney = 5
    n_canshu = 1
    faiby = 0.8
    faibx = 0.8
    G11 = (abs(frame_reactions[1]) / f / frame_area) + (
                abs(frame_reactions[5]) / f / rx / Frame_section_property['S22']
                ) + (abs(frame_reactions[6]) / f / ry / Frame_section_property['S33']) - 1

    G21 = (abs(frame_reactions[1]) / f / frame_area / faix) + (
                bmx * abs(frame_reactions[5]) / f / rx / Frame_section_property['S22']
                / (1 - 0.8 * abs(frame_reactions[1]) / abs(Frame_section_property['I22']) / 1846434.18 *
                   frame_length *
                   frame_length)) + n_canshu * (
                      bty * abs(frame_reactions[6]) / f / Frame_section_property['S33'] / faiby) - 1
    G31 = (abs(frame_reactions[1]) / f / frame_area / faiy) + n_canshu * (
                btx * abs(frame_reactions[5]) / f / Frame_section_property['S22']
                / faibx) + (bmy * abs(frame_reactions[5]) / f / ry / Frame_section_property['I22'] / (
                1 - 0.8 * abs(frame_reactions[1]) / Frame_section_property[
            'I33'] / 1846434.18 * frame_length * frame_length)) - 1

    return [G11,G21,G31]

def output_data(SapModel,FEA_info2):
    comb_name = list(FEA_info2['load_combinations'].keys())[0]
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    ret = SapModel.Results.Setup.SetComboSelectedForOutput(comb_name)

    frame_reaction = out_put_reaction(SapModel,FEA_info2['frames_index'])
    node_info =out_put_displacement(FEA_info2['nodes_geo'], SapModel)

    node_dis_dict ={}
    frame_reaction_dict = {}
    for i in range(len(node_info)):
        node_dis_dict["nodes"+str(i)] = node_info[i].tolist()
        frame_reaction_dict["frame"+str(i)] = frame_reaction[i].tolist()

    all_infor=[node_dis_dict,frame_reaction_dict]
    json_str = json.dumps(all_infor)
    with open('calculate_data.json', 'w') as json_file:
        json_file.write(json_str)
    return all_infor

def plot_3D_members(nodes, edges, planes,all_up, file_path='Results/', node_radius=250, edge_radius=200):
    colors = [
        [222.0 / 255, 226.0 / 255, 230.0 / 255],  ## grey
        [254.0 / 255, 95.0 / 255, 85.0 / 255],  ## red
        [67.0 / 255, 97.0 / 255, 238.0 / 255],  ## deep blue
        [140.0 / 255, 179.0 / 255, 105.0 / 255],  ## deep green
        [44.0 / 255, 110.0 / 255, 73.0 / 255],  ## for edge1
        [214.0 / 255, 140 / 255, 69.0 / 255],  ## for edge2
        [255.0 / 255, 173.0 / 255, 173.0 / 255],  ## for edge3
        [0.0 / 255, 78.0 / 255, 137.0 / 255],
    ]
    nodes_tp = np.array(nodes, dtype=float)
    x = nodes_tp[:, 0]
    y = nodes_tp[:, 1]
    z = nodes_tp[:, 2]

    # p2 = pv.Plotter(shape=(1, 1))
    p2 = pv.Plotter(window_size=[1600, 1200])
    p = pv.MultiBlock()
    # import pdb;
    # pdb.set_trace()
    for i in range(nodes_tp.shape[0]):
        sphere = pv.Sphere(radius=node_radius, center=(nodes_tp[i, 0], nodes_tp[i, 1], nodes_tp[i, 2]))
        p.append(sphere)
    # p = p.combine()
    p2.add_mesh(p, color=colors[1], show_edges=False)

    indx = np.array(edges)
    p = pv.MultiBlock()
    for i in range(indx.shape[0]):
        if i not in all_up:
            tube = pv.Tube(
                (x[indx[i, 0]], y[indx[i, 0]], z[indx[i, 0]]),
                (x[indx[i, 1]], y[indx[i, 1]], z[indx[i, 1]]),
                radius=edge_radius,
            )
            p.append(tube)
    p = p.combine()
    p2.add_mesh(p, color=colors[2], show_edges=False)

    p = pv.MultiBlock()
    for i in range(indx.shape[0]):
        if i in all_up:
            tube = pv.Tube(
                (x[indx[i, 0]], y[indx[i, 0]], z[indx[i, 0]]),
                (x[indx[i, 1]], y[indx[i, 1]], z[indx[i, 1]]),
                radius=edge_radius,
            )
            p.append(tube)
    p = p.combine()
    p2.add_mesh(p, color=colors[1], show_edges=False)

    # plane = np.array(planes)
    # p = pv.MultiBlock()
    # for i in range(plane.shape[0]):
    #     if len(plane[i]) == 3:
    #         point1 = [x[plane[i, 0]], y[plane[i, 0]], z[plane[i, 0]]]
    #         point2 = [x[plane[i, 1]], y[plane[i, 1]], z[plane[i, 1]]]
    #         point3 = [x[plane[i, 2]], y[plane[i, 2]], z[plane[i, 2]]]
    #         # point4 = [x[plane[i,3]], y[plane[i,3]], z[plane[i,3]]]
    #         Triangle = pv.Triangle([point1, point2, point3])
    #         p.append(Triangle)
    #
    #     elif len(plane[i]) == 4:
    #         point1 = [x[plane[i, 0]], y[plane[i, 0]], z[plane[i, 0]]]
    #         point2 = [x[plane[i, 1]], y[plane[i, 1]], z[plane[i, 1]]]
    #         point3 = [x[plane[i, 2]], y[plane[i, 2]], z[plane[i, 2]]]
    #         point4 = [x[plane[i, 3]], y[plane[i, 3]], z[plane[i, 3]]]
    #         Triangle = pv.Triangle([point1, point2, point3])
    #         p.append(Triangle)
    #         Triangle = pv.Triangle([point1, point3, point4])
    #         p.append(Triangle)
    #
    # p = p.combine()
    # p2.add_mesh(p, color=colors[0], show_edges=False, opacity=0.35)

    p2.set_background("white")
    p2.camera.azimuth = 200
    p2.camera.elevation = 0

    file_name = os.path.join(file_path, 'mic_geo.pdf')
    # isExist = os.path.exists(file_name)
    # if not isExist:
    #     os.makedirs(file_name)
    p2.save_graphic(file_name)
    pass


def draw_up_member(results):
    all_up = []
    for i in range(len(results)):
        for j in range(len(results[i][1])):
            for z in range(len(results[i][1][j])):
                if results[i][1][j][z] > 0:
                    all_up.append(i)

    all_up = list(set(all_up))

    MiC_info2 = ut.modify_mic_geo('FEMData/', contraction=200)
    nodes, edges, planes = ut.transform_mic_data2(MiC_info2)
    plot_3D_members(nodes, edges, planes, all_up)

def generate_chromosome(modular_num,section_info,pop_size):
    all_chro = []
    for i in range(pop_size):
        chromo = [random.randint(0, len(section_info)-1) for _ in range(modular_num*3)]
        all_chro.append(chromo)
    return all_chro

def SAPanalysis_GA_run2(APIPath):


    cfg = configparser.ConfigParser()
    cfg.read("Configuration.ini", encoding='utf-8')
    ProgramPath = cfg['SAP2000PATH']['dirpath']
    if not os.path.exists(APIPath):
        try:
            os.makedirs(APIPath)
        except OSError:
            pass

    AttachToInstance = False
    SpecifyPath = True

    # ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    helper = comtypes.client.CreateObject('SAP2000v1.Helper')
    helper = helper.QueryInterface(comtypes.gen.SAP2000v1.cHelper)
    if AttachToInstance:
        # attach to a running instance of SAP2000
        try:
            # get the active SapObject
            mySapObject = helper.Getobject("CSI.SAP2000.API.SapObject")
        except (OSError, comtypes.COMError):
            print("No running instance of the program found or failed to attach.")
            sys.exit(-1)
    else:
        if SpecifyPath:
            try:
                # 'create an instance of the SAPObject from the specified path
                mySapObject = helper.CreateObject(ProgramPath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from" + ProgramPath)
                sys.exit(-1)
        else:
            try:
                # create an instance of the SapObject from the latest installed SAP2000
                mySapObject = helper.CreateObjectProgID("CSI.SAP2000.API.SapObject")
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program")
                sys.exit(-1)

        # start SAP2000 application
        mySapObject.ApplicationStart()

    # create SapModel object
    SapModel = mySapObject.SapModel
    # initialize model
    SapModel.InitializeNewModel()
    ModelPath = os.path.join(APIPath, 'API_1-001.sdb')
    # create new blank model
    return mySapObject,ModelPath, SapModel

def select_2(pop, fitness):  # nature selection wrt pop's fitness

    fit_ini = copy.deepcopy(fitness)
    luyi = copy.deepcopy(fitness)
    luyi.sort(reverse=True)
    sort_num = []
    for i in range(len(fit_ini)):
        sort_num.append(luyi.index(fit_ini[i]))

    for i in range(len(sort_num)):
        if sort_num[i]==0:
            sort_num[i]+=0.01

    idx = np.random.choice(np.arange(len(pop)), size=len(pop), replace=True,
                           p=np.array(sort_num) / (sum(sort_num)))
    pop2 = np.zeros((len(pop), len(pop[0])))
    for i in range(len(pop2)):
        pop2[i] = pop[int(idx[i])]
    return pop2


def calculate_fitness(pop_all,u):
    pop = copy.deepcopy(pop_all)
    all_fitness= []
    all_weight = []
    SapModel, mySapObject = FEA.sap2000_initialization()
    for num in range(len(pop)):
        merged_list = [pop[num][i:i + 3] for i in range(0, len(pop[num]), 3)]
        modular_FEM = {}

        # 用 for 循环生成字典
        for i in range(0, modular_num):  # 假设你需要生成键 1 到 2
            modular_FEM[i + 1] = {"sections": merged_list[i]}

        SapModel, mySapObject = FEA.parsing_to_sap2000(FEA_info2, 'FEMData/FEA_semantic_lists.json', modular_FEM)
        all_infor = output_data(SapModel, FEA_info2)
        FC.output_index(modular_FEM)
        with open('max_values.json', 'r') as file:
            index_data = json.load(file)
        value_dict = {key: item["value"] for key, item in index_data.items()}
        values_list = list(value_dict.values())
        value_id = []
        for j in range(len(values_list)-1):
            if values_list[j]<0:
                value_id.append(0)
            else:
                value_id.append(values_list[j])
        weight = values_list[-1]
        fit = weight + u*(sum(value_id))
        all_fitness.append(fit)
        all_weight.append(weight)
    return all_fitness,all_weight,SapModel, mySapObject

def crossover_and_mutation(pop2,CROSSOVER_RATE,MUTATION_RATE,section_info):
    pop = pop2

    new_pop = np.zeros((len(pop),len(pop[0])))
    for i in range(len(pop)):
        father = pop[i]
        child = father
        if np.random.rand() < CROSSOVER_RATE:
            mother = pop[np.random.randint(len(pop2))]
            cross_points1 = np.random.randint(low=0, high=len(pop[0]))
            cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            while cross_points2==cross_points1:
                cross_points2 = np.random.randint(low=0, high=len(pop[0]))
            exchan = []
            exchan.append(cross_points2)
            exchan.append(cross_points1)
            for j in range(min(exchan),max(exchan)):
                child[j] = mother[j]
        mutation(child,MUTATION_RATE,section_info)
        new_pop[i] = child


    return new_pop

def mutation(child,MUTATION_RATE,section_info):

    for i in range(len(child)):
        if np.random.rand() < MUTATION_RATE:
            child[i] = random.randint(0,len(section_info)-1)


def GA_run_modular(modular_num,section_info,pop_size,u,CROSSOVER_RATE,MUTATION_RATE):
    pop_all= generate_chromosome(modular_num,section_info,pop_size)

    pop_zhongqun_all = []  # 记录每代种群（不重复）

    pop_all_weight=[]
    pop_all_fitness=[]
    weight_min=[]
    min_ru = []
    sap_run_time = 0
    for run_time in range(N_GENERATIONS):
        pop_zhongqun_all.append(pop_all)

        # 计算fitness等参数
        all_fitness, all_weight,SapModel, mySapObject = calculate_fitness(pop_all, u)

        pop_all_weight.append(all_weight)
        pop_all_fitness.append(all_fitness)
        mm = all_fitness.index(min(all_fitness))
        weight_min.append(all_weight[mm])
        min1 = min(all_fitness)
        mm2_all = pop_all[mm]# 最小值对应pop2编码

        min_ru.append(min(all_fitness))# 统计历代最小值
        #选择
        pop_all = np.array(pop_all)
        pop_all = select_2(pop_all, all_fitness)
        #交叉变异
        pop_all = crossover_and_mutation(pop_all,CROSSOVER_RATE,MUTATION_RATE,section_info)
        pop_all.tolist()

        #精英策略
        # if max1 <= m.log(GA(aaa,pop3_ga)[0][0]):
        if min1 <=calculate_fitness([pop_all[0]],u)[0][0]:
            sap_run_time += 1
            pop_all[0] = mm2_all


    ret = mySapObject.ApplicationExit(False)
    SapModel = None
    mySapObject = None



    return min_ru



if __name__ == '__main__':
    modular_num = 3
    pop_size = 4
    # num_thread = 5 #线程数
    N_GENERATIONS = 3
    CROSSOVER_RATE=0.6
    MUTATION_RATE = 0.15
    section_info = FC.extract_section_info()
    FEA_info2 = ut.implement_FEA_info_enrichment('FEMData/')
    modular_FEM = {
        1: {"sections": [17, 17, 17]},
        2: {"sections": [17, 17, 17]}
    }

    # min_ru=GA_run_modular(modular_num, section_info, pop_size, 10000,CROSSOVER_RATE, MUTATION_RATE)

    FEA.parsing_to_sap2000(FEA_info2, 'FEMData/FEA_semantic_lists.json', modular_FEM)
    # FP.output_data(SapModel, FEA_info2)
    # 输出指标
    FC.output_index(modular_FEM)

    # frame_reactions = [0,0,-10095,0,0,0,-4620456]
    # Frame_section_property={'S22':100,'S33':100,'I22':100,'I33':100}
    # frame_area = 500
    # frame_length=500
    # a = calculate_g(Frame_section_property, frame_reactions, frame_area, frame_length)






