import copy
import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA
import os
import MULIT_FEM as MF

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

case_number = 1
case_name = 'layout' + str(case_number) + '.json'
with open(os.path.join(file_data["file_paths"]["Layout_Resulst"], case_name), 'r') as f:
    modular_plan = json.load(f)
modular_plan = {int(key): value for key, value in modular_plan.items()}

with open(os.path.join(file_data["file_paths"]["BuildingData"], file_data["file_names"]["mic_types"]), 'r') as f:
    tp = json.load(f)
modular_type = tp['case1']
modular_type = {int(key): value for key, value in modular_type.items()}

# endregion


#  region FEM modelling and analysis ------------
reload(ut)
project_info = ut.output_structured_data(building_data, modular_plan, modular_type, story_height, FEM_basic_data)
MiC_info = ut.implement_modular_structure_data(FEM_basic_data, FEM_mic_data_ori)
nodes, edges, planes = ut.transform_mic_data(MiC_info)

MiC_info2 = ut.modify_mic_geo(FEM_mic_data_ori, FEM_mic_data_ref, contraction=200)
nodes, edges, planes = ut.transform_mic_data2(MiC_info2)

# ut.plot_3D_members(nodes, edges, planes)

# endregion ------------------


# region FEM information enrichment and generation
reload(ut)
# FEA_info = ut.implement_FEA_info('FEMData_prescribed/')
modular_FEM = {
    1: {"sections": [6, 8, 12]},
    2: {"sections": [2, 7, 17]}
}

FEA_info2 = ut.implement_FEA_info_enrichment(FEM_mic_data_ref, FEM_loading, mic_FEM_data)

import FEM_parser as FEA
FEA.parsing_to_sap2000(FEA_info2, FEM_sematics, modular_FEM, os.path.dirname(mic_FEM_data))
# endregion -------------------


# region Evaluationt
import FEM_Index_calculation as FC
# reload(FC)
file_path = 'D:\desktop\\xia\modular-structural-team\FEMData'
FC.output_index(modular_FEM,file_path ,file_path,mic_FEM_data)

# endregion

# modular_num = 3 #模块种类数
# num_thread = 2 #线程数
# pop_size = 4 #种群数量
#
# section_info = FC.extract_section_info()
#
#
# #生成初始种群
# pop2 = MF.generate_chromosome(modular_num, section_info, pop_size)
# #所有生成运行及保存路径
# SapModel_name, mySapObject_name, ModelPath_name, File_Path = MF.mulit_sap(num_thread)
# #多线程运算
# MF.thread_sap(File_Path, ModelPath_name, mySapObject_name, SapModel_name, num_thread, pop2, mic_FEM_data, FEM_sematics,modular_num,FEA_info2)
# #关闭所有线程模型
# for i in range(len(mySapObject_name)):
#     ret = mySapObject_name[i].ApplicationExit(False)
#     SapModel_name[i] = None
#     mySapObject_name[i] = None
#

