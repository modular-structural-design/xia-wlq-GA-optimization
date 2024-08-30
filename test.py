import copy
import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA
import os
import FEM_parser as FEA
import configparser
import comtypes.client
import sys







def mulit_get_sap(num_thread):
    case_name = []
    File_Path = []
    mySapObject_name = []
    SapModel_name = []
    ModelPath_name = []
    for i in range(num_thread):
        case_name.append(f"cases{i}")
        File_Path.append(os.path.join(os.getcwd(), f"cases{i}"))
        mySapObject_name.append(f"mySapObject{i}")
        SapModel_name.append(f"SapModel{i}")
        ModelPath_name.append(f"ModelPath{i}")
        mySapObject_name[i],ModelPath_name[i], SapModel_name[i] = FEA.sap2000_initialization_mulit(File_Path[i])
    return ModelPath_name,mySapObject_name,SapModel_name,File_Path




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

# FEA.parsing_to_sap2000(FEA_info2, FEM_sematics, modular_FEM, os.path.dirname(mic_FEM_data))
# endregion -------------------

num_thread= 1
SAP_path = file_data["file_paths"]["sap_dirpath"]
if not os.path.exists(file_data["file_paths"]["FEMData"]):
    os.makedirs(file_data["file_paths"]["FEMData"])

# ModelPath_name,mySapObject_name,  SapModel_name, File_Path=mulit_get_sap(num_thread)

def mulit_sap(num_thread):
    for i in range(num_thread):
        File_Path=[]
        all_SapModel =[]
        all_mySapObject = []
        all_model_path = []
        File_Path.append(os.path.join(os.getcwd(), f"cases{i}"))
        sap_model_file = os.path.join(File_Path[i], 'FEM_sap2000\\MiC1.sdb')
        SapModel, mySapObject = FEA.sap2000_initialization(File_Path[i])
        all_SapModel.append(SapModel)
        all_mySapObject.append(mySapObject)
        all_model_path.append(sap_model_file)
    return all_SapModel,all_mySapObject,all_model_path,File_Path

all_SapModel,all_mySapObject,all_model_path,File_Path=mulit_sap(num_thread)
# FEA.parsing_to_sap2000_mulit(FEA_info2, FEM_sematics, modular_FEM, File_Path[0],SapModel_name[0], mySapObject_name[0],ModelPath_name[0])
FEA.parsing_to_sap2000_mulit(FEA_info2, FEM_sematics, modular_FEM, File_Path[0],all_SapModel[0], all_mySapObject[0],all_model_path[0])
