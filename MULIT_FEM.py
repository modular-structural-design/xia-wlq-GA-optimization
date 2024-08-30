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
import FEM_Index_calculation as FC


def mulit_sap(num_thread):
    File_Path = []
    all_SapModel = []
    all_mySapObject = []
    all_model_path = []
    for i in range(num_thread):

        File_Path.append(os.path.join(os.getcwd(), f"cases{i}"))
        sap_model_file = os.path.join(File_Path[i], 'FEM_sap2000\\MiC1.sdb')
        SapModel, mySapObject = FEA.sap2000_initialization(File_Path[i])
        all_SapModel.append(SapModel)
        all_mySapObject.append(mySapObject)
        all_model_path.append(sap_model_file)
    return all_SapModel,all_mySapObject,all_model_path,File_Path


def thread_sap(File_Path,ModelPath,mySapObject_name,SapModel_name,num_thread,pop2,mic_FEM_data,FEM_sematics,modular_num,FEA_info2,all_chro_data):

    q = queue.Queue()
    threads = []
    for i in range(len(pop2)):
        q.put(i)
    for i in range(num_thread):
        t = threading.Thread(target=mulitrun_GA_1, args=(File_Path[i],ModelPath[i],mySapObject_name[i],SapModel_name[i],pop2,q,mic_FEM_data,FEM_sematics,modular_num,FEA_info2,all_chro_data))
        t.start()
        threads.append(t)
    for i in threads:
        i.join()
    # return result,weight_1,col_up,beam_up,gx_te
    return all_chro_data

def mulitrun_GA_1(File_Path,ModelPath,mySapObject, SapModel,pop_all,q,mic_FEM_data,FEM_sematics,modular_num,FEA_info2,all_chro_data):
    while True:
        if q.empty():
            break
        time = q.get()
        pop2= pop_all[time]

        # 染色体解码
        merged_list = [pop2[i:i + 3] for i in range(0, len(pop2), 3)]
        modular_FEM = {}

        # 用 for 循环生成字典
        for i in range(0, modular_num):  # 假设你需要生成键 1 到 2
            modular_FEM[i + 1] = {"sections": merged_list[i]}

        FEA.parsing_to_sap2000_mulit(FEA_info2, FEM_sematics, modular_FEM, File_Path,SapModel, mySapObject,ModelPath)

        FC.output_index(modular_FEM, File_Path, File_Path,mic_FEM_data)
        calaulate_fitness(File_Path, all_chro_data, 1000, time)
def generate_chromosome(modular_num,section_info,pop_size):
    all_chro = []
    for i in range(pop_size):
        chromo = [random.randint(0, len(section_info)-1) for _ in range(modular_num*3)]
        all_chro.append(chromo)
    return all_chro

def calaulate_fitness(file_path,all_chrom_data,u,run_time):
    with open(os.path.join(file_path, 'max_values.json'), 'r') as file:
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
    all_chrom_data[f'chro{run_time}']['fitness'] = fit
    all_chrom_data[f'chro{run_time}']['weight'] = weight
if __name__ == '__main__':

    with open('config.json', 'r') as f:
        file_data = json.load(f)

    with open(os.path.join(file_data["file_paths"]["BuildingData"], file_data["file_names"]["building_data"]),
              'r') as f:
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
    mic_FEM_data = file_data["file_names"]["mic_FEM_data"] ##############
    mic_results = os.path.join(file_data["file_paths"]["FEMData"],
                               file_data["file_names"]["mic_results"])

    FEM_loading = os.path.join(file_data["file_paths"]["FEMData_prescribed"],
                               file_data["file_names"]["FEA_loading"])
    FEM_sematics = os.path.join(file_data["file_paths"]["FEMData_prescribed"],
                               file_data["file_names"]["FEA_semantics"])

    SAP_path = file_data["file_paths"]["sap_dirpath"]
    if not os.path.exists(file_data["file_paths"]["FEMData"]):
        os.makedirs(file_data["file_paths"]["FEMData"])

    modular_num = 3
    num_thread = 2
    pop_size = 4
    FEM_loading = os.path.join(file_data["file_paths"]["FEMData_prescribed"],
                               file_data["file_names"]["FEA_loading"])
    FEA_info2 = ut.implement_FEA_info_enrichment(FEM_mic_data_ref,FEM_loading, mic_FEM_data)
    # FEA_semantic_file='FEMData/FEA_semantic_lists.json'
    SapModel_name, mySapObject_name,ModelPath_name,File_Path=mulit_sap(num_thread)
    section_info = FC.extract_section_info()
    pop2=generate_chromosome(modular_num, section_info, pop_size)
    thread_sap(File_Path,ModelPath_name, mySapObject_name, SapModel_name, num_thread, pop2,mic_FEM_data,FEM_sematics,modular_num,FEA_info2)
    # for i in range(len(mySapObject_name)):
    #     ret = mySapObject_name[i].ApplicationExit(False)
    #     SapModel_name[i] = None
    #     mySapObject_name[i] = None
