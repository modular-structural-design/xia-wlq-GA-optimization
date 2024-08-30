import copy

import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA
import FEM_Index_calculation as FC
import FEM_parser as FP
# region Reading data
with open('BuildingData/data_case1.json', 'r') as f:
    building_data = json.load(f)
out_space_num = len(building_data['outer_space_config'])
out_space_info = building_data["outer_space_per_building"]
out_space_cfg = building_data["outer_space_config"]
inner_space_info = building_data["outer_space_has_inner_space"]
inner_space_cfg = building_data["inner_space_config"]
out_space_relationship = building_data["outer_space_relationship"]
# endregion

# region Analysis the layout
# modular_type = {
#     0: 10000000,
#     1: 3000,
#     2: 3200,
#     3: 3400,
#     4: 3600,
#     5: 3800,
#     6: 4000
# }
modular_type = {
    0: 10000000,
    1: 3000,
    2: 4000
}
# endregion

# region layout optimization
# GA setting
reload(ut)
reload(GA)
max_length, DNA_digits, _ = GA.individual_cfg(out_space_info, out_space_cfg, modular_type)

region_index = [0, 2, 9, 10]
entire_region_dict = {
    0: [3, 6, 1, 4, 7],
    2: [5, 8],
    9: [11, 13],
    10: [12, 14]
}
gen_low, gen_up = GA.gen_cfg(DNA_digits, region_index, modular_type)

indi1 = GA.generate_random_individual(DNA_digits, region_index, len(modular_type))
pop1 = GA.generate_inital_population(DNA_digits, region_index, len(modular_type), 5)
pop1 = GA.evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits, pop1)
pop1 = GA.fitness_rank_pop_calculation(pop1)

# # GA part
# reload(GA)
# rank_fitness = GA.get_rank_fitness(pop1)
# best_ind = GA.select_best(pop1)
# sel_ind1 = GA.select_individual(pop1)
# sel_ind2 = GA.select_individual(pop1)
# tp1, tp2 = GA.operation_cross(sel_ind1, sel_ind2)
# tp3 = GA.operation_mutation(sel_ind1, gen_low, gen_up)

# run_GA
reload(GA)
pop_ori = GA.generate_inital_population(DNA_digits, region_index, len(modular_type), 100)
pop_ori = GA.evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits, pop_ori)
pop_ori = GA.fitness_rank_pop_calculation(pop_ori)
best_ind, index = GA.select_best(pop_ori)

iters = 100
best_ind_hist = []
pop_new = copy.deepcopy(pop_ori)
best_ind_new = copy.deepcopy(best_ind)
count = 0
for i in range(iters):
    count += 1
    if count % 10 == 0:
        print('add new population')
        pop_tp = GA.generate_inital_population(DNA_digits, region_index, len(modular_type), 100)
        pop_tp = GA.evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits,
                                        pop_tp)
        pop_tp = GA.fitness_rank_pop_calculation(pop_tp)
        best_ind, index = GA.select_best(pop_tp)
        pop_tp[index] = copy.deepcopy(best_ind)
        if best_ind_new['fitness'] > best_ind['fitness']:
            best_ind_new = copy.deepcopy(best_ind)

        pop_new = copy.deepcopy(pop_tp)
    pop_new, best_ind_new = GA.runGA(pop_new, best_ind_new, gen_low, gen_up, building_data, modular_type,
                                     entire_region_dict, region_index, DNA_digits, r_cross=0.6, r_mut=0.2)
    best_ind_hist.append(best_ind_new)
    print(
        f"step: {count};  fitness:  {best_ind_new['fitness']};  coverage: {best_ind_new['evals']['out_coverage']};  modular_num: {best_ind_new['evals']['modular_num']}")

# endregion

# region layout plot
modular_plan_x = GA.decode(entire_region_dict, region_index, DNA_digits, best_ind_new['gen'])

reload(ut)
data1 = ut.evaluate_modulars(modular_plan_x)
data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_plan_x)
data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type, modular_plan_x)

case1 = ut.draw_data_transform(modular_plan_x, modular_type, out_space_info, out_space_cfg)
ut.draw_case(case1)
# endregion

# region FEM modelling and analysis
file_name = 'FEMData_prescribed/basic_structure_data.json'
story_height = {"0": 3000, "1": 3000, "2": 3000}

reload(GA)
modular_plan_x = GA.decode(entire_region_dict, region_index, DNA_digits, best_ind_new['gen'])
tp = {}
for key, value in modular_plan_x.items():
    value_tp = [int(i) for i in value]
    tp[str(key)] = list(value_tp)
with open('plan_tp1.json', 'w') as f:
    json.dump(tp, f, indent=4)

reload(ut)
project_info = ut.output_structured_data(building_data, modular_plan_x, modular_type, story_height, file_name)
MiC_info = ut.implement_modular_structure_data('FEMData_prescribed/')
nodes, edges, planes = ut.transform_mic_data(MiC_info)

MiC_info2 = ut.modify_mic_geo('FEMData_prescribed/', contraction=200)
nodes, edges, planes = ut.transform_mic_data2(MiC_info2)
ut.plot_3D_members(nodes, edges, planes)

# FEM information enrichment and generation
reload(ut)
# FEA_info = ut.implement_FEA_info('FEMData_prescribed/')
modular_FEM = {
    1: {"sections": [12, 12, 12]},
    2: {"sections": [12, 12, 17]}
}
FEA_info2 = ut.implement_FEA_info_enrichment('FEMData_prescribed/')
import FEM_parser as FEA
# FEA.parsing_to_sap2000(FEA_info2,  'FEMData_prescribed/FEA_semantic_lists.json', modular_FEM)

# endregion
#输出计算结果
SapModel, mySapObject = FEA.parsing_to_sap2000(FEA_info2, 'FEMData_prescribed/FEA_semantic_lists.json', modular_FEM)
FP.output_data(SapModel, FEA_info2)
#输出指标
FC.output_index(modular_FEM)