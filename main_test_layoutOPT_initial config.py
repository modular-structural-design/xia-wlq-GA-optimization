import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA

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
modular_type = {
    0: 10000000,
    1: 3000,
    2: 4000
}
modular_x = {}  # initialization of modular distribution
for i in range(out_space_num):
    modular_x[i] = []

modular_x_test = modular_x
modular_x_test[0] = [2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2]
modular_x_test[1] = [2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2]
modular_x_test[2] = [1]

modular_x_test[3] = [2, 2, 1, 1, 2, 1, 2, 2, 1, 2, 2, 2]
modular_x_test[4] = [2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2]
modular_x_test[5] = [1, 1, 2, 2, 2, 1, 1, 2, 2, 2]

modular_x_test[6] = [2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2]
modular_x_test[7] = [2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 2, 2, 2]
modular_x_test[8] = [1, 1, 2, 2, 2, 1, 1, 2, 2, 2]
# 竖向区域补在后面
modular_x_test[9] = [1]
modular_x_test[10] = [1]

modular_x_test[11] = [1, 1, 1, 1]
modular_x_test[12] = [1, 1, 1, 1]

modular_x_test[13] = [1, 1, 1, 1]
modular_x_test[14] = [1, 1, 1, 1]

# Evaluation functions
reload(ut)
data1 = ut.evaluate_modulars(modular_x_test)
data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_x_test)
data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type, modular_x_test)

case1 = ut.draw_data_transform(modular_x, modular_type, out_space_info, out_space_cfg)
ut.draw_case(case1)
# endregion

# GA setting
reload(ut)
reload(GA)
max_length, DNA_digits, _ = GA.individual_cfg(out_space_info, out_space_cfg, modular_type)


region_index = [0, 1, 2, 9, 10]
entire_region_dict = {
    0: [3, 6],
    1: [4, 7],
    2: [5, 8],
    9: [11, 13],
    10: [12, 14]
}
gen_low, gen_up = GA.gen_cfg(DNA_digits, region_index, modular_type)

indi1 = GA.generate_random_individual(DNA_digits, region_index, len(modular_type))
pop1 = GA.generate_inital_population(DNA_digits, region_index, len(modular_type), 5)
pop1 = GA.evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits, pop1)
pop1 = GA.fitness_rank_pop_calculation(pop1)

# GA part
reload(GA)
rank_fitness = GA.get_rank_fitness(pop1)
best_ind = GA.select_best(pop1)
sel_ind1 = GA.select_individual(pop1)
sel_ind2 = GA.select_individual(pop1)
tp1, tp2 = GA.operation_cross(sel_ind1, sel_ind2)
tp3 = GA.operation_mutation(sel_ind1, gen_low, gen_up)


# region backup
modular_plan_x, tp = GA.decode(entire_region_dict, region_index, DNA_digits, indi1)

reload(ut)
data1 = ut.evaluate_modulars(modular_plan_x)
data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_plan_x)
data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type, modular_plan_x)

case1 = ut.draw_data_transform(modular_plan_x, modular_type, out_space_info, out_space_cfg)
ut.draw_case(case1)
# endregion
