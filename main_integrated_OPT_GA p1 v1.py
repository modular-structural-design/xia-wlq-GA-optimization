import copy
import numpy as np
import json
from importlib import reload
import utils as ut
import GA as GA
import os

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

with open(os.path.join(file_data["file_paths"]["BuildingData"], file_data["file_names"]["mic_types"]), 'r') as f:
    tp = json.load(f)
modular_type = tp['case1']
modular_type = {int(key): value for key, value in modular_type.items()}
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

iters = 10
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

# region save and layout plot
reload(ut)
modular_plan_x = GA.decode(entire_region_dict, region_index, DNA_digits, best_ind_new['gen'])
tp = ut.output_layouts(modular_plan_x, 1)

data1 = ut.evaluate_modulars(modular_plan_x)
data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_plan_x)
data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type, modular_plan_x)

case1 = ut.draw_data_transform(modular_plan_x, modular_type, out_space_info, out_space_cfg)
ut.draw_case(case1)
# endregion

