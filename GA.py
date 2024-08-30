import copy
import numpy as np
import utils as ut
import random


# Problem definition
def individual_cfg(out_space_info, out_space_cfg, modular_type):
    max_length = {}
    DNA_digits = {}
    min_modular_length = min(modular_type.values())
    for key, value in out_space_info.items():
        space_num = key
        direction = value['direction']
        points = out_space_cfg[space_num]
        match direction:
            case 'h':
                max_length[space_num] = points[1][0] - points[0][0]
            case 'v':
                max_length[space_num] = points[2][1] - points[1][1]
        DNA_digits[space_num] = round(max_length[space_num] / min_modular_length)

    DNA_max = min(modular_type.keys())

    return max_length, DNA_digits, DNA_max


def gen_cfg(DNA_digits, region_index, modular_type):
    individual_list = [DNA_digits[str(index)] for index in region_index]
    num = sum(individual_list)
    gen_low = np.zeros(num)
    gen_up = np.zeros(num) + max(modular_type.keys())
    return gen_low, gen_up


# GA aspect
def generate_random_individual(DNA_digits: object, region_index: object, modular_type: object) -> object:
    individual = {}
    individual_list = [DNA_digits[str(index)] for index in region_index]
    individual_DNA = np.random.randint(0, modular_type, sum(individual_list))
    individual['gen'] = individual_DNA
    individual['evals'] = None
    individual['eval_flag'] = True
    individual['fitness'] = None
    return individual


def decode(entire_region_dict, region_index, DNA_digits, individual_DNA):
    modular_plan_x = {}
    individual = individual_DNA

    results = {}
    start = 0
    # 循环通过每个长度来切片列表
    for length in region_index:
        end = start + DNA_digits[str(length)]

        try:
            results[length] = individual[start:end]
        except:
            import pdb;
            pdb.set_trace()
            print("Error")
        start = end

    for key, values in entire_region_dict.items():
        for value in values:
            results[value] = results[key]

    for i in range(len(results)):
        modular_plan_x[i] = results[i]
    tp = copy.deepcopy(modular_plan_x)
    for key in modular_plan_x:
        modular_plan_x[key] = [x for x in modular_plan_x[key] if x != 0]

    return modular_plan_x


def generate_inital_population(DNA_digits, region_index, modular_type, population_size):
    population = {}
    for i in range(population_size):
        individual = generate_random_individual(DNA_digits, region_index, modular_type)
        population[i] = individual
    return population


# Evaluation aspect
def fitness_calculation(indi1_eval):
    fitness = 0
    penalty = 1e5
    alpha = 10
    data1 = indi1_eval['data1']
    data2 = indi1_eval['data2']
    total_number = sum(data1.values())
    # import pdb;
    # pdb.set_trace()
    tp = [i for i in data1.values()]
    total_type = 1 -  np.std(tp)/np.mean(tp)
    total_coverage = ut.softmax(data2[1])
    fitness = total_number + alpha * total_type + penalty * total_coverage
    return fitness, total_number, total_coverage


def evaluate_inidvidual(building_data, modular_type, entire_region_dict, region_index, DNA_digits, indi1):
    out_space_num = len(building_data['outer_space_config'])
    out_space_info = building_data["outer_space_per_building"]
    out_space_cfg = building_data["outer_space_config"]
    inner_space_info = building_data["outer_space_has_inner_space"]
    inner_space_cfg = building_data["inner_space_config"]
    out_space_relationship = building_data["outer_space_relationship"]
    individual_DNA = indi1['gen']
    modular_plan_x = decode(entire_region_dict, region_index, DNA_digits, individual_DNA)
    if indi1['eval_flag'] or indi1['evals'] == None:
        indi1['eval_flag'] = False
        data1 = ut.evaluate_modulars(modular_plan_x)
        data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_plan_x)
        data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type,
                                       modular_plan_x)

        indi1['evals'] = {
            'data1': data1,
            'data2': data2
        }
        fitness, modular_num, out_coverage = fitness_calculation(indi1['evals'])
        indi1['fitness'] = fitness
        indi1['evals']['modular_num'] = modular_num
        indi1['evals']['out_coverage'] = out_coverage

    return indi1


def evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits, population):
    for key, indi in population.items():
        population[key] = evaluate_inidvidual(building_data, modular_type, entire_region_dict, region_index, DNA_digits,
                                              indi)
    return population


def fitness_rank_pop_calculation(population):
    fitmax = 2.0
    fitmin = 0.1
    oldfit = []
    for i in range(len(population)):
        oldfit.append(-1 * population[i]['fitness'])

    listorder = np.argsort(oldfit)
    for i, item in enumerate(listorder):
        population[item]['fitness_rank'] = fitmin + (fitmax - fitmin) * (i / (len(population) - 1))

    # oldfit2 = []
    # for i in range(len(population)):
    #     oldfit2.append(population[i]['fitness_rank'])

    return population


def get_rank_fitness(population):
    oldfit2 = []
    for i in range(len(population)):
        oldfit2.append(population[i]['fitness_rank'])
    return oldfit2


# GA operation
def select_best(Pop):
    r1 = []
    for i in range(len(Pop)):
        r1.append(Pop[i]['fitness_rank'])
    s_inds = np.argmax(r1)
    return Pop[s_inds], s_inds


def select_individual(Pop_ori):
    Pop = copy.deepcopy(Pop_ori)
    fitness = []
    for i in range(len(Pop)):
        fitness.append(Pop[i]['fitness_rank'])
    sum_fitness = sum(fitness)
    flag = True
    selected_individual = []
    while flag:
        u = random.random() * sum_fitness
        tp_sum = 0.
        for ind in range(len(Pop_ori)):
            tp_sum += fitness[ind]
            if tp_sum >= u:
                flag = False
                selected_individual = Pop[ind].copy()
                break
    return selected_individual


def operation_cross(individual1, individual2):
    newInd1 = copy.deepcopy(individual1)
    newInd2 = copy.deepcopy(individual2)

    try:
        dim = len(newInd1['gen'])
    except:
        import pdb;
        pdb.set_trace()
        print("Error")

    gen1 = []
    gen2 = []

    if dim == 1:
        pos1 = 1
        pos2 = 1
    else:
        pos1 = random.randrange(0, dim)
        pos2 = random.randrange(0, dim)
    for i in range(dim):
        if min(pos1, pos2) <= i < max(pos1, pos2):
            gen1.append(newInd1['gen'][i])
            gen2.append(newInd2['gen'][i])
        else:
            gen1.append(newInd2['gen'][i])
            gen2.append(newInd1['gen'][i])

    newInd1['gen'] = gen1
    newInd2['gen'] = gen2

    return newInd1, newInd2


def operation_mutation(individual, gen_low, gen_up):
    # import pdb; pdb.set_trace()
    newInd = copy.deepcopy(individual)
    dim = len(newInd['gen'])
    pos = []
    pos_num = min(5, int(dim ** 0.5))
    if dim == 1:
        for i in range(pos_num):
            pos.append(0)
    else:
        for i in range(pos_num):
            pos.append(random.randrange(0, dim))

    for i in range(pos_num):
        if gen_low[pos[i]] == gen_up[pos[i]]:
            newInd['gen'][pos[i]] = gen_up[pos[i]]
        else:
            newInd['gen'][pos[i]] = np.random.randint(gen_low[pos[i]], gen_up[pos[i]] + 1)

    return newInd


# run GA
def runGA(pop_ori, best_ind, gen_low, gen_up, building_data, modular_type, entire_region_dict, region_index, DNA_digits,
          r_cross=0.6, r_mut=0.2):
    current_pop = copy.deepcopy(pop_ori)
    new_pop = {}
    pop_num = len(pop_ori)
    count = -1
    while len(new_pop) != pop_num:
        offspring1 = select_individual(current_pop)
        offspring2 = select_individual(current_pop)
        if random.random() < r_cross:
            offspring1, offspring2 = operation_cross(offspring1, offspring2)
            offspring1['eval_flag'] = True
            offspring2['eval_flag'] = True
        if random.random() < r_mut:
            offspring1 = operation_mutation(offspring1, gen_low, gen_up)
            offspring2 = operation_mutation(offspring2, gen_low, gen_up)
            offspring1['eval_flag'] = True
            offspring2['eval_flag'] = True
        if len(new_pop) < pop_num:
            count += 1
            new_pop[count] = offspring1
        if len(new_pop) < pop_num:
            count += 1
            new_pop[count] = offspring2

    new_pop = evaluate_population(building_data, modular_type, entire_region_dict, region_index, DNA_digits, new_pop)
    new_pop = fitness_rank_pop_calculation(new_pop)

    new_best_ind, index = select_best(new_pop)
    if new_best_ind['fitness'] > best_ind['fitness']:
        new_pop[index] = copy.deepcopy(best_ind)
        new_best_ind = copy.deepcopy(best_ind)

    return new_pop, new_best_ind
