import numpy as np
import json
from importlib import reload
import utils as ut

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
    0: None,
    1: 3000,
    2: 4000
}
modular_x = {}  # initialization of modular distribution
for i in range(out_space_num):
    modular_x[i] = []

modular_x_test = {}
modular_x_test[0] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[1] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[2] = [2, 2, 2]
modular_x_test[3] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[4] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[5] = [2, 2, 2]
modular_x_test[6] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[7] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
modular_x_test[8] = [2, 2, 2]
# 竖向区域补在后面
modular_x_test[9] = [2, 2, 2]
modular_x_test[10] = [2, 2, 2]
modular_x_test[11] = [2, 2, 2]
modular_x_test[12] = [2, 2, 2]
modular_x_test[13] = [2, 2, 2]
modular_x_test[14] = [2, 2, 2]

# Evaluation functions
reload(ut)
data1 = ut.evaluate_modulars(modular_x_test)
data2 = ut.evaluate_outspace(out_space_info, out_space_cfg, modular_type, modular_x_test)
data3 = ut.evaluate_innerspace(out_space_info, inner_space_info, inner_space_cfg, modular_type, modular_x_test)


case1 = ut.draw_data_transform(modular_x, modular_type, out_space_info, out_space_cfg)
ut.draw_case(case1)
# endregion
