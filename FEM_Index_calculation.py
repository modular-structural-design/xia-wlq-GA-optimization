import json
import math
import os

def extract_nodes_and_frames(model_result_path):
    """
    从calculate_data.json中提取所有以'nodes'和'frame'开头的信息。
    返回:
    dict: 包含所有'nodes'和'frame'相关数据的字典。
    """
    # 存储nodes和frame数据
    all_data = {}

    # 打开并加载 JSON 文件
    with open(os.path.join(model_result_path, 'calculate_data.json'), 'r') as file:
        data = json.load(file)

    # 遍历数据中的每一个对象
    for obj in data:
        # 处理每个对象中的nodes数据
        for key, value in obj.items():
            if key.startswith('nodes'):
                all_data[key] = value

        # 处理每个对象中的frame数据
        for key, value in obj.items():
            if key.startswith('frame'):
                all_data[key] = value
    return all_data


def extract_section_info():
    # 从FEA_semantic_lists.json中提取截面类型信息
    with open('FEMData_prescribed/FEA_semantic_lists.json', 'r') as file:
        data = json.load(file)
    # 提取截面类型信息
    channel_data = data.get('section_types', {}).get('Rect', {})

    return channel_data


def read_fem_data(mic_FEM_data_file):
    # 读取mic_FEM_data.json
    with open(mic_FEM_data_file, 'r') as file:
        data = json.load(file)

    # 提取frames两端节点
    frames_index = data.get('frames_index', {})

    # 提取frames截面类型
    frames_sections = data.get('frames_sections', {})

    # 提取节点坐标
    nodes_geo = {}
    nodes_geo = data.get('nodes_geo', {})

    return frames_index, frames_sections, nodes_geo


def extract_section_properties(frames_sections, section_info, modular_FEM):
    """
    根据模块类型和边缘类型提取截面属性。
    """
    section_properties = {}

    # modular_FEM = {
    #     1: {"sections": [6, 8, 12]},
    #     2: {"sections": [2, 7, 17]}
    # }
    # 提取截面属性
    for frame_key, frame_value in frames_sections.items():
        # 从嵌套字典中提取模块类型和边缘类型
        modular_type = frame_value['modular_type']
        edge_type = frame_value['edge_type']

        # 确定截面类型
        sections = modular_FEM[modular_type]['sections'][edge_type]

        # 获取截面属性
        section_properties[frame_key] = {
            'Area': section_info[str(sections)]['Area'],
            'I33': section_info[str(sections)]['I33'],
            'I22': section_info[str(sections)]['I22'],
            'S33': section_info[str(sections)]['S33'],
            'S22': section_info[str(sections)]['S22']
        }

    return section_properties


def calculate_frame_lengths(frames_index, nodes_geo):
    """
    计算frame两端节点之间的欧几里得距离。
    参数:
    - frames_index: 包含帧及其起始和结束节点索引的字典。
    - nodes_geo: 包含节点索引及其三维坐标的字典。
    返回:
    - frame_lengths: 包含每个帧的长度的字典。
    """
    frame_lengths = {}

    for frame, endpoints in frames_index.items():
        start_node_index, end_node_index = endpoints
        start_node_key = f"nodes{start_node_index}"
        end_node_key = f"nodes{end_node_index}"

        # 获取节点坐标
        start_node = nodes_geo[start_node_key]
        end_node = nodes_geo[end_node_key]

        # 计算欧几里得距离
        distance = math.sqrt(
            (end_node[0] - start_node[0]) ** 2 +
            (end_node[1] - start_node[1]) ** 2 +
            (end_node[2] - start_node[2]) ** 2
        )

        frame_lengths[frame] = distance

    return frame_lengths


def calculate_g(section_properties, frame_reactions, frame_length):
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

    G11 = (abs(frame_reactions[1]) / f / section_properties['Area']) + (
            abs(frame_reactions[5]) / f / rx / section_properties['S22']
    ) + (abs(frame_reactions[6]) / f / ry / section_properties['S33']) - 1

    G21 = (abs(frame_reactions[1]) / f / section_properties['Area'] / faix) + (
            bmx * abs(frame_reactions[5]) / f / rx / section_properties['S22']
            / (1 - 0.8 * abs(frame_reactions[1]) / abs(section_properties['I22']) / 1846434.18 *
               frame_length *
               frame_length)) + n_canshu * (
                  bty * abs(frame_reactions[6]) / f / section_properties['S33'] / faiby) - 1

    G31 = (abs(frame_reactions[1]) / f / section_properties['Area'] / faiy) + n_canshu * (
            btx * abs(frame_reactions[5]) / f / section_properties['S22']
            / faibx) + (bmy * abs(frame_reactions[5]) / f / ry / section_properties['I22'] / (
            1 - 0.8 * abs(frame_reactions[1]) / section_properties[
        'I33'] / 1846434.18 * frame_length * frame_length)) - 1

    return [G11, G21, G31]


def calculate_node_differences(all_data):
    # 计算层间位移角
    story_drift = {}

    # 从node248开始处理数据
    start_node = 248

    # 遍历从node248开始的所有节点
    for i in range(start_node, len(all_data)):
        node_key = f"nodes{i}"
        if node_key in all_data:
            current_node = all_data[node_key]

            # 计算前248个节点的键名
            previous_node_key = f"nodes{i - 248}"

            # 如果前248个节点存在于数据中
            if previous_node_key in all_data:
                previous_node = all_data[previous_node_key]

                # 计算X和Y方向的差值，并除以3000
                diff_x = abs(current_node[0] - previous_node[0]) * 250 / 3000 - 1
                diff_y = abs(current_node[1] - previous_node[1]) * 250 / 3000 - 1

                # 存储结果
                story_drift[node_key] = [diff_x, diff_y]
    return story_drift


def calculate_abs_node_differences(all_data, nodes_geo):
    # 计算绝对层间位移
    abs_story_drift = {}

    # 从node248开始处理数据
    start_node = 248

    # 遍历从node248开始的所有节点
    for i in range(start_node, len(nodes_geo)):
        node_key = f"nodes{i}"
        if node_key in all_data:
            current_node = all_data[node_key]
            node_geo = nodes_geo[node_key]

            # 计算X和Y方向，并除以3000
            diff_x = abs(current_node[0]) * 600 / node_geo[2] - 1
            diff_y = abs(current_node[1]) * 600 / node_geo[2] - 1

            # 存储结果
            abs_story_drift[node_key] = [diff_x, diff_y]
    return abs_story_drift


def find_max_coordinates(results):
    """
    寻找结果中最大 G11、G12 和 G13 值及其对应的frame。
    参数:
    - results (dict): G11,G12,G13及对应的frame。
    返回:
    - dict: 包含最大 G11、G12 和 G13 值及其对应frame。
    """
    # 将列表转换为字典
    results = {frame_name: frame_data for frame_name, frame_data in results}

    max_G11_values = []
    max_G12_values = []
    max_G13_values = []

    frames_with_max_G11 = []
    frames_with_max_G12 = []
    frames_with_max_G13 = []

    for frame_name, frame_d in results.items():
        x_values = [point[0] for point in frame_d]
        y_values = [point[1] for point in frame_d]
        z_values = [point[2] for point in frame_d]

        max_x = max(x_values)
        max_y = max(y_values)
        max_z = max(z_values)

        # 更新最大 G11 值
        if not max_G11_values or max_x > max(max_G11_values):
            max_G11_values.append(max_x)
            frames_with_max_G11 = [frame_name]
        elif max_x == max(max_G11_values):
            frames_with_max_G11.append(frame_name)

        # 更新最大 G12 值
        if not max_G12_values or max_y > max(max_G12_values):
            max_G12_values.append(max_y)
            frames_with_max_G12 = [frame_name]
        elif max_y == max(max_G12_values):
            frames_with_max_G12.append(frame_name)

        # 更新最大 G13 值
        if not max_G13_values or max_z > max(max_G13_values):
            max_G13_values.append(max_z)
            frames_with_max_G13 = [frame_name]
        elif max_z == max(max_G13_values):
            frames_with_max_G13.append(frame_name)

    # 输出最大值及其对应的frame
    max_G11 = max(max_G11_values)
    max_G12 = max(max_G12_values)
    max_G13 = max(max_G13_values)

    return max_G11, frames_with_max_G11, max_G12, frames_with_max_G12, max_G13, frames_with_max_G13


def find_max_story_drift(story_drift):
    """
    筛选最大层间位移。
    参数:
    - story_drift: 层间位移。
    返回:
    -  max_x_value, max_y_value, node_with_max_x, node_with_max_y: x和y向的最大层间位移及节点。
    """
    # 初始化最大值及其对应的节点
    max_x_value = float('-inf')
    max_y_value = float('-inf')
    node_with_max_x = None
    node_with_max_y = None

    # 遍历数据字典
    for node, coordinates in story_drift.items():
        x_value = coordinates[0]
        y_value = coordinates[1]

        # 更新最大 x 值
        if x_value > max_x_value:
            max_x_value = x_value
            node_with_max_x = node

        # 更新最大 y 值
        if y_value > max_y_value:
            max_y_value = y_value
            node_with_max_y = node

    return max_x_value, max_y_value, node_with_max_x, node_with_max_y


def calculate_total_weight(frame_lengths, section_properties):
    """
    计算结构总重。
    """
    # 密度
    constant = 0.00000000785

    # 计算总重量
    total_weight = 0.0
    for frame in frame_lengths:
        length = frame_lengths[frame]
        area = section_properties[frame]['Area']
        weight = length * area * constant
        total_weight += weight

    return total_weight


def output_index(modular_FEM, mic_FEM_data_file, output_file,mic_FEM_data):
    # 提取数据
    all_data = extract_nodes_and_frames(mic_FEM_data_file)
    # 导出构件信息，节点位置
    frames_index, frames_sections, nodes_geo = read_fem_data(mic_FEM_data)
    # 提取截面信息
    section_info = extract_section_info()
    frame_lengths = calculate_frame_lengths(frames_index, nodes_geo)
    section_properties = extract_section_properties(frames_sections, section_info, modular_FEM)
    total_weight = calculate_total_weight(frame_lengths, section_properties)
    # 存储计算结果
    results = []

    # 对于每个frame，计算G值
    for key, value in all_data.items():
        if key.startswith('frame'):
            # 转换列数据为行数据
            frame_data = list(zip(*value))

            # 计算每一列的数据
            column_results = []
            for i in range(len(frame_data)):
                # 选择所需的元素作为 frame_reactions
                frame_reactions = frame_data[i]
                section_propertie = section_properties[key]
                frame_length = frame_lengths[key]
                result = calculate_g(section_propertie, frame_reactions, frame_length)
                column_results.append(result)

            results.append((key, column_results))

    story_drift = calculate_node_differences(all_data)  # 节点层间位移角
    abs_story_drift = calculate_abs_node_differences(all_data, nodes_geo)  # 绝对节点层间位移角
    max_x_story_drift, max_y_story_drift, node_with_max_x_story_drift, node_with_max_y_story_drift = find_max_story_drift(
        story_drift)  # 筛选最大节点层间位移角
    max_x_abs_story_drift, max_y_abs_story_drift, node_with_max_x_abs_story_drift, node_with_max_y_abs_story_drift = find_max_story_drift(
        abs_story_drift)  # 筛选最大节点绝对层间位移角
    max_G11, frames_with_max_G11, max_G12, frames_with_max_G12, max_G13, frames_with_max_G13 = find_max_coordinates(
        results)  # 筛选最大内力构件

    result = {
        "maximum_G11": {
            "value": max_G11,
            "frames": frames_with_max_G11
        },
        "maximum_G12": {
            "value": max_G12,
            "frames": frames_with_max_G12
        },
        "maximum_G13": {
            "value": max_G13,
            "frames": frames_with_max_G13
        },
        "maximum_X_story_drift": {
            "value": max_x_story_drift,
            "node": node_with_max_x_story_drift
        },
        "maximum_Y_story_drift": {
            "value": max_y_story_drift,
            "node": node_with_max_y_story_drift
        },
        "maximum_X_abs_story_drift": {
            "value": max_x_abs_story_drift,
            "node": node_with_max_x_abs_story_drift
        },
        "maximum_Y_abs_story_drift": {
            "value": max_y_abs_story_drift,
            "node": node_with_max_y_abs_story_drift
        },
        "total_weight": {
            "value": total_weight,
        }
    }

    # 将结果写入 JSON 文件
    # with open(output_file, 'w') as json_file:
    with open(os.path.join(output_file, 'max_values.json'), 'w') as json_file:
        json.dump(result, json_file, indent=4)

    return None
