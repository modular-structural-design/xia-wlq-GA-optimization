import copy
import json
import numpy as np
import os
import sys
import comtypes.client
import math as m
import json
import utils as ut

with open('config.json', 'r') as f:
    analysis_data = json.load(f)

sap_dirpath = analysis_data["file_paths"]["sap_dirpath"]  ####该地址、
# analysis_model_path = os.path.join(os.getcwd(), "FEM_sap2000")


def sap2000_initialization(model_file_path):
    # SAP initialization
    ## 1. SAP initialization
    sap_model_file = os.path.join(model_file_path, 'FEM_sap2000')
    AttachToInstance = False
    SpecifyPath = True
    if not os.path.exists(sap_model_file):
        try:
            os.makedirs(sap_model_file)
        except OSError:
            pass
    # ModelPath = os.path.join(model_file_path, "API_1-001.sdb")
    # ModelPath = os.path.join(model_file_path, "API_1-001.sdb")
    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
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
                mySapObject = helper.CreateObject(sap_dirpath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from" + sap_dirpath)
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
    ret = SapModel.File.NewBlank()
    # switch units
    N_mm_C = 9
    ret = SapModel.SetPresentUnits(N_mm_C)
    return SapModel, mySapObject


def sap2000_initialization_mulit(model_file_path):
    # SAP initialization
    ## 1. SAP initialization
    sap_model_file = os.path.join(model_file_path, 'FEM_sap2000')
    AttachToInstance = False
    SpecifyPath = True
    if not os.path.exists(sap_model_file):
        try:
            os.makedirs(sap_model_file)
        except OSError:
            pass
    # ModelPath = os.path.join(model_file_path, "API_1-001.sdb")
    # ModelPath = os.path.join(model_file_path, "API_1-001.sdb")
    helper = comtypes.client.CreateObject("SAP2000v1.Helper")
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
                mySapObject = helper.CreateObject(sap_dirpath)
            except (OSError, comtypes.COMError):
                print("Cannot start a new instance of the program from" + sap_dirpath)
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
    modef_path1= copy.deepcopy(model_file_path)
    sap_model_file = os.path.join(modef_path1, 'FEM_sap2000\\MiC1.sdb')
    if not os.path.exists(os.path.dirname(sap_model_file)):
        os.makedirs(os.path.dirname(sap_model_file))

    return SapModel,sap_model_file, mySapObject



def FEM_properties_dataset(SapModel, semantic_list):
    # switch units
    ret = SapModel.File.NewBlank()
    N_mm_C = 9
    ret = SapModel.SetPresentUnits(N_mm_C)
    # materials
    Q345 = semantic_list["materials"]["Q345"]
    steel_Q345_name = "Q345"
    ret = SapModel.PropMaterial.SetMaterial(steel_Q345_name, 1)
    ret = SapModel.PropMaterial.SetWeightAndMass(steel_Q345_name, 2, Q345["density"])
    ret = SapModel.PropMaterial.SetMPIsotropic(
        steel_Q345_name, Q345["E"], Q345["poisson_ratio"], Q345["thermal_coefficient"]
    )
    ret = SapModel.PropMaterial.SetOSteel_1(
        steel_Q345_name,
        Q345["fy"],
        Q345["fu"],
        Q345["Fye"],
        Q345["Fue"],
        Q345["SSType"],
        Q345["SSHysType"],
        Q345["StrainAtHardening"],
        Q345["StrainAtMaxStress"],
        Q345["StrainAtRupture"],
        Q345["FinalSlope"]
    )
    steel_fy = Q345["fy"]

    ## 3. Cross sections
    C_type_section = semantic_list["section_types"]["Channel"]
    I_type_section = semantic_list["section_types"]["I-type"]
    Rect_type_section = semantic_list["section_types"]["Rect"]
    material_name = steel_Q345_name
    # default Channel-0 & Q345
    for i in range(len(C_type_section)):
        section_name = "Channel" + str(i)
        section_data = C_type_section[str(i)]
        ret = SapModel.PropFrame.SetChannel(
            section_name,
            material_name,
            section_data["outside_depth"],
            section_data["outside_flange_width"],
            section_data["flange_thickness"],
            section_data["web_thickness"],
            -1,
        )
    for i in range(len(I_type_section)):
        section_name = "I-type" + str(i)
        section_data = I_type_section[str(i)]
        ret = SapModel.PropFrame.SetISection(
            section_name,
            material_name,
            section_data["height"],
            section_data["width"],
            section_data["tf"],
            section_data["tw"],
            section_data["width"],
            section_data["tf"],
            -1,
        )
    for i in range(len(Rect_type_section)):
        section_name = "Rect" + str(i)
        section_data = Rect_type_section[str(i)]
        ret = SapModel.PropFrame.SetTube(
            section_name,
            material_name,
            section_data["height"],
            section_data["width"],
            section_data["flange_thickness"],
            section_data["web_thickness"],
            -1,
        )

    ## 4. connection properties
    basic_properties = semantic_list["inter_connection_properties"]["basic_properties"]
    Horizontal = semantic_list["inter_connection_properties"]["Horizontal"]
    Vertical = semantic_list["inter_connection_properties"]["Vertical"]

    ret = SapModel.PropLink.SetMultiLinearElastic(
        "HOR1",
        basic_properties["Dofs"],
        basic_properties["DofFix"],
        basic_properties["NonLinear"],
        basic_properties["Ke"],
        basic_properties["Ce"],
        2,
        0,
    )
    ret = SapModel.PropLink.SetMultiLinearElastic(
        "VER1",
        basic_properties["Dofs"],
        basic_properties["DofFix"],
        basic_properties["NonLinear"],
        basic_properties["Ke"],
        basic_properties["Ce"],
        2,
        0,
    )

    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 1, 5, Horizontal["RF_u1"], Horizontal["RD_u1"])
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 2, 5, Horizontal["RF_u2"], Horizontal["RD_u2"])
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 3, 5, Horizontal["RF_u3"], Horizontal["RD_u3"])
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 4, 7, Horizontal["RF_r1"], Horizontal["RD_r1"])
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 5, 7, Horizontal["RF_r2"], Horizontal["RD_r2"])
    ret = SapModel.PropLink.SetMultiLinearPoints("HOR1", 6, 7, Horizontal["RF_r3"], Horizontal["RD_r3"])

    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 1, 5, Vertical["RF_u1"], Vertical["RD_u1"])
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 2, 5, Vertical["RF_u2"], Vertical["RD_u2"])
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 3, 5, Vertical["RF_u3"], Vertical["RD_u3"])
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 4, 7, Vertical["RF_r1"], Vertical["RD_r1"])
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 5, 7, Vertical["RF_r2"], Vertical["RD_r2"])
    ret = SapModel.PropLink.SetMultiLinearPoints("VER1", 6, 7, Vertical["RF_r3"], Vertical["RD_r3"])

    # 5. plane sections
    ret = SapModel.PropArea.SetShell_1("Plane0", 1, True, "4000Psi", 0, 0, 0)

    return SapModel


def FEM_member_modelling(SapModel, FEM_info, modular_FEM):
    nodes_geo = FEM_info["nodes_geo"]
    frames_index = FEM_info["frames_index"]
    plane_index = FEM_info["plane_index"]
    inter_connections_index = FEM_info["inter_connections_index"]
    section_name = 'Rect10'
    frames_sections = FEM_info["frames_sections"]
    inter_connections_types = FEM_info["inter_connections_type"]

    for node_name, value in nodes_geo.items():
        x = value[0]
        y = value[1]
        z = value[2]
        ret = SapModel.PointObj.AddCartesian(x, y, z, None, node_name, "Global")

    for frame_name, value in frames_index.items():
        indx1 = value[0]
        indx2 = value[1]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        frame_name = frame_name
        tp_frame_section = frames_sections[frame_name]
        tp = modular_FEM[tp_frame_section['modular_type']]['sections']
        tp = tp[tp_frame_section['edge_type']]
        tp = "Rect" + str(tp)
        ret = SapModel.FrameObj.AddByPoint(Point1, Point2, " ", tp, frame_name)

    for connection_name, value in inter_connections_index.items():
        indx1 = value[0]
        indx2 = value[1]
        Point1 = "nodes" + str(indx1)
        Point2 = "nodes" + str(indx2)
        location_type = inter_connections_types[connection_name]
        if int(location_type) == 1:
            ret = SapModel.LinkObj.AddByPoint(Point1, Point2, connection_name, False, "HOR1")
        else:
            ret = SapModel.LinkObj.AddByPoint(Point1, Point2, connection_name, False, "VER1")

    for plane_name, value in plane_index.items():
        indx1 = value[0]
        indx2 = value[1]
        indx3 = value[2]
        indx4 = value[3]

        Point1 = nodes_geo['nodes' + str(indx1)]
        Point2 = nodes_geo['nodes' + str(indx2)]
        Point3 = nodes_geo['nodes' + str(indx3)]
        Point4 = nodes_geo['nodes' + str(indx4)]
        x = [Point1[0], Point2[0], Point3[0], Point4[0]]
        y = [Point1[1], Point2[1], Point3[1], Point4[1]]
        z = [Point1[2], Point2[2], Point3[2], Point4[2]]
        ret = SapModel.AreaObj.AddByCoord(4, x, y, z, "Plane0", "Default", plane_name, "Global")

    return SapModel


def FEM_boundary(SapModel, model_info):
    node_names = model_info["boundary_nodes"]
    res1 = [True, True, True, True, True, True]
    for name in node_names:
        ret = SapModel.PointObj.setRestraint(name, res1)
    return SapModel


def FEM_loading(SapModel, model_info):
    load_patterns = model_info["load_patterns"]
    for key, value in load_patterns.items():
        ret = SapModel.LoadPatterns.Add(key, int(value))

    frame_loads = model_info["frame_loads"]
    for frame_name, values in frame_loads.items():
        ret = SapModel.FrameObj.SetLoadDistributed(frame_name, values['LoadPat'], int(values['MyType']),
                                                   int(values['Dir']), float(values['Dist1']), float(values['Dist2']),
                                                   float(values['Val1']), float(values['Val2']))

    plane_loads = model_info["plane_loads"]
    for plane_name, loads in plane_loads.items():
        for key, value in loads.items():
            ret = SapModel.AreaObj.SetLoadUniformToFrame(plane_name, value['LoadPat'], float(value['Value']),
                                                         int(value['Dir']), int(value['DistType']),
                                                         bool(value['Replace']), value['CSys'])

    seismic_info = model_info["seismic_info"]
    for key, values in seismic_info.items():
        case_name = values['case_name']
        direction = int(values['direction'])
        eccen = float(values['eccen'])
        PeriodFlag = 2
        UserT = 0
        UserZ = False
        TopZ = 0
        BottomZ = 0
        AlphaMax = float(values['AlphaMax'])
        Si = int(values['SI'])
        DampRatio = float(values['DampRatio'])
        Tg = float(values['Tg'])
        TDF = float(values['TDF'])
        EnhanceFactor = float(values['EnhanceFactor'])
        ret = SapModel.LoadPatterns.Add(case_name, 5)
        ret = SapModel.LoadPatterns.AutoSeismic.SetChinese2002(
            case_name, direction, eccen, PeriodFlag, UserT, UserZ,
            TopZ, BottomZ, AlphaMax, Si, DampRatio, Tg, TDF, EnhanceFactor)
        ret = SapModel.LoadCases.StaticLinear.SetLoads(case_name, 1, "Load", case_name, [0.01])

    loadcomb_info = model_info["load_combinations"]
    for key, value in loadcomb_info.items():
        ret = SapModel.RespCombo.Add(key, 0)
        for i in range(len(value[0])):
            ret = SapModel.RespCombo.SetCaseList(key, 0, value[0][i], value[1][i])

    return SapModel


#
def out_put_reaction(SapModel, frames):
    name_re = []
    frame_reactions = []
    frame_reactions_all = []
    for edge_indx in range(len(frames)):
        result = []
        P_na = []
        mm1 = np.zeros((7, 3))
        mm2 = []
        Obj, ObjSta, P, V2, V3, T, M2, M3 = get_frame_reactions("frame" + str(edge_indx), SapModel)
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


def get_frame_reactions(frames, SapModel):
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
     ret] = SapModel.Results.FrameForce(frames, Object11, NumberResults, Obj, ObjSta, Elm, ElmSta, LoadCase, StepType,
                                        StepNum, P, V2, V3, T, M2, M3)
    return Obj, ObjSta, P, V2, V3, T, M2, M3


def out_put_displacement(Nodes, SapModel):
    displacements = []
    displacements_hor = []
    name_all_nodes = []
    for i in range(len(Nodes)):
        result = []
        Obj, U1, U2, U3, R1, R2, R3 = get_point_displacement("nodes" + str(i), SapModel)
        if len(U1) != 0:
            name_all_nodes.append(Obj[0])
            result.append(U1[0])
            result.append(U2[0])
            result.append(U3[0])
            # result.append(R1[0])
            # result.append(R2[0])
            # result.append(R3[0])
            displacements.append(result)
            displacements_hor.append(m.sqrt(U1[0] ** 2 + U2[0] ** 2))
    displacements = np.array(displacements)

    return displacements


def get_point_displacement(nodes, SapModel):
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
    [NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3, ret] = SapModel.Results.JointDispl(
        nodes, ObjectElm, NumberResults, Obj, Elm, ACase, StepType, StepNum, U1, U2, U3, R1, R2, R3)
    return Obj, U1, U2, U3, R1, R2, R3


def output_data(SapModel, FEA_info2, data_file_path):
    '''

    :param SapModel: sap2000运行后的模型
    :param FEA_info2: data_case1.json中读取的信息
    :return: dict{node_dis_dict节点位移信息,frame_reaction_dict构件作用力信息}
            node_dis_dict：{节点名：列表[x,y,z]}
            frame_reaction_dict:{构件名：列表[[构件的距离]，[轴力]，[2-2剪力]，[3-3剪力]，[扭矩]，[2-2弯矩]，[3-3弯矩]]}
    '''
    comb_name = list(FEA_info2['load_combinations'].keys())[0]
    ret = SapModel.Results.Setup.DeselectAllCasesAndCombosForOutput()
    ret = SapModel.Results.Setup.SetComboSelectedForOutput(comb_name)

    frame_reaction = out_put_reaction(SapModel, FEA_info2['frames_index'])
    node_info = out_put_displacement(FEA_info2['nodes_geo'], SapModel)

    node_dis_dict = {}
    frame_reaction_dict = {}
    for i in range(len(node_info)):
        node_dis_dict["nodes" + str(i)] = node_info[i].tolist()
        frame_reaction_dict["frame" + str(i)] = frame_reaction[i].tolist()

    all_infor = [node_dis_dict, frame_reaction_dict]
    json_str = json.dumps(all_infor)
    with open(os.path.join(data_file_path, 'calculate_data.json'), 'w') as json_file:
        json_file.write(json_str)
    pass


#
def parsing_to_sap2000(total_info: object, FEA_semantic_file: object, modular_FEM: object, model_file_path) -> object:
    with open(FEA_semantic_file, "r") as f:
        semantic_list = json.load(f)

    SapModel, mySapObject = sap2000_initialization(model_file_path)
    SapModel = FEM_properties_dataset(SapModel, semantic_list)
    SapModel = FEM_member_modelling(SapModel, total_info, modular_FEM)
    SapModel = FEM_boundary(SapModel, total_info)
    SapModel = FEM_loading(SapModel, total_info)

    ####### save and analysis ##########

    sap_model_file = os.path.join(model_file_path, 'FEM_sap2000\\MiC1.sdb')
    if not os.path.exists(os.path.dirname(sap_model_file)):
        os.makedirs(os.path.dirname(sap_model_file))
    ret = SapModel.File.Save(sap_model_file)
    ret = SapModel.Analyze.RunAnalysis()

    ## output analysis data###
    output_data(SapModel, total_info, model_file_path)

    ######## close sap2000 ############
    ret = mySapObject.ApplicationExit(False)
    SapModel = None
    mySapObject = None

    # pass
    return None

def parsing_to_sap2000_mulit(total_info: object, FEA_semantic_file: object, modular_FEM: object, model_file_path,SapModel, mySapObject,sap_model_file) -> object:
    with open(FEA_semantic_file, "r") as f:
        semantic_list = json.load(f)

    modef_path1 = copy.deepcopy(model_file_path)
    modef_path2 = copy.deepcopy(model_file_path)
    # SapModel, mySapObject = sap2000_initialization(model_file_path)
    SapModel = FEM_properties_dataset(SapModel, semantic_list)
    SapModel = FEM_member_modelling(SapModel, total_info, modular_FEM)
    SapModel = FEM_boundary(SapModel, total_info)
    SapModel = FEM_loading(SapModel, total_info)

    ####### save and analysis ##########

    # sap_model_file = os.path.join(modef_path1, 'FEM_sap2000\\MiC1.sdb')
    # if not os.path.exists(os.path.dirname(sap_model_file)):
    #     os.makedirs(os.path.dirname(sap_model_file))
    ret = SapModel.File.Save(sap_model_file)
    ret = SapModel.Analyze.RunAnalysis()

    ## output analysis data###
    output_data(SapModel, total_info, modef_path2)
    ret = SapModel.SetModelIsLocked(False)
    ######## close sap2000 ############
    # ret = mySapObject.ApplicationExit(False)
    # SapModel = None
    # mySapObject = None

    # pass
    return None
