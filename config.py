import argparse
class config:
    parser = argparse.ArgumentParser(description='HPCSchduler')
    # for HPC job scheduler
    parser.add_argument('--schedule_type', type=str, default='Batch', choices=['Batch','FCFS','BF'])
    parser.add_argument('--model_name', type=str, default='large')
    parser.add_argument('--time_slot', default=60, type=int)
    parser.add_argument('--cluster_scale', default='large', type=str, choices=['small', 'middle', 'large'])
    parser.add_argument('--plot_cluster', default=False, action='store_true')
    parser.add_argument('--test_name', type=str, default='1800_40')
    # for DRL
    parser.add_argument('--DRL', default=False, action='store_true')
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--rl_run_time', type=int, default = 1)
    parser.add_argument('--PPO', default=False, action='store_true')
    parser.add_argument('--layer_type', type=int, default=0) # 0, full connect, 1, conv2d
    parser.add_argument('--layer_index', type=int, default=0)
    parser.add_argument('--kernal_size', type=int, default=10)
    parser.add_argument('--Pointer', default=False, action='store_true')
    parser.add_argument('--re_scale_state', default=False, action='store_true')
    parser.add_argument('--with_distance_feature', default=False, action='store_true')
    # for SA
    parser.add_argument('--test_with_SA', default=False, action='store_true')
    parser.add_argument('--SA_with_RL', default=False, action='store_true')
    parser.add_argument('--compare', default=False, action='store_true')
    parser.add_argument('--timelimitvalue', type=int, default=30)
    parser.add_argument('--method_find_nodes', type=int, default=1) # while the center node is given, how to find other nodes. 
    parser.add_argument('--stepsforsa', type=int, default=500) # run steps for sa

    args = parser.parse_args()
    schedule_type = args.schedule_type
    model_name = args.model_name
    time_slot = args.time_slot # /s
    cluster_scale = args.cluster_scale # small, middle, large
    plot_cluster = args.plot_cluster
    test_name = args.test_name

    DRL = args.DRL
    train = args.train
    rl_run_time = args.rl_run_time
    usePPO = args.PPO
    usePointer = args.Pointer
    layer_type = args.layer_type
    layer_index = args.layer_index
    kernal_size = args.kernal_size
    re_scale_state = args.re_scale_state
    assert not (usePPO and usePointer)
    if DRL:
        assert (usePPO or usePointer)
    with_distance_feature = args.with_distance_feature
    
    test_with_SA = args.test_with_SA
    SA_with_RL = args.SA_with_RL
    stepsforsa = args.stepsforsa
    timelimitvalue = args.timelimitvalue
    method_find_nodes = args.method_find_nodes
    device = 'cpu'
    
    run_num = 0
    checkpoint_path = "PPO_preTrained/PPO_test{}_method{}_layer{}_index{}_kernal{}_rescale{}_num{}.pth".format(test_name, method_find_nodes, layer_type, layer_index,
                                                                                                 kernal_size, re_scale_state*1, run_num)
    if test_name == '1800_40_2':
        checkpoint_path = "PPO_preTrained/PPO_test{}_method{}_layer{}_index{}_kernal{}_rescale{}_num{}.pth".format('1800_40', method_find_nodes, layer_type, layer_index,
                                                                                                 kernal_size, re_scale_state*1, run_num)
    log_path = "{}/PPO_test{}_method{}_layer{}_index{}_kernal{}_rescale{}_num{}.csv".format("PPO_logs", test_name, method_find_nodes, layer_type, layer_index,
                                                                              kernal_size, re_scale_state*1, run_num)
    

    if args.cluster_scale == 'middle':       
        workload = './data/data_middle.txt'
        model_name = 'middle'

    elif args.cluster_scale == 'large':
        workload = './data/data_large_{}.txt'.format(test_name)
        model_name = 'large'
    

    # for cluster with topology
    x_node_num_each_unit = 10
    x_distance_two_nodes = 0.1
    if cluster_scale == "small":
        x_unit_num = 1    
        y_unit_num = 4
        z_unit_num = 1
        job_max_request = 4  # 每个作业请求的最大节点数
        conv2dW = x_node_num_each_unit * x_unit_num # 10
        conv2dH = y_unit_num * z_unit_num #  4

    
    if cluster_scale == "middle":
        x_unit_num = 2    
        y_unit_num = 4
        z_unit_num = 2
        job_max_request = 20  # 每个作业请求的最大节点数
        conv2dW = x_node_num_each_unit * x_unit_num # 20
        conv2dH = y_unit_num * z_unit_num #  8
        
    
    if cluster_scale == "large":
        x_unit_num = 5    
        y_unit_num = 4
        z_unit_num = 5
        job_max_request = 0  # 每个作业请求的最大节点数, set when read workload
        conv2dW = x_node_num_each_unit * x_unit_num # 50
        conv2dH = y_unit_num * z_unit_num #  20
        if re_scale_state:
            conv2dH_rescale = 5
            conv2dW_rescale = 30
    
    total_node_num = x_unit_num * y_unit_num * z_unit_num * x_node_num_each_unit
    x_distance_between_two_unit = 5
    y_distance_between_two_unit = 1
    z_distance_between_two_unit = 5

    # job
    max_queue_size = 5

    # each action node num
    action_node_num = job_max_request

    compare_with_model = args.compare

    cancel_job_cost = 1000000000

    job_score_type = 5
    
    consider_other_job = 1
    if consider_other_job:
        node_feature_num = 3 + 1 + 1 # containerid, rackid, used or free, current scheduling job at this node, other jobs
    else:
        node_feature_num = 3 + 1   
    
    if args.train:
        if consider_other_job == 0:
            max_queue_size = 1
        else:
            max_queue_size = 2

    if with_distance_feature:
        node_feature_num = 3 + 2 + 1 + 1 # containerid, rackid, distance to left, distance to right, used or free, current scheduling job at this node, other jobs

    if test_name == '3600_20':
        max_nodes_for_drl = 100
    elif test_name == '1800_40' or test_name == '1800_40_2':
        max_nodes_for_drl = 100
    elif test_name == '1200_60':
        max_nodes_for_drl = 100
        
    state_dim = node_feature_num * max_nodes_for_drl

    # if re_scale_state:
    #     state_dim = node_feature_num * max_nodes_for_drl
    if not re_scale_state:
        max_nodes_for_drl = total_node_num
        state_dim = node_feature_num * total_node_num

    if layer_type == 1:
        assert re_scale_state
    

        
    

class Machine_State:
    run_state = 0
    idle_state = 1

