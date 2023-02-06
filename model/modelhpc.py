import os
from platform import machine
import time
from tkinter import N
from tkinter.messagebox import NO
from ortools.linear_solver import pywraplp
from config import *
import numpy as np
from hpc.cluster import ClusterWithTopology
import sys

def build_and_solve(jobs, cluster: ClusterWithTopology, temp_can_not_use_node_ids, timelimit, enableoutput):
    # temp = sys.stdout
    # sys.stdout = open('simanneal/SCIP_solving.log')
    start_time = time.time()
    machines = cluster.machines
    model = pywraplp.Solver.CreateSolver('SCIP')
    # model.SuppressOutput()
    if enableoutput:
        model.EnableOutput()
    can_use_machines_ids = []
    cluster_model_node_ids_dic = {}
    model_cluster_node_ids_dic = {}
    nb_jobs = len(jobs)   
    node_id_in_model = 0
    for machine in machines:
        if machine.state == Machine_State.idle_state:
            can_use_machines_ids.append(machine.id)
            cluster_model_node_ids_dic[machine.id] = node_id_in_model
            model_cluster_node_ids_dic[node_id_in_model] = machine.id
            node_id_in_model += 1
            
    nb_can_use_machines = len(can_use_machines_ids)
    
    # x = np.zeros((nb_jobs, nb_can_use_machines))
    # y = np.zeros((nb_jobs, nb_can_use_machines))
    # z = np.zeros((nb_can_use_machines))

    x = [[None]*nb_can_use_machines for i in range(nb_jobs)]
    y = [[None]*nb_can_use_machines for i in range(nb_jobs)]
    z = [None] * nb_jobs

    # x = np.array(nb_jobs, nb_can_use_machines)
    # y = np.array(nb_jobs, nb_can_use_machines)
  
    # 定义变量
    for j in range(nb_jobs):
        for i in range(nb_can_use_machines):
            x[j][i] = model.IntVar(0.0, 1.0, 'x['+ str(j)+','+ str(i)+']')
        
    for j in range(nb_jobs):
        for i in range(nb_can_use_machines):
            y[j][i] = model.IntVar(0.0, 1.0, 'y['+ str(j)+','+ str(i) +']')
            
    for j in range(nb_jobs):
        z[j] = model.IntVar(0.0, 1.0, 'z['+ str(j) +']')

    # initial solution

    # 添加约束
    for j in range(nb_jobs):
        expression = 0
        for i in range(nb_can_use_machines):
            expression += y[j][i]
        model.Add(expression == z[j])
        
    for i in range(nb_can_use_machines):
        expression = 0
        for j in range(nb_jobs):
            expression += x[j][i]
        model.Add(expression <= 1)
    
    for j in range(nb_jobs):
        expression = 0
        for i in range(nb_can_use_machines):
            expression += x[j][i]
        model.Add(expression == jobs[j].request_number_of_nodes * z[j]) 
    
    cost_y = [[0] * nb_can_use_machines for j in range(nb_jobs)]
    for j in range(nb_jobs):
        for i in range(nb_can_use_machines):
            node_id_in_cluster = model_cluster_node_ids_dic[i]
            if temp_can_not_use_node_ids != None and temp_can_not_use_node_ids.__contains__(node_id_in_cluster):
                continue
            if config.method_find_nodes == 0:
                next_select_node_ids = cluster.find_most_adjacent_nodes(node_id_in_cluster, jobs[j].request_number_of_nodes, temp_can_not_use_node_ids)
            else:
                next_select_node_ids = cluster.find_most_adjacent_nodes_by_seq(node_id_in_cluster, jobs[j].request_number_of_nodes, temp_can_not_use_node_ids)
            # next_select_node_ids = cluster.find_most_adjacent_nodes_inline(node_id_in_cluster, jobs[j].request_number_of_nodes, temp_can_not_use_node_ids, idel_node_ids_in_line)
            assert len(next_select_node_ids) == jobs[j].request_number_of_nodes
            cost_y[j][i] = cluster.calculate_total_distance_nodes(next_select_node_ids)
            for k1 in next_select_node_ids:
                model.Add(y[j][i] <= x[j][cluster_model_node_ids_dic[k1]])
            
            # next_select_node_ids = cluster.find_most_adjacent_nodes(node_id_in_cluster, jobs[j].request_number_of_nodes, temp_can_not_use_node_ids)
            # assert len(next_select_node_ids) == jobs[j].request_number_of_nodes
            # cost_y[j][i] = cluster.calculate_total_distance_nodes(next_select_node_ids)
            # for k1 in next_select_node_ids:
            #     model.Add(y[j][i] <= x[j][cluster_model_node_ids_dic[k1]])

            # select_all_nodes_by_the_current_node = [model_cluster_node_ids_dic[i]]
            # for k in range(jobs[j].request_number_of_nodes-1):
            #     find_node_id = cluster.find_most_adjacent_nodes_to_sets_of_nodes(select_all_nodes_by_the_current_node, 1, temp_can_not_use_node_ids) 
            #     assert not select_all_nodes_by_the_current_node.__contains__(find_node_id)           
            #     select_all_nodes_by_the_current_node.append(find_node_id)
            # assert len(select_all_nodes_by_the_current_node) == jobs[j].request_number_of_nodes
            # cost_y[j][i] = cluster.calculate_total_distance_nodes(select_all_nodes_by_the_current_node)
            # for k1 in select_all_nodes_by_the_current_node:
            #     model.Add(y[j][i] <= x[j][cluster_model_node_ids_dic[k1]])
          
    for j in range(nb_jobs):
        for i in range(nb_can_use_machines):
            model.Add(x[j][i] <= z[j])
    
    if temp_can_not_use_node_ids != None:
        for node_ids in temp_can_not_use_node_ids:
            node_id_in_model = cluster_model_node_ids_dic[node_ids]
            for j in range(nb_jobs):
                model.Add(x[j][node_id_in_model] == 0)
    
    # 目标
    expression = 0
    for j in range(nb_jobs):
        for i in range(nb_can_use_machines):
            expression += y[j][i] * cost_y[j][i]
                
    for j in range(nb_jobs):
        expression += (1-z[j])*config.cancel_job_cost


    if not timelimit == -1:
        # model.set_time_limit(int(timelimit))
        # model.SetTimeLimit(int(timelimit*1000))
        model.SetTimeLimit(int(config.timelimitvalue*1000 - (time.time() - start_time)))
    # else:
    #     model.SetTimeLimit(60*1000)

    model.Minimize(expression)

    # print(os.getcwd())
    # f1 = open("model/model.lp",'r+', encoding='UTF-8') # 用r打开，会有not writable的问题
    # f1.write(model.ExportModelAsLpFormat(False))
    # f1.close()
    
    status = model.Solve()
    all_job_assign_nodes_dic = {}
    # 结果输出
    if status == model.OPTIMAL or status == model.FEASIBLE:
        # for i in range(data['variable_number']):
        #     print(x[i].name(), '=', x[i].solution_value())
        cancel_job_ids = []
        for j in range(nb_jobs):
            job_assign_node_ids = []              
            for i in range(nb_can_use_machines):
                if x[j][i].solution_value() >= 0.9:
                    job_assign_node_ids.append(model_cluster_node_ids_dic[i])
            if z[j].solution_value() >= 0.9:
                assert jobs[j].request_number_of_nodes == len(job_assign_node_ids)
            else:
                assert len(job_assign_node_ids) == 0
                cancel_job_ids.append(j)

            all_job_assign_nodes_dic[j] = job_assign_node_ids

        # print('objective=', model.Objective().Value())
        # print('problem solved in %f ms' % model.wall_time())
        # print('problem solved in %d iterations' % model.iterations())
        # print('problem solved in %d branch-and-bound node' % model.nodes())
    else:
        print('problem have no optimal solution')
    duration = time.time() - start_time
    is_optimal = False
    if status == model.OPTIMAL:
        is_optimal = True
    # sys.stdout = temp
    # return all_job_assign_nodes_dic, nb_can_use_machines, model.wall_time()/1000, cancel_job_ids
    return all_job_assign_nodes_dic, nb_can_use_machines, duration, cancel_job_ids, is_optimal

        
    
        
    

        
    
            
    
    
            
        
    
    
    
    