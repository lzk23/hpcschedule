from copy import deepcopy
from config import *
import numpy as np
from hpc.cluster import ClusterWithTopology
from itertools import chain, combinations


def dpalg(jobs, cluster: ClusterWithTopology, temp_can_not_use_node_ids):
    machines = cluster.machines
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

    S = chain.from_iterable(combinations(range(nb_jobs), i) for i in range(1, nb_jobs))
    
    S = list(map(list, S))

    S_list = list(map(str, S))
    S_dic = {}

    for i in range(len(S_list)):
        S_dic[','.join(S_list[i])] = i

    subset_num = len(S)

    f = np.zeros((nb_can_use_machines, nb_can_use_machines, subset_num))
    f.fill(10000000)

    jobs_request_list = []
    for job in jobs:
        if not jobs_request_list.__contains__(job.request_number_of_nodes):
            jobs_request_list.append(job.request_number_of_nodes)

    jobs_request_list.sort() # increase

    request_number_of_nodes_kind = len(jobs_request_list)

    cost_mat = np.zeros((nb_can_use_machines, request_number_of_nodes_kind))

    for i in range(nb_can_use_machines):
        for k in range(request_number_of_nodes_kind):
            node_ids = []
            for j in range(i, i+request_number_of_nodes_kind[k]):
                node_ids.append(model_cluster_node_ids_dic[i])
            cost_mat[i, k] = cluster.calculate_total_distance_nodes(node_ids=node_ids)

    for m1 in range(nb_can_use_machines - jobs_request_list[0]):
        for m2 in range(m1 + jobs_request_list[0], nb_can_use_machines):
            for job_index in range(nb_jobs):
                min_value = 10000000
                s_index = S_dic[str(job_index)]
                for m3 in range(m1, m2):
                    if cost_mat[m3, jobs[job_index].request_number_of_nodes] < min_value:
                        min_value = cost_mat[m3, jobs[job_index].request_number_of_nodes] 
  
    for m1 in range(nb_can_use_machines - jobs_request_list[0]):
        for m2 in range(m1 + jobs_request_list[0], nb_can_use_machines):
            for s_index in range(subset_num):
                min_value = 10000000
                for m3 in range(m1, m2):
                    for job_index in S[s_index]:
                        s_temp = deepcopy(S[s_index])
                        s_index1 = S_dic[','.join(s_temp.remove(job_index))]
                        s_index2 = S_dic[str(job_index)]
                        v = f[m1, m3-1, s_index1] + f[m3, m2, s_index2]
                        if v < min_value:
                            min_value = v
                f[m1, m2, s_index] = min_value


    
