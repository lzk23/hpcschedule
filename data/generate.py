import numpy as np
import sys 
import os
# sys.path.append("..") 
# from config import config

cluster_scale = "large"

if cluster_scale == "middle":
    arrival_time_max = 30
    arrival_time_min = 5
    run_time_max = 240
    run_time_min = 5
    request_node_max = 20
    request_node_min = 1

if cluster_scale == "large":
    arrival_time_max = 30
    arrival_time_min = 5
    run_time_max = 1800
    run_time_min = 10
    request_node_max = 40
    request_node_min = 1


def generation():
    jobs_num = 100000
    save_file_name = 'data/data_'+cluster_scale+".txt"
    # mean_interval_arrval_time = 10 # /s
    # mean_run_time = 120 #/s
    
    f = open(save_file_name, mode='w')

    # f.write('id, arrival time, run time, request node\n')    

    id = 1
    begin_time = 0
    for job_index in range(jobs_num):
        arrival_time = begin_time + np.random.randint(arrival_time_min, arrival_time_max+1)
        begin_time = arrival_time
        run_time = np.random.randint(run_time_min, run_time_max+1)
        request_node = np.random.randint(request_node_min, request_node_max+1)

        f.write('{}  {}  {}  {}\n'.format(id,arrival_time,run_time,request_node))

        id += 1

    f.close()
    print("succeed!")

if __name__ == '__main__':
    generation()