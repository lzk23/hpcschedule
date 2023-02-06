import imp
import math
import os
import sys
import time
from hpc import hpcproblem
from config import config

if __name__ == '__main__':
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, config.workload)

    if config.DRL == False:
        problem = hpcproblem.HPCInstance(workload_file=workload_file)
        max_ep = 10
        max_time_steps = 300
        if config.schedule_type == 'Batch':
            from simanneal.test_sa import test2
            test2(max_ep, max_time_steps)
        else:
            start_time_step = 0
            if config.schedule_type == 'FCFS':
                fw = open('simanneal/comm_wait_cost_FCFS.txt', 'w')
            else:
                fw = open('simanneal/comm_wait_cost_BF.txt', 'w')
            for ep in range(1, max_ep+1):
                if config.schedule_type == 'FCFS':
                    problem.schedule_curr_sequence_reset(problem.fcfs_score, False, start_time_step, max_time_steps)
                else:
                    problem.schedule_curr_sequence_reset(problem.fcfs_score, True, start_time_step, max_time_steps)
                assert len(problem.scheduled_logs.values()) == max_time_steps
                ave_comm, ave_wait = problem.deal_logs()
                fw.write("ep:{}, comm. cost: {}, waiting. cost: {}\n".format(ep, int(ave_comm), int(ave_wait)))
                start_time_step += max_time_steps            
        print("completed!")
    
    else:
        if config.train:
            if config.usePPO:
                from PPO.train import train
                train()
            elif config.usePointer:
                from Pointer.train import train
                train()
        elif not config.test_with_SA:
            from PPO.test import test
            test()
        elif config.test_with_SA:
            from simanneal.test_sa import test
            test()

class Temp:
    def __init__(self, arg) -> None:
        self.conf = config(arg)
        

    

    