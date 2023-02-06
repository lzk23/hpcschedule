from distutils.log import error
from lib2to3.pytree import Node
from operator import mod
from sched import scheduler
from turtle import update
import math
from hpc.job import Job, Workloads
from hpc.cluster import ClusterWithTopology
import os
import random
from statistics import mean

import numpy as np

import gym
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding
from model.modelhpc import build_and_solve
from config import Machine_State, config 
import model.modelhpc
from copy import deepcopy
import sys


class HPCEnv(gym.Env):
    def __init__(self):
        super(HPCEnv, self).__init__()
        print("=====================================================")
        print("*\t Initialize Fair HPC Env")
        workload = Workloads(config.workload)
        self.all_jobs = workload.all_jobs
        config.job_max_request = workload.max_requested_node
        
        self.cluster = ClusterWithTopology()

        self.action_space = spaces.Discrete(config.max_nodes_for_drl)

        if config.layer_type == 0:
            self.observation_space = spaces.Box(low=0.0, high=1.0,
                                                shape=(config.state_dim,),
                                                dtype=np.float32)
        elif config.layer_type == 1:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(config.node_feature_num, self.cluster.nb_machines), dtype=float)
        
        elif config.layer_type == 3:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(config.node_feature_num, config.conv2dH, config.conv2dW), dtype=float)

        self.job_queue = []
        self.running_jobs = []
        self.need_schedule_by_RL_jobs = []
        self.request_one_node_jobs = []
        self.current_scheduling_job_index_in_need_schedule = -1
        self.scheduled_rl = {}
        self.best_scheduled_rl = {} 
        self.best_cost = np.inf
        self.scheduled_time = 0
        self.mask_for_invalid_action = []
        self.state = []
        # self.idle_node_ids_in_line = []
        self.each_row_rack_idle_nodes = [] # for layer type 3

        self.current_timestamp = 0
        self.start_index = 0
        self.next_arriving_job_idx = 0
        self.start_idx_last_reset = 0
        self.update_current_scheduling_job_index_flag = False   
        self.job_number = len(self.all_jobs)

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        # 5: Average Bounded Slowdown + Resource Utilization (0 + 3)
        self.job_score_type = config.job_score_type
        self.scheduled_logs = {}
        self.time_step = 0
        self.seed(0)
        self.cost_rl = 0
        self.cost_seq = 0
        self.cost_model = 0

    def reset_to_initial_cluster(self, machine_states, idle_num, run_num):
        for node_id in range(self.cluster.nb_machines):
            self.cluster.machines[node_id].state = machine_states[node_id]
        self.cluster.idle_node_num = idle_num
        self.cluster.run_node_num = run_num
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.cluster.reset()
        for job in self.all_jobs:
            job.scheduled_time = -1
            job.allocated_machine_ids = None
        self.job_queue = []
        self.running_jobs = []
        self.need_schedule_by_RL_jobs = []
        self.request_one_node_jobs = []
        self.current_scheduling_job_index_in_need_schedule = -1
        self.scheduled_rl = {}
        self.best_scheduled_rl = {} 
        self.best_cost = np.inf
        self.scheduled_time = 0
        self.each_row_rack_idle_nodes = []

        self.current_timestamp = 0
        self.start_index = 0
        self.next_arriving_job_idx = 0

        self.start_index = self.np_random.randint(0,(self.job_number - 128))
        # print("start_index:", self.start_index)
        # if self.start_idx_last_reset == self.start_index:
        #     print("start at the same index")
        #else:
            # print("start at different start index")
        self.start_idx_last_reset = self.start_index
        self.current_timestamp = self.all_jobs[self.start_index].submit_time + config.time_slot
        self.next_arriving_job_idx = self.start_index + 1
        
        self.mask_for_invalid_action = [Machine_State.idle_state for i in range(config.max_nodes_for_drl)]
        
        self.cost_rl = 0
        self.cost_seq = 0
        self.cost_model = 0
        self.scheduled_logs = {}

        self.build_observation()

    def reset_for_next_time_step(self, update_timestamp = True):
        self.current_scheduling_job_index_in_need_schedule = -1
        if update_timestamp:
            self.current_timestamp += config.time_slot
        self.need_schedule_by_RL_jobs = []
        self.request_one_node_jobs = []
        self.scheduled_rl = {}
        self.best_scheduled_rl = {}
        self.best_cost = np.inf
        self.scheduled_time = 0
        self.cost_rl = 0
        self.cost_seq = 0
        self.cost_model = 0
        self.each_row_rack_idle_nodes = []
        # self.obtain_all_idles_node_ids_before_step()

    def build_observation(self):
        max_containerid, max_rackid = self.cluster.get_max_container_rackid()
        if config.layer_type == 0:
            self.state = np.zeros(config.node_feature_num*self.cluster.nb_machines, dtype=float)            
            for i in range(self.cluster.nb_machines):
                m = self.cluster.machines[i]
                self.state[i*config.node_feature_num:i*config.node_feature_num+2] = [m.containerid/max_containerid, m.rackid/max_rackid]

        elif config.layer_type == 1:
            self.state = np.zeros((1, config.node_feature_num, self.cluster.nb_machines), dtype=float)                   
            for i in range(self.cluster.nb_machines):
                m = self.cluster.machines[i]
                self.state[0,0:2,i] = [m.containerid/max_containerid, m.rackid/max_rackid]

    def job_score(self, job_for_scheduling):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        if self.job_score_type == 0:
            # bsld
            _tmp = max(1.0, (float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
                             /
                             max(job_for_scheduling.run_time, 10)))
        elif self.job_score_type == 1:
            # wait time
            _tmp = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
        elif self.job_score_type == 2:
            # turnaround time
            _tmp = float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time)
        elif self.job_score_type == 3:
            # utilization
            _tmp = -float(job_for_scheduling.run_time * job_for_scheduling.request_number_of_processors)
        elif self.job_score_type == 4:
            # sld
            _tmp = float(
                job_for_scheduling.scheduled_time - job_for_scheduling.submit_time + job_for_scheduling.run_time) \
                   / job_for_scheduling.run_time
        elif self.job_score_type == 5:
            _tmp_1 = self.cluster.calculate_total_distance_nodes(node_ids=job_for_scheduling.allocated_machine_ids)
            _tmp_2 = float(job_for_scheduling.scheduled_time - job_for_scheduling.submit_time)
            return [_tmp_1, _tmp_2]
        else:
            raise NotImplementedError

            # Weight larger jobs.
        # _tmp = _tmp * (job_for_scheduling.run_time * job_for_scheduling.request_number_of_processors)
        return _tmp

    # 尽可能选择越多的作业数量完成分配
    def select_jobs_can_be_allocated(self):
        self.need_schedule_by_RL_jobs = []
        self.request_one_node_jobs = []
        # self.job_queue.sort(key = lambda job: job.request_number_of_nodes)  # 升序排序
        self.job_queue.sort(key = lambda job: [-job.wait_stage_num, job.request_number_of_nodes])
        has_add_node_num = 0
        for job in self.job_queue:
            if has_add_node_num + job.request_number_of_nodes <= self.cluster.idle_node_num:
                if job.request_number_of_nodes != 1:
                    if len(self.need_schedule_by_RL_jobs) < config.max_queue_size:
                        self.need_schedule_by_RL_jobs.append(job)
                    else:
                        break
                else:
                    self.request_one_node_jobs.append(job)
                has_add_node_num += job.request_number_of_nodes
            else:
                break

        # update job queue
        need_schedule_job_num = len(self.request_one_node_jobs) + len(self.need_schedule_by_RL_jobs)
        if len(self.job_queue) > need_schedule_job_num:
            # modified: self.job_queue = self.job_queue[need_schedule_job_num:-1]      
            self.job_queue = self.job_queue[need_schedule_job_num:]
            # 对于没有被调度的作业，放到下一次调度，需要更新wait_stage_num
            for job in self.job_queue:
                job.wait_stage_num += 1
        else:
            self.job_queue = []

    # return whether need to solve by rl
    def need_proceed_by_RL(self, alway_continue = False):
        # print("--------------------------current time {}---------------------:".format(self.current_timestamp))
        # new start
        if self.start_index >= self.job_number and len(self.job_queue) == 0:
            assert self.current_timestamp >= self.all_jobs[-1].submit_time
            return False

        if self.scheduled_rl == {} and not self.best_scheduled_rl:
            if (not alway_continue) or len(self.job_queue) == 0:
                has_remove, has_new_arrive = self.remove_finished_add_new_jobs()
            else:
                has_new_arrive, has_remove = True, True
            if len(self.job_queue) == 0 or (not has_remove and not has_new_arrive):
                self.reset_for_next_time_step()
                return False
            self.select_jobs_can_be_allocated()

            if not self.need_schedule_by_RL_jobs:
                if self.request_one_node_jobs:
                    for job in self.request_one_node_jobs:
                        self.allocate_for_jobs_without_RL(job)
                self.reset_for_next_time_step()
                return False
            # if only one job needs to schedule and its request number of nodes is equal to the idle number of nodes in the cluster
            elif len(self.need_schedule_by_RL_jobs)==1 and self.cluster.idle_node_num == self.need_schedule_by_RL_jobs[0].request_number_of_nodes:
                assert not self.request_one_node_jobs
                self.allocate_for_jobs_without_RL(self.need_schedule_by_RL_jobs[0])  
                self.reset_for_next_time_step()
                return False

            #cself.need_schedule_by_RL_jobs.sort(key = lambda job: job.request_number_of_nodes, reverse=True)  # commented at 2022-6-30
            assert len(self.need_schedule_by_RL_jobs) <= config.max_queue_size

            assert self.current_scheduling_job_index_in_need_schedule == -1
            self.current_scheduling_job_index_in_need_schedule = 0
            # update state
            self.obtain_all_idles_node_ids_before_step()
            if not config.test_with_SA:
                self.update_state(before_step = True)
            self.check()
            return True
        
        else:
            assert self.need_schedule_by_RL_jobs                                  
            return True

    def allocate_for_jobs_without_RL(self, job, duration = 0):
        node_ids_temp = []
        success_flag = False
        for m in self.cluster.machines:
            if m.state == Machine_State.idle_state:
                node_ids_temp.append(m.id)          
                # update mask , put those lines in update state function
                # assert self.mask_for_invalid_action[m.id] == 1
                # self.mask_for_invalid_action[m.id] = 0
            if len(node_ids_temp) == job.request_number_of_nodes:
                success_flag = True
                break
        assert success_flag
        self.cluster.allocate_in_Batch(node_ids_temp)
        job.allocated_machine_ids = node_ids_temp
        job.scheduled_time = self.current_timestamp + duration
        self.running_jobs.append(job)

    # @profile
    def step(self, a, update_state = True):
        assert self.need_schedule_by_RL_jobs
        # update current scheduling job index in visible
        current_scheduling_job = self.need_schedule_by_RL_jobs[self.current_scheduling_job_index_in_need_schedule]
        # rest_number_of_nodes = current_scheduling_job.request_number_of_nodes
        # if self.scheduled_rl.__contains__(current_scheduling_job.id):
        #     rest_number_of_nodes = current_scheduling_job.request_number_of_nodes - len(self.scheduled_rl[current_scheduling_job.id])
        # print("current schduling job:", current_scheduling_job.id, ", rest of request nodes:", rest_number_of_nodes)
        self.update_current_scheduling_job_index_flag = False
        all_jobs_scheduled_flag = False
        assert a < self.cluster.nb_machines and self.cluster.machines[a].state == Machine_State.idle_state
        
        need_schedule_node_num = current_scheduling_job.request_number_of_nodes
        
        # if self.scheduled_rl.__contains__(current_scheduling_job.id):
        #     if current_scheduling_job.request_number_of_nodes - len(self.scheduled_rl[current_scheduling_job.id]) <= config.action_node_num:
        #         need_schedule_node_num = current_scheduling_job.request_number_of_nodes - len(self.scheduled_rl[current_scheduling_job.id])
        # else:
        #     if current_scheduling_job.request_number_of_nodes <= config.action_node_num:
        #         need_schedule_node_num = current_scheduling_job.request_number_of_nodes

        # select_all_node_ids_by_the_current_node = [a]
        # for i in range(need_schedule_node_num-1):
        #     find_node_id = self.cluster.find_most_adjacent_nodes_to_sets_of_nodes(select_all_node_ids_by_the_current_node, 1, []) 
        #     assert not select_all_node_ids_by_the_current_node.__contains__(find_node_id)           
        #     select_all_node_ids_by_the_current_node.append(find_node_id)
        if config.method_find_nodes == 0:
            # select_all_node_ids_by_the_current_node = self.cluster.find_most_adjacent_nodes_inline(a, need_schedule_node_num, [], self.idel_node_ids_in_line)
            select_all_node_ids_by_the_current_node = self.cluster.find_most_adjacent_nodes(a, need_schedule_node_num, [])
        elif config.method_find_nodes == 1:
            select_all_node_ids_by_the_current_node = self.cluster.find_most_adjacent_nodes_by_seq(a, need_schedule_node_num, [])
        # elif config.method_find_nodes == 2:
        #     select_all_node_ids_by_the_current_node = self.cluster.find_most_adjacent_nodes(a, need_schedule_node_num, [])
        
        assert len(select_all_node_ids_by_the_current_node) == need_schedule_node_num
        self.cluster.allocate_in_Batch(select_all_node_ids_by_the_current_node)
        if config.action_node_num == config.job_max_request:
            assert not self.scheduled_rl.__contains__(current_scheduling_job.id)
        if self.scheduled_rl.__contains__(current_scheduling_job.id):
            for node_id in select_all_node_ids_by_the_current_node:
                assert not self.scheduled_rl[current_scheduling_job.id].__contains__(node_id)
                self.scheduled_rl[current_scheduling_job.id].append(node_id)
        else:
            self.scheduled_rl[current_scheduling_job.id] = select_all_node_ids_by_the_current_node
        
        # for node_id in select_all_node_ids_by_the_current_node:
        #     self.mask_for_invalid_action[node_id] = 0

        # if the request nodes of current job have all been satisfied
        assert len(self.scheduled_rl[current_scheduling_job.id]) <= current_scheduling_job.request_number_of_nodes
        if len(self.scheduled_rl[current_scheduling_job.id]) == current_scheduling_job.request_number_of_nodes:
            if self.current_scheduling_job_index_in_need_schedule < len(self.need_schedule_by_RL_jobs) -1:
                self.current_scheduling_job_index_in_need_schedule += 1
                self.update_current_scheduling_job_index_flag = True
            else: # all jobs in self.need_schedul_jobs have been scheduled
                all_jobs_scheduled_flag = True
        
        # update input state
        # update state for nodes
        # update state for jobs
        if self.update_current_scheduling_job_index_flag:
            assert not self.scheduled_rl.__contains__(self.need_schedule_by_RL_jobs[self.current_scheduling_job_index_in_need_schedule])           

        # computer reward       
        reward = 0
        if all_jobs_scheduled_flag: # done
            for value in self.scheduled_rl.values():
                reward -= self.cluster.calculate_total_distance_nodes(value)
            
        else: # not done
            # if self.scheduled_rl.__contains__(current_scheduling_job.id):
            #     node_list_temp = self.scheduled_rl[current_scheduling_job.id]
            #     if len(node_list_temp) >= 2:
            #         # reward -= (self.cluster.calculate_total_distance_nodes(node_list_temp)-
            #         #     self.cluster.calculate_total_distance_nodes(node_list_temp[:-1]))
            #         reward = - self.cluster.calculate_total_distance_nodes(node_list_temp)
            # else:
            reward = 0
        
        # update state
        if not all_jobs_scheduled_flag and update_state:
            self.update_state(before_step = False)

        # ready for next time stamp
        if all_jobs_scheduled_flag:
            self.scheduled_time += 1
            if -reward + 1 < self.best_cost:
                # if self.scheduled_time != 1:
                #     print('update!! from cost', self.best_reward, " to ", -reward)
                self.best_cost = -reward
                self.best_scheduled_rl = deepcopy(self.scheduled_rl)

        return reward, all_jobs_scheduled_flag, False
    
    def schedule_finish_assign_best_solution(self, best_scheduled, duration):
        if not best_scheduled == None:
            for node_ids in best_scheduled.values():
                self.cluster.allocate_in_Batch(node_ids) 

        for job in self.need_schedule_by_RL_jobs:
            job.scheduled_time = self.current_timestamp + duration
            job.allocated_machine_ids = best_scheduled[job.id] if not best_scheduled == None else self.best_scheduled_rl[job.id] 
            self.running_jobs.append(job)
        if self.request_one_node_jobs:
            for job in self.request_one_node_jobs:
                self.allocate_for_jobs_without_RL(job, duration)

        # log
        all_jobs = self.need_schedule_by_RL_jobs + self.request_one_node_jobs
        for job_for_scheduling in all_jobs:
            score = self.job_score(job_for_scheduling)  # calculated reward
            assert not self.scheduled_logs.__contains__(job_for_scheduling.id)
            self.scheduled_logs[job_for_scheduling.id] = score # scheduled_logs uses job as the key     

    def remove_current_schedule(self):
        all_schedule_node_ids = list(np.concatenate(list(self.scheduled_rl.values())))
        self.cluster.release(all_schedule_node_ids)

    def update_state(self, before_step):
        if before_step:
            assert not self.scheduled_rl
            assert self.current_scheduling_job_index_in_need_schedule == 0

        next_schedule_job = self.need_schedule_by_RL_jobs[self.current_scheduling_job_index_in_need_schedule]
        
        start_index = 2
        # update nodes state
        for i in range(self.cluster.nb_machines):
            m = self.cluster.machines[i]
            if config.layer_type == 0:
                self.state[i*config.node_feature_num + start_index] = m.state
            elif config.layer_type == 1:
                self.state[0, start_index, i] = m.state
            elif config.layer_type == 3:
                self.state[0, start_index, m.h, m.w] = m.state
            # update mask for invalid action
            if not config.re_scale_state:
                self.mask_for_invalid_action[i] = m.state

        request_rate_first_job = 0
        if self.scheduled_rl.__contains__(next_schedule_job.id):
            request_rate_first_job = (next_schedule_job.request_number_of_nodes-
            len(self.scheduled_rl[next_schedule_job.id]))/config.job_max_request
        else:
            request_rate_first_job = next_schedule_job.request_number_of_nodes/config.job_max_request

        for i in range(self.cluster.nb_machines):
            m = self.cluster.machines[i]
            if config.layer_type == 0:
                if m.state == Machine_State.idle_state:
                    self.state[i*config.node_feature_num + start_index+1] = request_rate_first_job
                else:
                    self.state[i*config.node_feature_num + start_index+1] = 0
            elif config.layer_type == 1:
                if m.state == Machine_State.idle_state:
                    self.state[0, start_index+1, i] = request_rate_first_job
                else:
                    self.state[0, start_index+1, i] = 0
            elif config.layer_type == 3:
                if m.state == Machine_State.idle_state:
                    self.state[0, start_index+1, m.h, m.w] = request_rate_first_job
                else:
                    self.state[0, start_index+1, m.h, m.w] = 0
                    
        if config.consider_other_job:
            if self.update_current_scheduling_job_index_flag or before_step:
                total_node_num_of_other_jobs = 0
                for job_index in range(self.current_scheduling_job_index_in_need_schedule + 1, len(self.need_schedule_by_RL_jobs)):
                    total_node_num_of_other_jobs += self.need_schedule_by_RL_jobs[job_index].request_number_of_nodes
                
                assert total_node_num_of_other_jobs <= config.job_max_request
                request_rate_other_jobs = total_node_num_of_other_jobs/config.job_max_request
                assert request_rate_other_jobs <= 1
                
                if config.layer_type == 0:
                    for i in range(self.cluster.nb_machines):
                        m = self.cluster.machines[i]
                        if m.state == Machine_State.idle_state:
                            self.state[i*config.node_feature_num + start_index+2] = request_rate_other_jobs
                        else:
                            self.state[i*config.node_feature_num + start_index+2] = 0
                elif config.layer_type == 1:
                    for i in range(self.cluster.nb_machines):
                        m = self.cluster.machines[i]
                        if m.state == Machine_State.idle_state:
                            self.state[0, start_index+2, i] = request_rate_other_jobs
                        else:
                            self.state[0, start_index+2, i] = 0
                elif config.layer_type == 3:
                    for i in range(self.cluster.nb_machines):
                        m = self.cluster.machines[i]
                        if m.state == Machine_State.idle_state:
                            self.state[0, start_index+2, m.h, m.w] = request_rate_other_jobs
                        else:
                            self.state[0, start_index+2, m.h, m.w] = 0

    def remove_finished_add_new_jobs(self):
        # 先移除已完成
        should_keep_run_jobs = []
        has_remove = True
        for job in self.running_jobs:
            if job.scheduled_time + job.run_time > self.current_timestamp:
                should_keep_run_jobs.append(job)
            else:
                self.cluster.release(job.allocated_machine_ids)
                # put those lines in update state function
                # for node_id in job.allocated_machine_ids:
                #     assert self.mask_for_invalid_action[node_id] == 0
                #     self.mask_for_invalid_action[node_id] = 1
                # print("release job:"+str(job.job_id))
        
        if len(self.running_jobs) == len(should_keep_run_jobs):
            has_remove = False
        self.running_jobs = should_keep_run_jobs
      
        # 添加上个时段到达的作业
        has_new_arrive = False
        next_stage_arrive_job_index = 0
        time_span = [self.current_timestamp - config.time_slot, self.current_timestamp]
        
        if self.start_index >= self.job_number:
            return has_remove, False
        
        assert self.all_jobs[self.start_index].submit_time >= self.current_timestamp - config.time_slot       
        assert self.start_index < self.job_number
        for i in range(self.start_index, self.job_number):
            job_scheduling = self.all_jobs[i]
            if job_scheduling.submit_time >= time_span[0] and job_scheduling.submit_time < time_span[1]:
                has_new_arrive = True
                self.job_queue.append(job_scheduling)
            else:
                next_stage_arrive_job_index = i
                break
        if not has_new_arrive:
            a = 0
        if next_stage_arrive_job_index == 0: # has scheduled at last job
            next_stage_arrive_job_index = self.job_number
        self.start_index = next_stage_arrive_job_index
        return has_remove, has_new_arrive
    
    def calculate_seq_cost_by_IP_model(self):
        assert self.cost_seq == 0
        total_allocated_node_ids = []
        idle_node_ids_in_line = self.obtain_all_idles_node_ids_before_step()
        for job in self.need_schedule_by_RL_jobs:
            all_job_assign_nodes_dic, nb_can_use_machines, solve_time, cancel_job_ids = build_and_solve([job], self.cluster, total_allocated_node_ids, idle_node_ids_in_line)
            assert len(cancel_job_ids) == 0
            assert len(all_job_assign_nodes_dic) == 1
            assert len(all_job_assign_nodes_dic[0]) == job.request_number_of_nodes
            if len(self.need_schedule_by_RL_jobs) >= 2:
                for node_id in all_job_assign_nodes_dic[0]:
                    assert not total_allocated_node_ids.__contains__(node_id)
                    total_allocated_node_ids.append(node_id)
            self.cost_seq += self.cluster.calculate_total_distance_nodes(all_job_assign_nodes_dic[0])

    def calculate_cost_by_seq_allocate(self):
        assert self.cost_seq == 0
        total_allocated_node_ids = []
        schedule_seq = {}
        for job in self.need_schedule_by_RL_jobs:
            select_node_ids = self.cluster.find_idle_nodes_by_seq(job.request_number_of_nodes, total_allocated_node_ids)
            assert len(select_node_ids) == job.request_number_of_nodes
            if len(self.need_schedule_by_RL_jobs) >= 2:
                for node_id in select_node_ids:
                    assert not total_allocated_node_ids.__contains__(node_id)
                    total_allocated_node_ids.append(node_id)
            self.cost_seq += self.cluster.calculate_total_distance_nodes(select_node_ids)
            schedule_seq[job.id] = select_node_ids
        return schedule_seq
    
    def calculate_model_cost(self, timelimit, enableoutput):
        assert self.cost_model == 0
        all_job_assign_nodes_dic, nb_can_use_machines, solve_time, cancel_job_ids, is_optimal = build_and_solve(self.need_schedule_by_RL_jobs, self.cluster, [], timelimit=timelimit, enableoutput=enableoutput)
        assert len(all_job_assign_nodes_dic.keys()) == len(self.need_schedule_by_RL_jobs)
        scheduled_solution = {}
        for job_index in range(len(all_job_assign_nodes_dic)):
            current_job = self.need_schedule_by_RL_jobs[job_index]
            if cancel_job_ids.__contains__(job_index):
                # print("job {} is canceled in model".format(self.need_schedule_by_RL_jobs[job_index].id))
                # self.cost_model += config.cancel_job_cost
                continue
            assert len(all_job_assign_nodes_dic[job_index]) == current_job.request_number_of_nodes
            self.cost_model += self.cluster.calculate_total_distance_nodes(all_job_assign_nodes_dic[job_index])
            scheduled_solution[current_job.id] = all_job_assign_nodes_dic[job_index]

        return scheduled_solution, round(solve_time, 3), len(cancel_job_ids), is_optimal

    def compare_with_other_policy(self, duration_NSA, duration_SA, best_cost_NSA, best_cost_SA, f_write, optimal_steps, enableoutput, with_timelimit_scip):
        assert config.compare_with_model
        assert self.cost_model == 0
        assert self.cost_seq == 0

        # all_schedule_node_ids = list(np.concatenate(list(self.best_scheduled_rl.values())))
        # self.cluster.release(all_schedule_node_ids)

        total_request_node_num = self.obtain_total_request_node()
        schedule_seq = self.calculate_cost_by_seq_allocate()

        if not with_timelimit_scip:
            all_job_assign_nodes_dic_, solving_time, canceld_job_num, is_optimal = self.calculate_model_cost(-1, enableoutput)
        else:
            all_job_assign_nodes_dic_, solving_time, canceld_job_num, is_optimal = self.calculate_model_cost(max(config.timelimitvalue, duration_NSA), enableoutput)
        # all_job_assign_nodes_dic_, solving_time, canceld_job_num = self.calculate_model_cost(duration)
        # solving_time, canceld_job_num = 0, 0
        # self.calculate_seq_cost()
        
        if not with_timelimit_scip:          
            print("notimelimitSCIP:job queue:", len(self.job_queue),", number of run state:", self.cluster.run_node_num, ", canceled in model:", canceld_job_num, ", scheduled jobs:", len(self.best_scheduled_rl.keys()), ", total request nodes:", total_request_node_num, ', cost_rl:',self.cost_rl, ', diff_SA_model:', int(best_cost_SA - self.cost_model),
            ", diff_NSA_model:", int(best_cost_NSA-self.cost_model), ", time scip:", solving_time, ", time SA:", duration_SA, ", time NSA:", duration_NSA)
        else:
            print("\ntimelimitSCIP:job queue:", len(self.job_queue),", number of run state:", self.cluster.run_node_num, ", canceled in model:", canceld_job_num, ", scheduled jobs:", len(self.best_scheduled_rl.keys()), ", total request nodes:", total_request_node_num, ', cost_rl:',self.cost_rl, ', diff_SA_model:', int(best_cost_SA - self.cost_model),
            ", diff_NSA_model:", int(best_cost_NSA-self.cost_model), ", time scip:", solving_time, ", time SA:", duration_SA, ", time NSA:", duration_NSA)

            if is_optimal:
                print("notimelimitSCIP: timelimitSCIP is optimal")
        
 
        # print("job queue:", len(self.job_queue),", scheduled jobs:", len(self.best_scheduled_rl.keys()), ", total request nodes:", 
        # total_request_node_num, ', cost_rl:',int(self.cost_rl), ", diff_rl_model:", int(self.cost_rl-self.cost_model))

        # print("job queue:", len(self.job_queue),", scheduled jobs:", len(self.best_scheduled_rl.keys()), ", total request nodes:", 
        # total_request_node_num, ', cost_rl:',int(self.cost_rl), ", diff_rl_model:", int(self.cost_rl-self.cost_seq))
        # idle node num, schedule job num, required num nodes, solving_time
        # machine_states = ''
        # for m in self.cluster.machines:
        #     machine_states += str(m.state) + ','
        if not with_timelimit_scip:
            f_write.write('{},{},{},{},{}\n'.format(self.cluster.idle_node_num, len(self.best_scheduled_rl.keys()), total_request_node_num, solving_time, optimal_steps))
        # refill
        # for node_ids in self.best_scheduled_rl.values():
        #     self.cluster.allocate_in_Batch(node_ids) 
        
        # return int(self.cost_rl-self.cost_seq)
        return all_job_assign_nodes_dic_, self.cost_seq, self.cost_model, solving_time, canceld_job_num, is_optimal

    def plot_cluster(self):
        # scheduled = deepcopy(env.scheduled_rl)
        scheduled = {}
        for job in self.running_jobs:
            if not scheduled.__contains__(job.id):
                scheduled[job.id] = job.allocated_machine_ids
        self.cluster.plot(scheduled) 

    def for_repeat_rl(self, refill_with_best_scheduled, update_state):
        all_schedule_node_ids = list(np.concatenate(list(self.scheduled_rl.values())))
        self.cluster.release(all_schedule_node_ids)
        if refill_with_best_scheduled:
            for node_ids in self.best_scheduled_rl.values():
                self.cluster.allocate_in_Batch(node_ids) 

        self.current_scheduling_job_index_in_need_schedule = 0
        self.scheduled_rl = {}
        if update_state:        
            self.update_state(True)


    ## obtain all idle nodes before step
    def obtain_all_idles_node_ids_before_step(self):
        idle_node_ids_in_line = []
        for m in self.cluster.machines:
            if m.state == Machine_State.idle_state:
                idle_node_ids_in_line.append(m.id)
        return idle_node_ids_in_line
                    
    def check(self):
        run_state_node_num = 0
        for job in self.running_jobs:
            run_state_node_num += job.request_number_of_nodes
        assert run_state_node_num == self.cluster.run_node_num  

    def get_comm_cost_wait_time(self):
        sum_comm = 0
        sum_wait = 0
        length = len(self.scheduled_logs.values())
        for v in self.scheduled_logs.values():
            sum_comm += v[0]
            sum_wait += v[1]
        
        return sum_comm//length, sum_wait//length     
    
    def obtain_total_request_node(self):
        total_request_node_num = 0        
        for job in self.need_schedule_by_RL_jobs:
            total_request_node_num += job.request_number_of_nodes
        return total_request_node_num
    
    def re_scale_state(self):
        idle_node_ids_in_line = self.obtain_all_idles_node_ids_before_step()
        
        if config.layer_type == 0:
            state = np.zeros((config.node_feature_num*config.max_nodes_for_drl), dtype=float)
        elif config.layer_type == 1:
            state = np.zeros((1, config.node_feature_num, config.max_nodes_for_drl), dtype=float)
        elif config.layer_type == 3:
            state = np.zeros((1, config.node_feature_num, config.conv2dH_rescale, config.conv2dW_rescale), dtype=float)
        
        
        # if len(idle_node_ids_in_line) <= config.max_nodes_for_drl:
        #     select_idle_ids = idle_node_ids_in_line
        # else:
        #     # select_idle_ids = random.sample(self.idle_node_ids_in_line, config.max_nodes_for_drl)
        #     select_idle_ids = idle_node_ids_in_line[:config.max_nodes_for_drl]
        idx = 0
        if config.layer_type == 0 or config.layer_type == 1:
            for node_id in idle_node_ids_in_line:
                m = self.cluster.machines[node_id]
                assert m.state == Machine_State.idle_state
                if config.layer_type == 0:
                    state[idx*config.node_feature_num:(idx+1)*config.node_feature_num] = self.state[node_id*config.node_feature_num:(node_id+1)*config.node_feature_num]
                elif config.layer_type == 1:
                    state[0, :, idx] = self.state[0, :, node_id]
                    
                self.mask_for_invalid_action[idx] = Machine_State.idle_state
                idx += 1
                if idx >= config.max_nodes_for_drl:
                    break
            
            for i in range(idx, config.max_nodes_for_drl):
                self.mask_for_invalid_action[i] = Machine_State.run_state
                
        elif config.layer_type == 3:
            self.each_row_rack_idle_nodes = []
            for row in range(config.z_unit_num):
                self.each_row_rack_idle_nodes.append([])
            row_node_num = config.total_node_num/config.z_unit_num
            for node_id in idle_node_ids_in_line:
                row_index = int(node_id/(row_node_num))
                self.each_row_rack_idle_nodes[row_index].append(node_id)
            for row in range(config.z_unit_num):
                # if len(self.each_row_rack_idle_nodes[row]) > config.conv2dW + 10:
                #     self.each_row_rack_idle_nodes[row] = random.sample(self.each_row_rack_idle_nodes[row], config.conv2dW)
                #     self.each_row_rack_idle_nodes[row].sort()
                # else:
                temp_num = min(config.conv2dW_rescale, len(self.each_row_rack_idle_nodes[row]))
                self.each_row_rack_idle_nodes[row] = self.each_row_rack_idle_nodes[row][:temp_num]

                    # self.each_row_rack_idle_nodes[row] = self.each_row_rack_idle_nodes[row][:config.conv2dW]
                
            idx = 0
            for row in range(config.z_unit_num):              
                width = 0
                for node_id in self.each_row_rack_idle_nodes[row]:
                    m = self.cluster.machines[node_id]
                    state[0,:, row, width] = self.state[0, :, m.h, m.w]
                    m.state == Machine_State.idle_state
                    self.mask_for_invalid_action[idx] = Machine_State.idle_state
                    width += 1
                    idx += 1
                for i in range(width, config.conv2dW_rescale):
                    self.mask_for_invalid_action[idx] = Machine_State.run_state
                    idx += 1
        
        assert idx <= config.max_nodes_for_drl
        return state, idle_node_ids_in_line
    
    def get_state(self):
        if config.re_scale_state:
            if self.scheduled_rl == {}:
                state, idle_node_ids_in_line, each_row_rack_idle_nodes = self.re_scale_state()
        else:
            state = env.state
        return state, idle_node_ids_in_line, each_row_rack_idle_nodes
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--workload', type=str, default='./data/lublin_256.swf')  # RICC-2010-2
    args = parser.parse_args()
    current_dir = os.getcwd()
    workload_file = os.path.join(current_dir, args.workload)

    env = HPCEnv(batch_job_slice=100, build_sjf=True)
    env.seed(0)
    env.my_init(workload_file=workload_file, sched_file=workload_file)
