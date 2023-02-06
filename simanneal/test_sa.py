from copy import deepcopy
import os
from sched import scheduler
import time
from datetime import datetime
from tkinter.tix import Tree
import numpy as np
from PPO.PPO import PPO
from config import Machine_State, config
from plot import plot_cluster_with_scheduled_jobs
from simanneal.anneal import Annealer
from hpc.HPCEnv import HPCEnv
from multiprocessing import Pool, Manager

class Instance:
    def __init__(self) -> None:
        self.scheduled_dic = []
        self.cost = []
    def reset(self, scheduled_dic, cost):
        self.scheduled_dic = scheduled_dic
        self.cost = cost

class HPCScheduler(Annealer):
    def __init__(self):
        env_name = "jobscheduling"
        action_std = 0.1            # set same std for action distribution which was used while saving
        K_epochs = 80               # update policy for K epochs
        eps_clip = 0.2              # clip parameter for PPO
        gamma = 0.99                # discount factor

        lr_actor = 0.0003           # learning rate for actor
        lr_critic = 0.001           # learning rate for critic

        #####################################################

        self.env = HPCEnv()

        random_seed = 0             #### set this to load a particular checkpoint trained on random seed
        run_num_pretrained = 0      #### set this to load a particular checkpoint num
        checkpoint_path = config.checkpoint_path
        # state space dimension
        state_dim = self.env.observation_space.shape[0]

        action_dim = self.env.action_space.n

        # initialize a PPO agent
        self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, False, action_std)
        self.ppo_agent.load(checkpoint_path)
        self.need_schedule_by_RL_jobs_before_move = []
        self.current_solution_cost = 0
        

    def move(self): 
        initial_reward = self.E # or initial energy

        assert len(self.need_schedule_by_RL_jobs_before_move) == len(self.env.scheduled_rl.keys())
        self.env.need_schedule_by_RL_jobs = deepcopy(self.need_schedule_by_RL_jobs_before_move)
        # random remove a certain number of jobs
        scheduling_job_num = len(self.env.scheduled_rl.keys())
        need_remove_num = self.env.np_random.randint(1, 3)
        # need_remove_num = 2

        all_schedule_job_ids = list(self.env.best_scheduled_rl.keys())

        if scheduling_job_num <= need_remove_num:
            random_select_job_ids = all_schedule_job_ids
        else:     
            random_select_job_ids = np.random.choice(all_schedule_job_ids, need_remove_num, replace=False) # replace element into set is false

        # for job_id in self.need_schedule_by_RL_jobs:
        #     if not job_id in random_select_job_ids:
        #         self.cluster.allocate_in_Batch(self.scheduled_rl[job_id])
        for job_id in random_select_job_ids:
            self.env.cluster.release(self.env.scheduled_rl[job_id])

        # modify state
        self.env.current_scheduling_job_index_in_need_schedule = 0

        # update self.env.need_schedule_by_RL and self.env.scheduled_rl
        self.env.need_schedule_by_RL_jobs = []
        for job in self.need_schedule_by_RL_jobs_before_move:
            if job.id in random_select_job_ids:
                self.env.scheduled_rl.pop(job.id)
                self.env.need_schedule_by_RL_jobs.append(job)
                        
        self.env.update_state(before_step=False)

        # use rl to re-insert
        scheduled_dic, reward = self.find_one_solution(with_RL=True)
        self.state = scheduled_dic
        self.E = -reward
        assert -reward == self.current_solution_cost
        return self.energy() - initial_reward

    def energy(self):
        # calculate_cost = 0
        # for job_id in self.state.keys():
        #     calculate_cost += self.env.cluster.calculate_total_distance_nodes(self.state[job_id])
        # assert abs(calculate_cost - self.E) < 1
        # assert calculate_cost == self.E
        return self.current_solution_cost
    
    # def insert(self):
    #     action = self.ppo_agent.select_action(self.env.mask_for_invalid_action)
    #     self.env.step(action)

    def find_one_solution(self, with_RL):
        # use rl to re-insert
        while(True):
            if with_RL and config.SA_with_RL:
                # action = self.ppo_agent.select_action(state=self.env.state, mask=self.env.mask_for_invalid_action)
                if config.re_scale_state:
                    state, idle_node_ids_in_line= self.env.re_scale_state()
                else:
                    state = self.env.state
                if config.re_scale_state:
                    action_in_idle_index = self.ppo_agent.select_action(state, self.env.mask_for_invalid_action)
                    action = idle_node_ids_in_line[action_in_idle_index]
                else:
                    action = self.ppo_agent.select_action(state, self.env.mask_for_invalid_action)
            else:
                idle_node_ids_in_line = self.env.obtain_all_idles_node_ids_before_step()               
                # action_in_idle_index = self.env.np_random.randint(0, len(idle_node_ids_in_line))
                action_in_idle_index = self.env.np_random.randint(0, min(len(idle_node_ids_in_line), config.max_nodes_for_drl))
                action = idle_node_ids_in_line[action_in_idle_index]
                
            reward, done, skip = self.env.step(action)
            if done:
                self.current_solution_cost = -reward
                break
        return self.env.scheduled_rl, reward
    
    def find_initial_solution(self):
        cost = 0
        schedule_initial = {}
        total_allocated_node_ids = []
        for job in self.need_schedule_by_RL_jobs_before_move:
            select_node_ids = self.env.cluster.find_idle_nodes_by_seq(job.request_number_of_nodes, total_allocated_node_ids)
            assert len(select_node_ids) == job.request_number_of_nodes
            # if len(self.need_schedule_by_RL_jobs_before_move) >= 2:
            #     for node_id in select_node_ids:
            #         assert not total_allocated_node_ids.__contains__(node_id)
            #         total_allocated_node_ids.append(node_id)
            self.env.cluster.allocate_in_Batch(select_node_ids)
            cost += self.env.cluster.calculate_total_distance_nodes(select_node_ids)
            assert not cost == 0
            schedule_initial[job.id] = select_node_ids
        self.env.scheduled_rl = deepcopy(schedule_initial)
        self.env.best_scheduled_rl = deepcopy(self.env.scheduled_rl)
        self.env.best_cost = cost 
        self.current_solution_cost = cost
        return schedule_initial, -cost


#################################### Testing ###################################
def test():
    print("============================================================================================")
    total_test_episodes = 10   # total num of testing episodes
    max_ep_len = 100           # max timesteps in one episode
    
    scheduler = HPCScheduler()

    save_file_name = 'simanneal/compare_test{}_layer{}_index{}_method{}_rescale{}_step{}.txt'.format(config.test_name, config.layer_type, config.layer_index,
                                                                                         config.method_find_nodes, config.re_scale_state*1, config.stepsforsa) 
    f = open(save_file_name, mode='w')
    f.write('epoch, avg. cost for seq, avg. cost for model(notimelimit), avg. time for model(notimelimit),'+
         'avg. cost for model(timelimit), avg. canceled job num(timelimit), avg. time for model(timelimit),'+
         'avg. cost for SA, avg. time for SA, avg. cost for SA+model, avg. time for SA+model\n' +
         'avg. cost for NSA, avg. time for NSA, avg. cost for NSA+model, avg. time for NSA+model\n')
    f2 = open('simanneal/SCIP_time_NSA_steps_{}_layer{}_index{}_method{}_rescale{}_step{}.txt'.format(config.test_name, config.layer_type, config.layer_index, config.method_find_nodes, config.re_scale_state*1, config.stepsforsa), mode = 'w')
    f2.write('idle node num, schedule job num, required num nodes, solving_time, optimal_steps\n')
    f3 = open('simanneal/solving_log_time_obj_{}_layer{}_index{}_method{}_rescale{}_step{}.txt'.format(config.test_name, config.layer_type, config.layer_index, config.method_find_nodes, config.re_scale_state*1, config.stepsforsa), mode = 'w')

    for ep in range(1, total_test_episodes+1):
        cumulative_cost_nsa = 0
        cumulative_cost_sa_step500 = 0
        cumulative_cost_sa_step1000 = 0
        cumulative_cost_model_notimelimit = 0
        cumulative_cost_model = 0
        cumulative_cost_seq = 0
        cumulative_cost_nsa_model = 0
        cumulative_solving_time_nsa = 0
        
        cumulative_solving_time_sa_step500 = 0
        cumulative_solving_time_sa_step1000 = 0
        cumulative_solving_time_model = 0
        cumulative_solving_time_model_notimelimit = 0
        cumulative_solving_time_nsa_model = 0
        cumulative_cancel_job_number_model_timelimit = 0
        cumulative_cancel_job_number_model_notimelimit = 0
        cumulative_schedule_job_num = 0
        
        scheduler.env.reset()
        for t in range(1, max_ep_len+1):
            need_to_process = scheduler.env.need_proceed_by_RL()
            if not need_to_process:
                continue
            f3.write('---------------------------------------------------------------------\n')
            f3.write('ep: {}, instance: {}, idle num: {}, job num: {}, total request node: {}\n'.format(ep, t, scheduler.env.cluster.idle_node_num,
            len(scheduler.env.need_schedule_by_RL_jobs), scheduler.env.obtain_total_request_node()))
            start_time = time.time()
            scheduler.need_schedule_by_RL_jobs_before_move = deepcopy(scheduler.env.need_schedule_by_RL_jobs)
            scheduled_dic, reward = scheduler.find_initial_solution()
            
            super(HPCScheduler, scheduler).__init__(scheduled_dic, -reward, config.stepsforsa)  # important!
            config.SA_with_RL = True         
            best_scheduled_dic_NSA, best_cost_NSA, optimal_steps_NSA = scheduler.anneal(ep, t, f3)
            assert abs(best_cost_NSA-scheduler.env.best_cost) <= 1           
            duration_nsa = round(time.time() - start_time, 3)
            # scheduler.env.schedule_finish_assign_best_solution(refill_with_best_scheduled=True, duration = duration)
            
            # for SA steps=500
            config.SA_with_RL = False
            scheduler.env.remove_current_schedule()
            best_solution_for_NSA = scheduler.env.best_scheduled_rl     # record best solution for NSA            
            scheduler.env.reset_for_next_time_step(update_timestamp=False)
            start_time = time.time()
            scheduler.env.need_schedule_by_RL_jobs = scheduler.need_schedule_by_RL_jobs_before_move
            scheduled_dic_1, reward_1 = scheduler.find_initial_solution()
            assert reward_1 == reward
            super(HPCScheduler, scheduler).__init__(scheduled_dic_1, -reward_1, config.stepsforsa)                 
            best_scheduled_dic_SA_step500, best_cost_SA_step500, optimal_steps_SA_step500 = scheduler.anneal(ep, t, f3)
            duration_sa_step500 = round(time.time() - start_time, 3)
            
            # for SA steps=1000
            scheduler.env.remove_current_schedule()         
            scheduler.env.reset_for_next_time_step(update_timestamp=False)
            start_time = time.time()
            scheduler.env.need_schedule_by_RL_jobs = scheduler.need_schedule_by_RL_jobs_before_move
            scheduled_dic_1, reward_1 = scheduler.find_initial_solution()
            assert reward_1 == reward
            super(HPCScheduler, scheduler).__init__(scheduled_dic_1, -reward_1, config.stepsforsa*2)                 
            best_scheduled_dic_SA_step1000, best_cost_SA_step1000, optimal_steps_SA_step1000 = scheduler.anneal(ep, t, f3)
            duration_sa_step1000 = round(time.time() - start_time, 3)
            
            scheduler.env.need_schedule_by_RL_jobs = scheduler.need_schedule_by_RL_jobs_before_move
            scheduler.env.remove_current_schedule()
            cumulative_schedule_job_num += len(scheduler.need_schedule_by_RL_jobs_before_move)
            # scheduler.env.plot_cluster()
            if config.plot_cluster:
                scheduler.env.plot_cluster()
            if config.compare_with_model:
                all_job_assign_nodes_dic_1, cost_seq_2, cost_model, solving_time, cancled_job_num, is_optimal = scheduler.env.compare_with_other_policy(duration_NSA = duration_nsa, duration_SA = duration_sa_step500, best_cost_NSA = best_cost_NSA, best_cost_SA
                =best_cost_SA_step500, f_write = f2, optimal_steps = optimal_steps_NSA, enableoutput = False, with_timelimit_scip = True)
                scheduler.env.cost_model = 0
                scheduler.env.cost_seq = 0
                enableoutput = False
                # if ep == total_test_episodes and solving_time > duration and cancled_job_num == 0 and cost_model >= best_cost:
                #     enableoutput = True
                if not is_optimal:
                    all_job_assign_nodes_dic_2, cost_seq_1, cost_model_notimelimit, solving_time_notimelimit, cancled_job_num_notimelimit, is_optimal_notimelimit = scheduler.env.compare_with_other_policy(duration_NSA = duration_nsa, duration_SA = duration_sa_step500, best_cost_NSA = best_cost_NSA, best_cost_SA
                =best_cost_SA_step500, f_write = f2, optimal_steps = optimal_steps_NSA, enableoutput = False, with_timelimit_scip = False)
                    assert is_optimal_notimelimit
                    if config.method_find_nodes == 1:
                        assert cancled_job_num_notimelimit == 0
                    cumulative_cancel_job_number_model_notimelimit += cancled_job_num_notimelimit
                else:
                    all_job_assign_nodes_dic_2 = all_job_assign_nodes_dic_1
                    cost_seq_1 = cost_seq_2
                    cost_model_notimelimit = cost_model
                    solving_time_notimelimit = solving_time
                    assert cancled_job_num == 0
                # if cost_model_notimelimit - 10000 > best_cost:
                #     scheduler.env.plot_cluster()
                                    
                assert cost_seq_1 == cost_seq_2
                
                cumulative_cost_seq += cost_seq_1
                cumulative_cost_model_notimelimit += cost_model_notimelimit
                cumulative_cost_model += cost_model
                
                cumulative_solving_time_model_notimelimit += solving_time_notimelimit
                cumulative_solving_time_model += solving_time
                cumulative_cancel_job_number_model_timelimit += cancled_job_num
                
                if cost_model < best_cost_NSA and cancled_job_num == 0:
                    cumulative_cost_nsa_model += cost_model
                    cumulative_solving_time_nsa_model += solving_time
                    scheduler.env.schedule_finish_assign_best_solution(all_job_assign_nodes_dic_1, duration_nsa)
                else:
                    cumulative_cost_nsa_model += best_cost_NSA
                    cumulative_solving_time_nsa_model += duration_nsa
                    scheduler.env.schedule_finish_assign_best_solution(best_solution_for_NSA, duration_nsa)
            else:
                scheduler.env.calculate_cost_by_seq_allocate()
                scheduler.env.schedule_finish_assign_best_solution(best_solution_for_NSA, duration_nsa)
                print('\nSeq:{}, SA time step 500:{}, SA cost step 500:{}, SA time step 1000:{}, SA cost step 1000:{}, NSA time:{}, NSA cost:{}'.format(scheduler.env.cost_seq,
                    duration_sa_step500, best_cost_SA_step500, duration_sa_step1000, best_cost_SA_step1000, duration_nsa, best_cost_NSA))

            cumulative_cost_nsa += best_cost_NSA
            cumulative_solving_time_nsa += duration_nsa   

            cumulative_cost_sa_step500 += best_cost_SA_step500
            cumulative_solving_time_sa_step500 += duration_sa_step500
            
            cumulative_cost_sa_step1000 += best_cost_SA_step1000
            cumulative_solving_time_sa_step1000 += duration_sa_step1000

            scheduler.env.reset_for_next_time_step()

        f.write('{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(ep, cumulative_cost_seq//max_ep_len, 
        cumulative_cost_model_notimelimit//max_ep_len, round(cumulative_cancel_job_number_model_notimelimit/cumulative_schedule_job_num,2), round(cumulative_solving_time_model_notimelimit/max_ep_len, 1), 
        cumulative_cost_model//max_ep_len, round(cumulative_cancel_job_number_model_timelimit/cumulative_schedule_job_num,2), round(cumulative_solving_time_model/max_ep_len,1), 
        cumulative_cost_sa_step500//max_ep_len, round(cumulative_solving_time_sa_step500/max_ep_len, 1),
        cumulative_cost_sa_step1000//max_ep_len, round(cumulative_solving_time_sa_step1000/max_ep_len, 1),
        cumulative_cost_nsa//max_ep_len, round(cumulative_solving_time_nsa/max_ep_len, 1), 
        cumulative_cost_nsa_model//max_ep_len, round(cumulative_solving_time_nsa_model/max_ep_len, 1)))
        f.flush()
        f2.flush()
        f3.flush()
    f.close()
    f2.close()
    f3.close()

def test2(max_ep, max_time_steps):
    print("============================================================================================")
    scheduler = HPCScheduler()
    start_time_step = 0
    fw = open('simanneal/comm_wait_cost_{}.txt'.format(config.time_slot), 'w')
    for ep in range(1, max_ep + 1):  
        scheduler.env.reset()
        scheduler.env.start_index = start_time_step
        scheduler.env.current_timestamp = scheduler.env.all_jobs[start_time_step].submit_time + config.time_slot
        max_job_number = start_time_step + max_time_steps
        scheduler.env.job_number = max_job_number
        while(True):
            if scheduler.env.start_index >= max_job_number and len(scheduler.env.job_queue) == 0:
                break
            need_to_process = scheduler.env.need_proceed_by_RL()
            if not need_to_process:
                continue
            start_time = time.time()
            scheduler.need_schedule_by_RL_jobs_before_move = deepcopy(scheduler.env.need_schedule_by_RL_jobs)
            scheduled_dic, reward = scheduler.find_initial_solution()
            super(HPCScheduler, scheduler).__init__(scheduled_dic, -reward, config.stepsforsa)  # important!
            best_scheduled_dic, best_cost, obtain_optimal_steps = scheduler.anneal(ep, scheduler.env.start_index, None)
            assert abs(best_cost-scheduler.env.best_cost) <= 1
            scheduler.env.need_schedule_by_RL_jobs = scheduler.need_schedule_by_RL_jobs_before_move
            duration = time.time() - start_time
            scheduler.env.remove_current_schedule()
            scheduler.env.schedule_finish_assign_best_solution(best_scheduled=best_scheduled_dic, duration = duration)
            scheduler.env.reset_for_next_time_step()
        
        
        avg_comm_cost, avg_wait_time = scheduler.env.get_comm_cost_wait_time()
        # assert len(scheduler.env.scheduled_logs.values()) == max_time_steps
        # print('\navg. comm. cost,', avg_comm_cost, ', avg. waiting time, ', avg_wait_time)
        fw.write('ep:{}, comm. cost: {}, waiting. cost: {}\n'.format(ep, avg_comm_cost, avg_wait_time))
        fw.flush()
        start_time_step += max_time_steps
    fw.close()
            
if __name__ == '__main__':
    test()
