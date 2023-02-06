from distutils.log import error
from tkinter.tix import Tree
from tracemalloc import start
import sys

from numpy import nbytes
from hpc.job import Job, Workloads
from hpc.cluster import ClusterWithTopology
from config import Machine_State, config
import model.modelhpc as model


class HPCInstance:
    def __init__(self, workload_file = ''):  # do nothing and return. A workaround for passing parameters to the environment
        print("Initialize Simple HPC Env")
        print ("loading workloads from dataset:", workload_file)    
        self.loads = Workloads(workload_file)       
        self.cluster = ClusterWithTopology()
        self.job_queue = []
        self.need_schedule_by_model_jobs = []
        self.request_one_node_jobs = []
        self.running_jobs = []
        self.start_index = 0
        self.next_arriving_job_idx = self.start_index + 1
        self.current_timestamp = self.loads[self.start_index].submit_time
        self.job_number = self.loads.size()
        self.idle_node_ids_in_line = []

        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown
        # 5: Average Bounded Slowdown + Resource Utilization (0 + 3)
        self.job_score_type = config.job_score_type
        self.scheduled_logs = {}
        
        print("number of nodes:"+str(self.cluster.nb_machines) + ", number of jobs:"+str(self.job_number))            
    
    def reset(self, start_time_step):  
        self.job_queue = []
        self.need_schedule_by_model_jobs = []
        self.request_one_node_jobs = []
        self.running_jobs = []
        self.start_index = start_time_step
        self.next_arriving_job_idx = start_time_step + 1
        self.current_timestamp = self.loads[self.start_index].submit_time
        self.scheduled_logs = {}
        self.cluster.reset()
    
    def assign_job_machines(self):
        # 先移除已完成
        should_keep_run_jobs = []
        has_remove = True
        for job in self.running_jobs:
            if job.scheduled_time + job.run_time > self.current_timestamp:
                should_keep_run_jobs.append(job)
            else:
                self.cluster.release(job.allocated_machine_ids)
                # print("release job:"+str(job.job_id))
        
        if len(self.running_jobs) == len(should_keep_run_jobs):
            has_remove = False
        self.running_jobs = should_keep_run_jobs
      
        # 添加上个时段到达的作业
        has_new_arrive = False
        next_stage_arrive_job_index = 0
        time_span = [self.current_timestamp - config.time_slot, self.current_timestamp]
        assert self.loads[self.start_index].submit_time >= self.current_timestamp - config.time_slot       
        for i in range(self.start_index, self.loads.size()):
            job_scheduling = self.loads[i]
            if job_scheduling.submit_time >= time_span[0] and job_scheduling.submit_time < time_span[1]:
                has_new_arrive = True
                self.job_queue.append(job_scheduling)
            else:
                next_stage_arrive_job_index = i
                break
        # assert next_stage_arrive_job_index > self.start_index
        # 如果没有移除的作业且没有新到达的作业，就没有必要求解
        if next_stage_arrive_job_index == 0: # has scheduled at the last node
            return self.job_number

        if len(self.job_queue) == 0 or self.cluster.idle_node_num == 0 or (not has_remove and not has_new_arrive):
            return next_stage_arrive_job_index

        used_nodes_num = 0
        for job in should_keep_run_jobs:
            used_nodes_num += job.request_number_of_nodes
        
        self.select_jobs_can_be_allocated()

        if not self.need_schedule_by_model_jobs:
            if self.request_one_node_jobs:
                for job in self.request_one_node_jobs:
                    self.allocate_for_jobs_without_RL(job)
            return next_stage_arrive_job_index

        assign_result, nb_can_use_machines, solve_time, cancel_job_ids = model.build_and_solve(jobs = self.need_schedule_by_model_jobs, cluster = self.cluster, temp_can_not_use_node_ids=[],
        idel_node_ids_in_line=[])
        
        assert len(cancel_job_ids) == 0

        solve_time = int(solve_time)

        assert used_nodes_num + nb_can_use_machines == self.cluster.nb_machines

        nb_jobs = len(self.need_schedule_by_model_jobs)
        unschedule_jobs = []
        for jobindex in range(nb_jobs):
            job = self.need_schedule_by_model_jobs[jobindex]
            assign_node_ids_this_job = assign_result[jobindex]
            assert len(assign_node_ids_this_job) == job.request_number_of_nodes             

            assert job.scheduled_time == -1
            job.scheduled_time = self.current_timestamp + solve_time
            job.allocated_machine_ids = assign_node_ids_this_job
            score = self.job_score(job)  # calculated reward
            self.scheduled_logs[job] = score # scheduled_logs uses job as the key
            self.running_jobs.append(job)
            self.cluster.allocate_in_Batch(assign_node_ids_this_job)

        nb_nodes_request = 0
        nb_nodes_request_for_each = []
        for job in self.need_schedule_by_model_jobs:
            nb_nodes_request += job.request_number_of_nodes
            nb_nodes_request_for_each.append(job.request_number_of_nodes)

        print("running job:", len(should_keep_run_jobs), ", can use nodes:", nb_can_use_machines, ", request nodes:", nb_nodes_request, ", request for each job:", nb_nodes_request_for_each)
        print("jobs need to assign:" + str(nb_jobs) + ", not assigned jobs:" + str(len(self.job_queue)))

        return next_stage_arrive_job_index

        # 尽可能选择越多的作业数量完成分配
    def select_jobs_can_be_allocated(self):
        self.need_schedule_by_model_jobs = []
        self.request_one_node_jobs = []
        self.job_queue.sort(key = lambda job: [-job.wait_stage_num, job.request_number_of_nodes])  # 升序排序
        has_add_node_num = 0
        for job in self.job_queue:
            if has_add_node_num + job.request_number_of_nodes <= self.cluster.idle_node_num:
                if job.request_number_of_nodes != 1:
                    if len(self.need_schedule_by_model_jobs) < config.max_queue_size:
                        self.need_schedule_by_model_jobs.append(job)
                    else:
                        break
                else:
                    self.request_one_node_jobs.append(job)
                has_add_node_num += job.request_number_of_nodes
            else:
                break

        # update job queue
        need_schedule_job_num = len(self.request_one_node_jobs) + len(self.need_schedule_by_model_jobs)
        if len(self.job_queue) > need_schedule_job_num:   
            self.job_queue = self.job_queue[need_schedule_job_num:]
            # 对于没有被调度的作业，放到下一次调度，需要更新wait_stage_num
            for job in self.job_queue:
                job.wait_stage_num += 1

        elif len(self.job_queue) == need_schedule_job_num:
            self.job_queue = []
        else:
            raise error
    
    def allocate_for_jobs_without_RL(self, job):
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
        job.scheduled_time = self.current_timestamp
        self.running_jobs.append(job)
    
    def convert_job_list_to_dic(self, jobs):
        schedule_dic = {}
        for job in jobs:
            assert not schedule_dic.__contains__(job.id)
            assert len(job.allocated_machine_ids) == job.request_number_of_nodes
            schedule_dic[job.id] = job.allocated_machine_ids
        return schedule_dic

    # BF、FCFS等启发式算法调度
    def schedule_curr_sequence_reset(self, score_fn, backfill, start_time_step, max_time_steps): 
        self.reset(start_time_step)
        self.job_queue.append(self.loads[self.start_index])
        max_job_number = start_time_step + max_time_steps
        while True:
            # 按照规则排序
            # self.job_queue.sort(key=lambda j: score_fn(j))
            job_for_scheduling = self.job_queue[0]
            # 如果资源不满足要求
            if not self.cluster.can_allocated(job_for_scheduling):
                if backfill:
                    self.moveforward_for_resources_backfill_greedy(job_for_scheduling, max_job_number)
                else:
                    self.skip_for_resources_greedy(job_for_scheduling, max_job_number)
            # 该作业之前没有被调度过
            assert job_for_scheduling.scheduled_time == -1  
            job_for_scheduling.scheduled_time = self.current_timestamp
            job_for_scheduling.allocated_machine_ids = self.cluster.allocate(job_for_scheduling.id,
                                                                          job_for_scheduling.request_number_of_nodes)
            # print('Allocating job:' + str(job_for_scheduling.id))

            self.running_jobs.append(job_for_scheduling)
            score = self.job_score(job_for_scheduling)  # calculated reward
            self.scheduled_logs[job_for_scheduling] = score # scheduled_logs uses job as the key
            self.job_queue.remove(job_for_scheduling)

            not_empty = self.moveforward_for_job(max_job_number)
            if not not_empty:
                break  

    # @profile
    def moveforward_for_resources_backfill_greedy(self, job, max_job_number):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        earliest_start_time = self.current_timestamp
        # sort all running jobs by estimated finish time
        self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
        free_nodes = self.cluster.idle_node_num
        for running_job in self.running_jobs:
            free_nodes += running_job.request_number_of_nodes
            earliest_start_time = (running_job.scheduled_time + running_job.run_time)
            if free_nodes >= job.request_number_of_nodes:
                break

        while not self.cluster.can_allocated(job):

            # try to backfill as many jobs as possible. Use FCFS
            self.job_queue.sort(key=lambda _j: self.fcfs_score(_j))
            job_queue_iter_copy = list(self.job_queue)
            for _j in job_queue_iter_copy:
                if (self.current_timestamp + _j.run_time) < earliest_start_time:
                    if self.cluster.can_allocated(_j):
                        # we should be OK to schedule the job now
                        assert _j.scheduled_time == -1  # this job should never be scheduled before.
                        _j.scheduled_time = self.current_timestamp
                        _j.allocated_machine_ids = self.cluster.allocate(_j.id, _j.request_number_of_nodes)
                        self.running_jobs.append(_j)
                        score = self.job_score(_j)  # calculated reward
                        self.scheduled_logs[_j] = score  # scheduled_logs uses job as key
                        self.job_queue.remove(_j)  # remove the job from job queue

            # move to the next timestamp
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machine_ids

            if self.next_arriving_job_idx < max_job_number \
                    and self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job

    def skip_for_resources_greedy(self, job, max_job_number):
        # note that this function is only called when current job can not be scheduled.
        assert not self.cluster.can_allocated(job)

        while not self.cluster.can_allocated(job):
            # schedule nothing, just move forward to next timestamp. It should just add a new job or finish a running job
            assert self.running_jobs
            self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
            next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
            next_resource_release_machines = self.running_jobs[0].allocated_machine_ids

            if self.next_arriving_job_idx < max_job_number and self.loads[
                self.next_arriving_job_idx].submit_time <= next_resource_release_time:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    # @profile
    def moveforward_for_job(self, max_job_number):
        if self.job_queue:
            return True

        # if we need to add job, but can not add any more, return False indicating the job_queue is for sure empty now.
        if self.next_arriving_job_idx >= max_job_number:
            # assert not self.job_queue
            return False

        # move forward to add jobs into job queue.
        while not self.job_queue:
            if not self.running_jobs:  # 如果没有运行作业
                next_resource_release_time = sys.maxsize  # always add jobs if no resource can be released.
                next_resource_release_machines = []
            else:
                self.running_jobs.sort(key=lambda running_job: (running_job.scheduled_time + running_job.run_time))
                next_resource_release_time = (self.running_jobs[0].scheduled_time + self.running_jobs[0].run_time)
                next_resource_release_machines = self.running_jobs[0].allocated_machine_ids

            if self.loads[self.next_arriving_job_idx].submit_time <= next_resource_release_time and self.next_arriving_job_idx < max_job_number:
                self.current_timestamp = max(self.current_timestamp, self.loads[self.next_arriving_job_idx].submit_time)
                self.job_queue.append(self.loads[self.next_arriving_job_idx])
                self.next_arriving_job_idx += 1
                return True  # job added
            else:
                self.current_timestamp = max(self.current_timestamp, next_resource_release_time)
                self.cluster.release(next_resource_release_machines)
                self.running_jobs.pop(0)  # remove the first running job.

    def fcfs_score(self, job):
        submit_time = job.submit_time
        return submit_time
    
    def job_score(self, job_for_scheduling):
        # 0: Average bounded slowdown, 1: Average waiting time
        # 2: Average turnaround time, 3: Resource utilization 4: Average slowdown 5: communication cost
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
    
    def deal_logs(self):
        sum_comm = 0
        sum_wait = 0
        length = len(self.scheduled_logs.values())
        for v in self.scheduled_logs.values():
            sum_comm += v[0]
            sum_wait += v[1]
        self.scheduled_logs = {}
        assert not self.job_queue
        return sum_comm/length, sum_wait/length


