import re
import sys
import math
from config import config
import random

class Job:
    """
    1. Job Number -- a counter field, starting from 1.
    2. Submit Time -- in seconds. The earliest time the log refers to is zero, and is usually the submittal time of the first job. The lines in the log are sorted by ascending submittal times. It makes sense for jobs to also be numbered in this order.
    3. Wait Time -- in seconds. The difference between the job's submit time and the time at which it actually began to run. Naturally, this is only relevant to real logs, not to models.
    4. Run Time -- in seconds. The wall clock time the job was running (end time minus start time).
    We decided to use ``wait time'' and ``run time'' instead of the equivalent ``start time'' and ``end time'' because they are directly attributable to the Scheduler and application, and are more suitable for models where only the run time is relevant.
    Note that when values are rounded to an integral number of seconds (as often happens in logs) a run time of 0 is possible and means the job ran for less than 0.5 seconds. On the other hand it is permissable to use floating point values for time fields.
    5. Number of Allocated Processors -- an integer. In most cases this is also the number of processors the job uses; if the job does not use all of them, we typically don't know about it.
    6. Average CPU Time Used -- both user and system, in seconds. This is the average over all processors of the CPU time used, and may therefore be smaller than the wall clock runtime. If a log contains the total CPU time used by all the processors, it is divided by the number of allocated processors to derive the average.
    7. Used Memory -- in kilobytes. This is again the average per processor.
    8. Requested Number of Processors.
    9. Requested Time. This can be either runtime (measured in wallclock seconds), or average CPU time per processor (also in seconds) -- the exact meaning is determined by a header comment. In many logs this field is used for the user runtime estimate (or upper bound) used in backfilling. If a log contains a request for total CPU time, it is divided by the number of requested processors.
    10. Requested Memory (again kilobytes per processor).
    11. Status 1 if the job was completed, 0 if it failed, and 5 if cancelled. If information about chekcpointing or swapping is included, other values are also possible. See usage note below. This field is meaningless for models, so would be -1.
    12. User ID -- a natural number, between one and the number of different users.
    13. Group ID -- a natural number, between one and the number of different groups. Some systems control resource usage by groups rather than by individual users.
    14. Executable (Application) Number -- a natural number, between one and the number of different applications appearing in the workload. in some logs, this might represent a script file used to run jobs rather than the executable directly; this should be noted in a header comment.
    15. Queue Number -- a natural number, between one and the number of different queues in the system. The nature of the system's queues should be explained in a header comment. This field is where batch and interactive jobs should be differentiated: we suggest the convention of denoting interactive jobs by 0.
    16. Partition Number -- a natural number, between one and the number of different partitions in the systems. The nature of the system's partitions should be explained in a header comment. For example, it is possible to use partition numbers to identify which machine in a cluster was used.
    17. Preceding Job Number -- this is the number of a previous job in the workload, such that the current job can only start after the termination of this preceding job. Together with the next field, this allows the workload to include feedback as described below.
    18. Think Time from Preceding Job -- this is the number of seconds that should elapse between the termination of the preceding job and the submittal of this one.
    """
    def __init__(self, job_id, line, num_procs_per_node=1):
        line = line.strip()
        s_array = re.split("\\s+", line)
        # self.id = int(s_array[0])
        self.id = job_id
        self.submit_time = int(s_array[1])
        self.run_time = int(s_array[2])
        self.request_number_of_nodes = int(s_array[3])
        self.wait_stage_num = 0
        # assert self.request_number_of_nodes <= config.job_max_request
        # self.wait_time = int(s_array[2])
        # self.run_time = int(s_array[3])
        # self.number_of_allocated_processors = int(s_array[4])
        # self.request_number_of_nodes = min(int(math.ceil(float(self.number_of_allocated_processors) / float(num_procs_per_node))), 
        # config.job_max_request)
        # if self.request_number_of_nodes == 1 and self.id % 2 == 0:
        #     self.request_number_of_nodes = 10
        
        self.scheduled_time = -1
        self.allocated_machine_ids = None
        
    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        return "J["+str(self.id)+"]-["+str(self.request_number_of_nodes)+"]-["+str(self.submit_time)+"]-["+str(self.run_time)+"]"
    def __feature__(self):
        return [self.id, self.submit_time, self.request_number_of_nodes, self.run_time]


class Workloads:
    def __init__(self, path):
        self.all_jobs = []
        self.max_exec_time = 0
        self.min_exec_time = sys.maxsize
        self.max_requested_node = 0
        self.min_request_node_num = sys.maxsize
        max_id = -1
        job_id = 0
        with open(path) as fp:
            for line in fp:
                j = Job(job_id, line, 1)
                job_id += 1
                if j.run_time > self.max_exec_time:
                    self.max_exec_time = j.run_time
                if j.run_time < self.min_exec_time:
                    self.min_exec_time = j.run_time
                if j.request_number_of_nodes > self.max_requested_node:
                    self.max_requested_node = j.request_number_of_nodes
                if j.request_number_of_nodes < self.min_request_node_num:
                    self.min_request_node_num = j.request_number_of_nodes
                assert j.id > max_id
                max_id = j.id
                # filter those illegal data whose runtime < 0
                if j.run_time < 0:
                    j.run_time = 10
                if j.run_time > 0:
                    self.all_jobs.append(j)

        print ("max request nodes:", self.max_requested_node,";min request nodes:", self.min_request_node_num,
               ";max execution time:", self.max_exec_time, ";min execution time:", self.min_exec_time, "; max job id:", max_id)

        # self.all_jobs.sort(key=lambda job: job.id)

    def size(self):
        return len(self.all_jobs)

    def reset(self):
        for job in self.all_jobs:
            job.scheduled_time = -1

    def __getitem__(self, item):
        return self.all_jobs[item]


if __name__ == "__main__":
    print ("Loading the workloads...")
    load = Workloads("data/lublin_1000.txt")
    print ("Finish loading the workloads...", type(load[0]))
    print (load.max_nodes, load.max_procs)
    print (load[0].__feature__())
    print (load[1].__feature__())
