from distutils.log import error
import math
from turtle import right

import numpy as np
from config import config, Machine_State
import matplotlib.pyplot as plt
import plot

class Machine:
    def __init__(self, id, x, y, z, h, w, rackid, containerid):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.h = h # for conv2d state, higth
        self.w = w # for conv2d state, width
        self.running_job_id = -1
        self.state = Machine_State.idle_state
        self.rackid = rackid
        self.containerid = containerid
        self.job_history = []
        
    #释放作业
    def release(self):
        if self.state==Machine_State.idle_state or self.state==Machine_State.sleep_state:
            return -1
        else:
            self.state = Machine_State.idle_state
            self.running_job_id = -1
            return 1
    
    # 重设为初始状态
    def reset(self):
        self.state = Machine_State.idle_state
        self.running_job_id = -1
        self.job_history = []
    
    # 判断两个节点是否相同
    def __eq__(self, other):
        return self.id == other.id
    
    # 得到节点名称
    def __str__(self):
        return "M["+str(self.id)+"] "

class ClusterWithTopology:
    def __init__(self) -> None:
        
        self.nb_machines = config.x_unit_num * config.x_node_num_each_unit * config.y_unit_num * config.z_unit_num
        self.nb_machines_one_rack = config.x_node_num_each_unit * config.y_unit_num
        self.xyz_to_id_dic = {}
        self.machines = self.construct_machines()
        self.each_node_sorted_distances = {}
        self.calculate_distances_and_sort()
        
        self.idle_node_num = self.nb_machines
        self.run_node_num = 0
        self.plt = plt
        
    
    def reset(self):
        self.run_node_num = 0
        self.idle_node_num = self.nb_machines
        for m in self.machines:
            m.reset()

    def normalize_xyz(self):
        max_x, max_y, max_z = 0, 0, 0
        for m in self.machines:
            if m.x > max_x:
                max_x = m.x
            if m.y > max_y:
                max_y = m.y
            if m.z > max_z:
                max_z = m.z
        for m in self.machines:
            m.x = m.x/max_x
            m.y = m.y/max_y
            m.z = m.z/max_z  
    
    def get_max_xyz(self):
        max_x, max_y, max_z = 0, 0, 0
        for m in self.machines:
            if m.x > max_x:
                max_x = m.x
            if m.y > max_y:
                max_y = m.y
            if m.z > max_z:
                max_z = m.z
        return max_x, max_y, max_z
    
    def get_max_container_rackid(self):
        max_container, max_rack = 0, 0
        for m in self.machines:
            if m.containerid > max_container:
                max_container = m.containerid
            if m.rackid > max_rack:
                max_rack = m.rackid
        return max_container, max_rack
        
    def construct_machines(self):
        machines = []
        id = 0
        rack_id = 0
        container_id = 0
        for k in range(config.z_unit_num):
            k1 = k * config.z_distance_between_two_unit
            for i in range(config.x_unit_num):               
                i3 = i * (config.x_distance_between_two_unit + config.x_distance_two_nodes * config.x_node_num_each_unit)
                for j in range(config.y_unit_num):
                    j1 = j * config.y_distance_between_two_unit
                    for i1 in range(config.x_node_num_each_unit):
                        i2 = i3 + i1*config.x_distance_two_nodes
                        h, w = self.node_id_to_point_in_conv2d(id)
                        machines.append(Machine(id, i2, j1, k1, h = h, w=w, rackid = rack_id, containerid = container_id))
                        key = ','.join([str(i*config.x_node_num_each_unit+i1), str(j), str(k)])
                        assert not self.xyz_to_id_dic.__contains__(key)
                        self.xyz_to_id_dic[key] = id
                        id += 1
                    container_id += 1
                rack_id += 1
        return machines
    
    ### 找到最临近节点集合的相应数量的节点, return result include center_nodeid 
    def find_most_adjacent_nodes_to_sets_of_nodes(self, center_node_ids, top_number, temp_can_not_use_ids):
        assert len(center_node_ids) >= 1
        temp_distance_dic = {}
        for m in self.machines:
            if m.state == Machine_State.run_state or (m.id in center_node_ids) or (m.id in temp_can_not_use_ids):
                continue
            else:
                temp_distance = 0
                for node_id in center_node_ids:
                    temp_distance += self.calculate_two_nodes_distance(m.id, node_id)
                assert not temp_distance_dic.__contains__(m.id)
                temp_distance_dic[m.id] = temp_distance
            
        temp_distance_list = sorted(temp_distance_dic.items(), key= lambda x:x[1]) # 升序
        return temp_distance_list[0][0]
    
    ### 找到最临近相应数量的节点, return result include start_nodeid 
    def find_most_adjacent_nodes_by_seq(self, start_nodeid , top_number, temp_can_not_use_node_ids):
        success_flag = 0
        select_node_indexs = []
        assert top_number != 0
        for m in self.machines:
            if m.id < start_nodeid or m.id in temp_can_not_use_node_ids:
                continue
            if m.state == Machine_State.idle_state:
                select_node_indexs.append(m.id)
            if len(select_node_indexs) == top_number:
                success_flag = 1
                break
        assert select_node_indexs.__contains__(start_nodeid)
        
        if not success_flag:
            for m in self.machines:
                if m.id in temp_can_not_use_node_ids:
                    continue
                if m.id >= start_nodeid:
                    break
                if m.state == Machine_State.idle_state:
                    select_node_indexs.append(m.id)
                if len(select_node_indexs) == top_number:
                    success_flag = 1
                    break
        if success_flag == 0:
            raise error('error')
            print("there are no enough nodes for job, the request number of nodes is ", top_number)   
        return select_node_indexs
    
    # without specify start_nodeid
    def find_idle_nodes_by_seq(self, top_number, temp_can_not_use_node_ids):
        success_flag = 0
        select_node_indexs = []
        assert top_number != 0
        for m in self.machines:
            if m.id in temp_can_not_use_node_ids:
                continue
            if m.state == Machine_State.idle_state:
                select_node_indexs.append(m.id)
            if len(select_node_indexs) == top_number:
                success_flag = 1
                break
        if success_flag == 0:
            raise error('error') 
        return select_node_indexs
    
    ### 找到最临近相应数量的节点在
    def find_most_adjacent_nodes_inline(self, start_nodeid , top_number, temp_can_not_use_node_ids, node_ids_on_line):
        success_flag = 0
        select_node_ids = []
        assert top_number != 0
        assert not node_ids_on_line == []

        start_index = node_ids_on_line.index(start_nodeid)
        assert self.machines[start_nodeid].state == Machine_State.idle_state
        select_node_ids.append(start_nodeid)
        
        can_select_node_num = len(node_ids_on_line)

        right_index = (start_index + 1) % can_select_node_num
        left_index = (start_index-1) % can_select_node_num
        while(True):
            assert not right_index == start_index and not left_index == start_index 
            right_node_id = node_ids_on_line[right_index]
            left_node_id = node_ids_on_line[left_index]
            if self.machines[right_node_id].state == Machine_State.run_state or right_node_id in temp_can_not_use_node_ids:
                right_index = (right_index + 1) % can_select_node_num
                continue
            if self.machines[left_node_id].state == Machine_State.run_state or left_node_id in temp_can_not_use_node_ids:
                left_index = (left_index - 1) % can_select_node_num
                continue 
                     
            right_distance = self.calculate_two_nodes_distance(start_index, left_index)
            left_distance = self.calculate_two_nodes_distance(start_index, left_index)

            if right_distance < left_distance - 1:
                select_node_ids.append(right_node_id)
                right_index = (right_index + 1) % can_select_node_num
            elif right_distance - 1 > left_distance:
                select_node_ids.append(left_node_id)
                left_index = (left_index - 1) % can_select_node_num
            else:
                select_node_ids.append(right_node_id)
                right_index = (right_index + 1) % can_select_node_num
                if len(select_node_ids) < top_number:
                    select_node_ids.append(left_node_id)
                    left_index = (left_index - 1) % can_select_node_num

            if len(select_node_ids) == top_number:
                success_flag = 1
                break

        if success_flag == 0:
            raise error('error')
        return select_node_ids

    ### 找到最临近相应数量的节点, return result include center_nodeid 
    def find_most_adjacent_nodes(self, center_nodeid , top_number, temp_can_not_use_node_ids):
        sorted_distances_index = self.each_node_sorted_distances[center_nodeid]
        select_node_indexs = []
        success_flag = 0
        for nodeid in sorted_distances_index:
            cur_node = self.machines[nodeid]
            if cur_node.state == Machine_State.run_state or (nodeid in temp_can_not_use_node_ids):
                continue
            select_node_indexs.append(nodeid)
            if len(select_node_indexs) == top_number:
                success_flag = 1
                break        
        # if success_flag == 0:
        #     # raise error('error')
        #     print("there are no enough nodes for job, the request number of nodes is ", top_number)

        return select_node_indexs

    ### calculate cost for a set of nodes
    def calculate_total_distance_nodes(self, node_ids):
        total_distance = 0
        assert len(node_ids) >= 1
        for i in node_ids:
            for j in node_ids:
                if i < j:
                    distance = self.calculate_two_nodes_distance(i, j)
                    total_distance += distance
        return max(total_distance//len(node_ids),1)

    ### 计算任意两节点之间的距离
    def calculate_distances_and_sort(self):
        for first_node_id in range(self.nb_machines):
            temp_distances = []
            for second_node_id in range(self.nb_machines):
                distance = self.calculate_two_nodes_distance(first_node_id, second_node_id)
                temp_distances.append(distance)

            sorted_index = np.argsort(temp_distances)
            assert len(sorted_index) == self.nb_machines
            assert sorted_index[0] == first_node_id
            # sorted_index = sorted_index[0: paramhelp.job_max_request + 1]            
            self.each_node_sorted_distances[first_node_id] = sorted_index
    
    ## 计算两个节点之间的距离
    def calculate_two_nodes_distance(self, first_node_id, second_node_id):
        # if first_node_id == second_node_id:
        #     return 0
        # first_node = self.machines[first_node_id]
        # second_node = self.machines[second_node_id]
        # return pow((first_node.x - second_node.x)**2 + 
        # (first_node.y - second_node.y)**2 + (first_node.z - second_node.z)**2,2)
        cost_one_hop = 1000
        if first_node_id == second_node_id:
            return 0
        first_node = self.machines[first_node_id]
        second_node = self.machines[second_node_id]
        if first_node.containerid == second_node.containerid:
            return 1*cost_one_hop
        elif first_node.rackid == second_node.rackid:
            return 3*cost_one_hop
        else:
            return 5*cost_one_hop

    # 计算跳数
    def calculate_two_nodes_hops(self, first_node_id, second_node_id):
        if first_node_id == second_node_id:
            return 0
        first_node = self.machines[first_node_id]
        second_node = self.machines[second_node_id]
        if first_node.containerid == second_node.containerid:
            return 1
        elif first_node.rackid == second_node.rackid:
            return 3
        else:
            return 5

    def construct_machines_one_rack(x_distance_two_nodes, x_node_num_each_unit, y_unit_num, y_distance_between_two_unit):
        machines = []
        id = 0        
        for j in range(y_unit_num):
            j1 = j * y_distance_between_two_unit
            if j%2 == 0:
                for i in range(x_node_num_each_unit):
                    i1 = i*x_distance_two_nodes
                    machines.append(Machine(id, i1, j1, 0))
                    id += 1
            else:
                for i in range(x_node_num_each_unit-1, -1, -1):
                    i1 = i*x_distance_two_nodes
                    machines.append(Machine(id, i1, j1, 0))
                    id += 1 
        
        Rack(0, machines)          
    
    def construct_containers(x_unit_num, x_node_num_each_unit, y_unit_num, z_unit_num):
        containers = []
        nb_containers = x_unit_num * y_unit_num * z_unit_num * x_node_num_each_unit
        for i in range(nb_containers):
            containers.append(Container(i, x_node_num_each_unit))
        return containers, nb_containers

    def construct_racks(x_unit_num, y_unit_num, z_unit_num):
        racks = []
        nb_rack = x_unit_num * y_unit_num * z_unit_num
        for i in range(nb_rack):
            racks.append(Rack(i, y_unit_num))
        return racks
    
    # 判断是否可分配作业job
    def can_allocated(self, job):
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes > self.idle_node_num:
            return False
        if job.request_number_of_nodes != -1 and job.request_number_of_nodes <= self.idle_node_num:
            return True
    
    # 分配作业
    def allocate_in_Batch(self, assign_node_index):
        for nodeid in assign_node_index:
            node = self.machines[nodeid]
            if node.state == Machine_State.idle_state:
                node.state = Machine_State.run_state                
            else:
                raise error('error')
        self.run_node_num += len(assign_node_index)
        self.idle_node_num -= len(assign_node_index)

    def allocate(self, job_id, request_num_node):
        allocated_node_ids = []
        allocated_num = 0
        for m in self.machines:
            if m.state == Machine_State.idle_state:
                allocated_num += 1
                m.state = Machine_State.run_state
                allocated_node_ids.append(m.id)
            
            if allocated_num == request_num_node:
                break
        assert allocated_num == request_num_node
        if allocated_num == request_num_node:
            self.run_node_num += request_num_node
            self.idle_node_num -= request_num_node
            return allocated_node_ids

        raise error("Error in allocation, there are enough free resources but can not allocated!")
        # return []

    # 释放releases中的节点
    def release(self, releases_node_ids):
        for node_id in releases_node_ids:
            assert self.machines[node_id].state == Machine_State.run_state
            self.machines[node_id].state = Machine_State.idle_state
        self.run_node_num -= len(releases_node_ids)
        self.idle_node_num += len(releases_node_ids)
        assert self.idle_node_num + self.run_node_num == self.nb_machines
        assert self.run_node_num >= 0 and self.idle_node_num >= 0

    def xyz_to_id(self, key):
        if not self.xyz_to_id_dic.__contains__(key):
            raise error('error')
        else:
            return self.xyz_to_id_dic[key]

    def plot(self, scheduled):
        # assert config.plot_cluster
        if scheduled == None:
            plot.plot_cluster(self.plt, self)
        else:
            plot.plot_cluster_with_scheduled_jobs(self.plt, self, scheduled)
    
    def get_machines_by_ids(self, ids):
        machines_result = []
        for id in ids:
            machines_result.append(self.machines[id])
        return machines_result
    
    def node_id_to_point_in_conv2d(self, node_id):
        # h = int(node_id/config.conv2dW) # 0,1,2,3 rack id
        # w = node_id - h * config.conv2dW
        # return h, w
        rack_id = int(node_id/self.nb_machines_one_rack)
        axz = int(rack_id/config.x_unit_num)
        axx = rack_id - axz * config.x_unit_num
        node_index_in_one_rack = node_id - self.nb_machines_one_rack * rack_id
        y = int(node_index_in_one_rack/config.x_node_num_each_unit)
        x1 = node_index_in_one_rack - y * config.x_node_num_each_unit
        h = axz * config.y_unit_num + y
        w = axx * config.x_node_num_each_unit + x1
        return h, w
    
    def get_machine_states(self):
        states = []
        for machine in self.machines:
            states.append(machine.state)
        return states, self.idle_node_num, self.run_node_num

class Rack:
    def __init__(self, id, y_unit_num: int) -> None:
        self.id = id
        self.containers = [i for i in range(id * y_unit_num, id * y_unit_num + y_unit_num)]
    def __init__(self, id, machines: list) -> None:
        self.id = id
        self.machines = machines
     
    # def copyrack(id, machines, x_unit_num, y_unit_num, z_unit_num):
    #     for k in range()
    #     Rack(id)
        
        
class Container:
    def __init__(self, id, x_node_num_each_unit) -> None:
        self.id = id
        self.machines = [i for i in range(id * x_node_num_each_unit, (id+1) * x_node_num_each_unit)]
        







    