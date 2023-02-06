
class debughelp:
     
    # print action
    def print_act_param(act, param):
        # print("act:{}, param:{}".format(act, param))
        # print(f'act:，{act}，param:{param}')
        print('act:', str(act), 'param:', str(param))
    
    def print_current_state(state):
        used_node_num, idle_node_num, sleep_node_num= state
        print('used:',used_node_num, 'idle:', idle_node_num, 'sleep:', sleep_node_num)

    def print_reward(reward):
        print('reward:', reward)      

    def print_current_state_reward(state, reward):
        used_node_num, idle_node_num, sleep_node_num = state
        print('used:',used_node_num, 'idle:', idle_node_num, 'sleep:', sleep_node_num, 'reward:', reward)

    def print_current_epoch(epoch):
        print("--------------------------epoch {}---------------------:".format(epoch))

    def print_scheduling_job_number(scheduling_job):
        print('Scheduling job:', scheduling_job)
