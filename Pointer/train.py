from sched import scheduler
from turtle import update
import numpy as np
from pandas import array
import torch
import torch.optim as optim
import os
from datetime import datetime
from config import config
from hpc.HPCEnv import HPCEnv
from Pointer.actor import HPCActorModel
from Pointer.critic import HPCCriticModel
import torch.nn.functional as F

def _actor_model_forward(actor, static_input, dynamic_input, hpcEnv:HPCEnv):

    node_logp = []
    dynamic_input_float = dynamic_input.float()

    while(True):
        # get mask
        mask = [1]*len(hpcEnv.idle_node_ids_in_line)
        for job_id in hpcEnv.scheduled_rl.keys():
            has_used_node_ids = hpcEnv.scheduled_rl[job_id]
            for node_id in has_used_node_ids:
                index = hpcEnv.idle_node_ids_in_line.index(node_id)
                mask[index] = 0
            
        mask = torch.tensor(mask).unsqueeze(0)

        # Forward pass. Returns a probability distribution over the point (tour end or depot) that origin should be connected to
        probs = actor.forward(static_input, dynamic_input_float)
        probs = F.softmax(probs + mask.log(), dim=1)  # Set prob of masked tour ends to zero

        if actor.training:
            m = torch.distributions.Categorical(probs)
            # Sometimes an issue with Categorical & sampling on GPU; See:
            # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
            ptr = m.sample()
            # while not torch.gather(1, ptr.data.unsqueeze(1)).byte().all():
            #     ptr = m.sample()
            logp = m.log_prob(ptr)
        else:
            prob, ptr = torch.max(probs, 1)  # Greedy selection
            logp = prob.log()

        ptr_np = ptr.cpu().numpy()
        action_ptr = ptr_np[0]
        # do action
        action = hpcEnv.idle_node_ids_in_line[action_ptr]         
        reward, done, skip = hpcEnv.step(action, update_state = False)
        # node_logp.append(logp.unsqueeze(1))
        node_logp.append(logp)
        
        if done:
            hpcEnv.schedule_finish_assign_best_solution(None, duration=0)
            hpcEnv.reset_for_next_time_step(update_timestamp = False)   
            break
    if len(node_logp) == 1:
        node_logp.append(torch.tensor([0]))
    node_logp = torch.cat(node_logp, dim=0).tolist()
    return node_logp, reward

def _critic_model_forward(critic, static_input, dynamic_input):
    dynamic_input_float = dynamic_input.float()

    # dynamic_input_float[:, :, 0] = dynamic_input_float[:, :, 0]

    return critic.forward(static_input, dynamic_input_float).view(-1)

################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "jobscheduling"

    max_ep_len = 1000                   # max timesteps in one episode
    max_training_timesteps = int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 1       # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 8           # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)          # save model frequency (in num timesteps)
    
    pointer_hidden_size = 128
    critic_hidden_size = 128
    max_grad_norm = 2

    #####################################################
    
    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 256     # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0001       # learning rate for actor network
    lr_critic = 0.0005       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = HPCEnv()

    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    log_dir = "Pointer_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + config.test_name + "_method_" + str(config.method_find_nodes) + '_' + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = config.cluster_scale      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)

    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, config.test_name, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print("============================================================================================")
    
    ################# training procedure ################
    actor = HPCActorModel(config.device, hidden_size=pointer_hidden_size).to(config.device)
    critic = HPCCriticModel(critic_hidden_size).to(config.device)

    actor_optim = optim.Adam(actor.parameters(), lr=lr_actor)
    actor.train()
    critic_optim = optim.Adam(critic.parameters(), lr=lr_critic)
    critic.train()

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    batch_size = 1

    # training loop
    while time_step <= max_training_timesteps:

        env.reset()
        current_ep_reward = 0

        rewards, logps, critic_ests = [], [], []

        for t in range(1, max_ep_len+1):
            # select action with policy
            need_to_process = env.need_proceed_by_RL(alway_continue=True)
            
            if not need_to_process:
                continue
            nb_input_points = env.cluster.idle_node_num
            static_input, dynamic_input = env.get_network_input(nb_input_points)
            
            static_input = torch.from_numpy(static_input).to(config.device).float()
            dynamic_input = torch.from_numpy(dynamic_input).to(config.device).float()

            static_input, dynamic_input = static_input.unsqueeze(0), dynamic_input.unsqueeze(0)

            critic_est = _critic_model_forward(critic, static_input, dynamic_input)
            node_logp, reward = _actor_model_forward(actor, static_input, dynamic_input, hpcEnv=env)
            
            time_step += 1
            env.time_step += 1
            current_ep_reward += reward
            
            reward = torch.tensor([reward])
            # # reward = torch.from_numpy(reward).float().to(config.device)
            # rewards = torch.cat((rewards, reward),0)
            # logps = torch.stack((logps, node_logp),0)
            # critic_est = torch.cat((critic_ests, critic_est),0)
            rewards.append(reward)
            logps.append(node_logp)
            critic_ests.append(critic_est)

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # printing average reward
            if time_step % print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % update_timestep == 0:
                rewards = torch.cat(rewards, dim=0)
                logps = torch.tensor(logps)
                critic_ests = torch.cat(critic_ests, dim=0)

                advantage = rewards - critic_ests
                # Actor loss computation and backpropagation
                actor_loss = torch.mean(advantage.detach() * logps.sum(dim=1))
                actor_loss.requires_grad_(True)
                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                # Critic loss computation and backpropagation
                critic_loss = torch.mean(advantage ** 2)
                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                rewards = []
                logps = []
                critic_ests = []

            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                model_data = {
                'parameters': actor.state_dict()
                }
                torch.save(model_data, checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")     

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")



if __name__ == '__main__':
    train()

