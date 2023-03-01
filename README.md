**This code is for the paper, "Optimization of Topology-Aware Job Allocation on a High-Performance Computing Cluster by Neural Simulated Annealing"(https://arxiv.org/abs/2302.03517), accepted by International Conference on Cluster Computing(2023).

# What it can do?
Jobs on high-performance computing (HPC) clusters can suffer significant performance degradation due to inter-job network interference. Topology-aware job allocation problem (TJAP) is such a problem that decides how to dedicate nodes to specific applications to mitigate inter-job network interference. In this paper, we study the window-based TJAP on a fat-tree network aiming at minimizing the cost of communication hop, a defined inter-job interference metric. The window-based approach for scheduling repeats periodically taking the jobs in the queue and solving an assignment problem that maps jobs to the available nodes. Two special allocation strategies are considered, i.e., static continuity assignment strategy (SCAS) and dynamic continuity assignment strategy (DCAS). For the SCAS, a 0-1 integer programming is developed. For the DCAS, an approach called neural simulated algorithm (NSA), which is an extension to simulated algorithm (SA) that learns a repair operator and employs them in a guided heuristic search, is proposed. The efficacy of NSA is demonstrated with a computational study against SA and SCIP. The results of numerical experiments indicate that both the model and algorithm proposed in this paper are effective.

# Getting Started
## Comparison for the NN architectures of RL

(FCN_1:) python main.py --DRL --PPO --layer_type 0 --layer_index 0 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40
(FCN_2:) python main.py --DRL --PPO --layer_type 0 --layer_index 1 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40
(FCN_3:) python main.py --DRL --PPO --layer_type 0 --layer_index 2 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40
(CNN_1:) python main.py --DRL --PPO --layer_type 1 --layer_index 0 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40
(CNN_2:) python main.py --DRL --PPO --layer_type 1 --layer_index 1 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40
(CNN_3:) python main.py --DRL --PPO --layer_type 1 --layer_index 3 --train --re_scale_state --method_find_nodes 1 --test_name 1800_40

To plot Fig. 4, please run: python PPO/plot_graph.py

![Comparison for the NN architectures of RL](./fig/Fig4.png)

## Comparison of SA and NSA, SCIP and NSA

To get the results of Table 3 and Table 4, please run:

python main.py --DRL --PPO --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --compare

To get the results of Fig.5, please run:

(FCN_1,t_max=500):python main.py --DRL --PPO --layer_type 0 --layer_index 0 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500
(FCN_2,t_max=600):python main.py --DRL --PPO --layer_type 0 --layer_index 1 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 600
(FCN_3,t_max=700):python main.py --DRL --PPO --layer_type 0 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 700
(CNN_1,t_max=800):python main.py --DRL --PPO --layer_type 1 --layer_index 0 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 800
(CNN_2,t_max=900):python main.py --DRL --PPO --layer_type 1 --layer_index 1 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 900
(CNN_3,t_max=1000):python main.py --DRL --PPO --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 1000

To plot Fig. 5, please run:
python result/NetandStep/boxline.png

![Comparison of SA and NSA, SCIP and NSA](./fig/Fig5.png)


## Comparison of different scheduling methods


To get the results of Fig. 6:

(NSA,length of time period=10:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 10
(NSA,length of time period=20:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 20
(NSA,length of time period=30:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 30
(NSA,length of time period=40:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 40
(NSA,length of time period=50:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 50
(NSA,length of time period=60:)python main.py --schedule_type Batch --layer_type 1 --layer_index 2 --re_scale_state --method_find_nodes 1 --test_name 1800_40 --test_with_SA --stepsforsa 500 --time_slot 60
(FCFS)python main.py --schedule_type FCFS
(BF)python main.py --schedule_type BF

To plot Fig.6, please run:
python simanneal/plot_box.py

![Comparison of different scheduling methods](./fig/Fig6.png)


