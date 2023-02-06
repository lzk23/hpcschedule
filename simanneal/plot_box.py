import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
# plt.rcParams['axes.unicode_minus']=False

# matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
plt.rc('font',family='Times New Roman')

files = ['comm_wait_cost_FCFS.txt', 'comm_wait_cost_BF.txt', 'comm_wait_cost_10.txt',
         'comm_wait_cost_20.txt', 'comm_wait_cost_30.txt', 'comm_wait_cost_40.txt',
         'comm_wait_cost_50.txt', 'comm_wait_cost_60.txt']
all_data_comm = []
all_data_wait = []
for file in files:
    f = open('simanneal/'+file, 'r')
    lines = f.readlines()
    data_comm = []
    data_wait = []
    for line in lines:
        data = line.split(',')
        commcost = float(data[1].split(':')[1].strip())
        waitcost = float(data[2].split(':')[1].strip())
        data_comm.append(commcost)
        data_wait.append(waitcost)
    all_data_comm.append(data_comm)
    all_data_wait.append(data_wait)

labels = ['FCFS', 'BF', '10', '20', '30', '40', '50', '60']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# rectangular box plot
bplot1 = ax1.boxplot(all_data_comm,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
# ax1.set_title('Rectangular box plot')

# notch shape box plot
bplot2 = ax2.boxplot(all_data_wait,
                     # notch=True,  # notch shape
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
# ax2.set_title('Notched box plot')

# fill with colors
colors = ['pink', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'azure', 'brown']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

# adding horizontal grid lines
for ax in [ax1, ax2]:
    ax.yaxis.grid(True)
    # ax.set_xlabel('Different scheduling methods and values of time period(/s)')
    # ax.set_ylabel('Observed values')
plt.suptitle('Different scheduling methods and lengths of time period(/s)', y=-0.0001)
ax1.set_ylabel('Avg. CH cost')
ax2.set_ylabel('Avg. waiting time/s')


plt.show()
plt.savefig('simanneal/box.png', dpi=600, bbox_inches = 'tight')