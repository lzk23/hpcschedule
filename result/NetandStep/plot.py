import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np

# plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签 
# plt.rcParams['axes.unicode_minus']=False
plt.rc('font',family='Times New Roman')

steps = [500, 600, 700, 800, 900, 1000]
labels = ['FCN-1', 'FCN-2', 'FCN-3', 'CNN-1', 'CNN-2', 'CNN-3']

def read_data(file_name):
    f = open("result/NetandStep/"+file_name, 'r')
    datas = f.readlines()
    sum_sa = 0
    sum_nsa = 0
    for line in datas[2:]:
        data = line.split(',')
        assert len(data) == 16
        sum_sa += int(data[10])
        sum_nsa += int(data[12])
    avg_sa = sum_sa//10
    avg_nsa = sum_nsa//10
    f.close()
    return avg_sa, avg_nsa



def add_value_labels(ax, typ, spacing=-10):
    space = spacing
    va = 'bottom'

    if typ == 'bar':
        for i in ax.patches:
            y_value = i.get_height()
            x_value = i.get_x() + i.get_width() / 2

            label = "{:.0f}".format(y_value)
            ax.annotate(label,(x_value, y_value), xytext=(0, space), 
                    textcoords="offset points", ha='center', va=va)     
    
    
    if typ == 'line':
        line = ax.lines[0]
        for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
            label = "{:.2f}".format(y_value)
            ax.annotate(label,(x_value, y_value), xytext=(0, space), fontsize=5, color = 'red', bbox=dict(boxstyle="round4", fc="w", ec='w'),
                textcoords="offset points", ha='center', va=va)  



def sub_plot(index, fig, ax):
    
    step = steps[index]
    avgs_sa = []
    avgs_nsa = []
    for layer_type in range(0, 2):
        for layer_index in range(0, 3):
            file_name = 'compare_test1800_40_layer{}_index{}_method1_rescale1_step{}.txt'.format(layer_type, layer_index, step)
            avg_sa, avg_nsa = read_data(file_name)
            avgs_sa.append(avg_sa/10000)
            avgs_nsa.append(avg_nsa/10000)



    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    rects1 = ax.bar(x - width/2, avgs_sa, width, label='Rewards of SA-{}'.format(steps[index]), color='blue', hatch="xxx")
    rects2 = ax.bar(x + width/2, avgs_nsa, width, label='Rewards of NSA-{}'.format(steps[index]), color='green', hatch="///")

    ax.set_ylim(8.8, 10)

    # 绘制折线图
    rate = []
    for i in range(len(avgs_sa)):
        rate.append(round((avgs_sa[i]-avgs_nsa[i])/avgs_sa[i], 4)*100)
    

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    format = '$t^{max}$'
    #ax.set_xlabel('{}={}'.format(format, steps[index]))
    ax.xaxis.set_tick_params(rotation=-30, labelsize=8)

    ax2 = ax.twinx()
    line = ax2.plot(rate, '-o', markersize = '2', label='Improvement', color = 'red')
    
    if index == 0 or index == 3:
        ax.set_ylabel('Cost/($x10^4$)')
        ax.yaxis.set_tick_params(rotation=-30, labelsize=10)
        ax2.set_yticklabels([])

    if index == 2 or index == 5:
        ax2.set_ylabel('Improvement/%')
        ax2.yaxis.set_tick_params(rotation=30, labelsize=10)
        ax.set_yticklabels([])

    if index == 1 or index == 4:
        ax.set_yticklabels([])
        ax2.set_yticklabels([])
    
    ax.legend(loc='upper left', frameon=False, fontsize='6')
    # if index == 5:
    #     ax.legend(loc='upper left', frameon=False, fontsize='6')
    # if index == 0:
    #     ax2.legend(loc='upper left', frameon=False, fontsize='6')
    ax2.set_ylim(-0.4, 1.5)
    # if index == 5:
    #     box_handles = ax.get_legend_handles_labels()
    #     line_handles = ax2.get_legend_handles_labels()
    #     ax.legend(handles=box_handles + line_handles, loc = 'center')
    add_value_labels(ax2, typ='line')


fig, axs = plt.subplots(nrows=2, ncols=3, constrained_layout=False)

for index, ax in enumerate(axs.flat):
    sub_plot(index, fig, ax)

fig.tight_layout()
# plt.show()

# plt.subplots_adjust(left=0,
#                     bottom=0, 
#                     right=0, 
#                     top=0, 
#                     wspace=0.2, 
#                     hspace=0.35)

plt.savefig('result/NetandStep/boxline.png', dpi=600, bbox_inches = 'tight')

# ax=plt.gca()##获取坐标轴信息,gca=get current axic
# print(ax)
# ax.spines['right'].set_color('none')##设置右边框颜色为无
# ax.spines['top'].set_color('none')

# ax.xaxis.set_ticks_position('bottom')##位置有bottom(left),top(right),both,default,none
# ax.yaxis.set_ticks_position('left')##定义坐标轴是哪个轴，默认为bottom(left)
# ax.spines['bottom'].set_position(('data',74 ))##移动x轴，到y=0
# ax.spines['left'].set_position(('data',-0.5))##还有outward（向外移动），axes（比例移动，后接小数）

# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib.patches import Circle
# from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
#                                   AnnotationBbox)
# from matplotlib.cbook import get_sample_data


# fig, ax = plt.subplots()

# # Define a 1st position to annotate (display it with a marker)
# xy = (0.5, 0.7)
# ax.plot(xy[0], xy[1], ".r")

# # Annotate the 1st position with a text box ('Test 1')
# offsetbox = TextArea("Test 1")

# ab = AnnotationBbox(offsetbox, xy,
#                     xybox=(-20, 40),
#                     xycoords='data',
#                     boxcoords="offset points",
#                     arrowprops=dict(arrowstyle="->"))
# ax.add_artist(ab)

# # Annotate the 1st position with another text box ('Test')
# offsetbox = TextArea("Test")

# ab = AnnotationBbox(offsetbox, xy,
#                     xybox=(1.02, xy[1]),
#                     xycoords='data',
#                     boxcoords=("axes fraction", "data"),
#                     box_alignment=(0., 0.5),
#                     arrowprops=dict(arrowstyle="->"))
# ax.add_artist(ab)

# # Define a 2nd position to annotate (don't display with a marker this time)
# xy = [0.3, 0.55]

# # Annotate the 2nd position with a circle patch
# da = DrawingArea(20, 20, 0, 0)
# p = Circle((10, 10), 10)
# da.add_artist(p)

# ab = AnnotationBbox(da, xy,
#                     xybox=(1.02, xy[1]),
#                     xycoords='data',
#                     boxcoords=("axes fraction", "data"),
#                     box_alignment=(0., 0.5),
#                     arrowprops=dict(arrowstyle="->"))

# ax.add_artist(ab)

# # Annotate the 2nd position with an image (a generated array of pixels)
# arr = np.arange(100).reshape((10, 10))
# im = OffsetImage(arr, zoom=2)
# im.image.axes = ax

# ab = AnnotationBbox(im, xy,
#                     xybox=(-50., 50.),
#                     xycoords='data',
#                     boxcoords="offset points",
#                     pad=0.3,
#                     arrowprops=dict(arrowstyle="->"))

# ax.add_artist(ab)

# # Annotate the 2nd position with another image (a Grace Hopper portrait)
# with get_sample_data("grace_hopper.jpg") as file:
#     arr_img = plt.imread(file)

# imagebox = OffsetImage(arr_img, zoom=0.2)
# imagebox.image.axes = ax

# ab = AnnotationBbox(imagebox, xy,
#                     xybox=(120., -80.),
#                     xycoords='data',
#                     boxcoords="offset points",
#                     pad=0.5,
#                     arrowprops=dict(
#                         arrowstyle="->",
#                         connectionstyle="angle,angleA=0,angleB=90,rad=3")
#                     )

# ax.add_artist(ab)

# # Fix the display limits to see everything
# ax.set_xlim(0, 1)
# ax.set_ylim(0, 1)

# plt.show()
