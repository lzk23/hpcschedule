from tkinter.messagebox import NO
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from config import Machine_State, config
import time

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_cluster(plt:plt, cluster):
    fig, axs = plt.subplots(config.z_unit_num, config.x_unit_num, figsize=
    (config.x_node_num_each_unit, config.y_unit_num))  # axs array shape [x, z]
    # plt.rcParams['figure.figsize']=(1000, 500)
    fig.set_figheight(8)
    fig.set_figwidth(15)
    for z in range(config.z_unit_num):
        for x in range(config.x_unit_num):
            ax = axs[z, x]
            id_data = np.zeros((config.y_unit_num, config.x_node_num_each_unit))
            color_data = np.zeros((config.y_unit_num, config.x_node_num_each_unit))
            if cluster:
                for y in range(config.y_unit_num):
                    for x1 in range(config.x_node_num_each_unit):
                        key = ','.join([str(x*config.x_node_num_each_unit+x1),str(y),str(z)])
                        node_id = cluster.xyz_to_id(key)
                        id_data[y, x1] = node_id
                        if cluster.machines[node_id].state == Machine_State.run_state:
                            color_data[y, x1] = 1
                        # if node_id % 2 == 0:
                        #     color_data[y, x1] = 1

            container_id = ["y {}".format(i) for i in range(config.y_unit_num)]
            node_index_in_one_container = ["x {}".format(i) for i in range(config.x_node_num_each_unit)]
            im= heatmap(data = color_data, row_labels=container_id, col_labels=node_index_in_one_container,
            ax=ax, cmap="Wistia", cbarlabel="taken by job")
            annotate_heatmap(im, data = id_data, valfmt="{x:.0f}", size=9)


    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
    plt.pause(0.5)
    plt.close()


def plot_cluster_with_scheduled_jobs(plt:plt, cluster, scheduled_jobs):
    fig, axs = plt.subplots(config.z_unit_num, config.x_unit_num, figsize=
    (config.x_node_num_each_unit, config.y_unit_num))
    fig.set_figheight(8)
    fig.set_figwidth(15)
    axs_iddata = {}
    axs_colordata = {}
    axs_num = axs.shape[0] * axs.shape[1] if config.z_unit_num > 1 or config.x_unit_num > 1 else 1
    axs = axs.reshape((1, axs_num))[0].tolist() if config.z_unit_num > 1 or config.x_unit_num > 1 else axs

    container_id = ["y {}".format(i) for i in range(config.y_unit_num)]
    node_index_in_one_container = ["x {}".format(i) for i in range(config.x_node_num_each_unit)]
    for axid in range(axs_num):
        axs_iddata[axid] = np.zeros((config.y_unit_num, config.x_node_num_each_unit))
        axs_colordata[axid] = np.zeros((config.y_unit_num, config.x_node_num_each_unit))
    
    jobid_to_color_index = {}
    color_index = 0
    for jobid in scheduled_jobs.keys():
        assert not jobid_to_color_index.__contains__(jobid)
        jobid_to_color_index[jobid] = color_index
        color_index += 1

    for jobid in scheduled_jobs.keys():
        for nodeid in scheduled_jobs[jobid]:
            axz, axx, y, x1 = node_id_to_point_in_plot(cluster, nodeid)
            axid = axz * config.x_unit_num + axx
            axs_iddata[axid][y, x1] = jobid
            axs_colordata[axid][y, x1] = jobid_to_color_index[jobid]
    
    for axid in range(axs_num):
        im = heatmap(data = axs_colordata[axid], 
        row_labels=container_id, col_labels=node_index_in_one_container,
        ax = axs[axid] if config.z_unit_num > 1 or config.x_unit_num > 1 else axs, cmap = "magma_r")
        annotate_heatmap(im, data = axs_iddata[axid], valfmt="{x:.0f}", size=7, textcolors=("red", "green"))

    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.1, hspace=0.1)
    plt.pause(0.5)
    plt.close()

def node_id_to_point_in_plot(cluster, node_id):
    rack_id = int(node_id/cluster.nb_machines_one_rack)
    axz = int(rack_id/config.x_unit_num)
    axx = rack_id - axz * config.x_unit_num
    node_index_in_one_rack = node_id - cluster.nb_machines_one_rack * rack_id
    y = int(node_index_in_one_rack/config.x_node_num_each_unit)
    x1 = node_index_in_one_rack - y * config.x_node_num_each_unit
    return axz, axx, y, x1

if __name__ == '__main__':
    from hpc.cluster import ClusterWithTopology
    cluster = ClusterWithTopology()
    plot_cluster(plt, cluster)