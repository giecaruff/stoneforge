import matplotlib.pyplot as plt

# based on "PythonParaGeofisicos" from Sep 24, 2020

def wellplot(well, depth, curves, colors, units, d_unit='m', size = (12,10)):

    n_tracks = len(curves)

    fig, ax = plt.subplots(1, n_tracks, sharey=True)
    fig.set_size_inches(size)

    ax[0].set_ylabel(depth + ' (' + d_unit + ')')
    ax[0].invert_yaxis()
    ii = 0
    for c in curves:
        ax[ii].plot(well[c], well[depth], color=colors[ii])
        ax[ii].set_title(c)
        ax[ii].set_xlabel(units[ii])
        ax[ii].grid()
        ii += 1

    plt.show()

def plito(lithology,depth,colors,linewidth = 1.):

    lists_dict = {}
    # Populate the dictionary with empty lists for each unique integer value
    for value in colors:
        lists_dict[value] = []
    
    # Iterate through the original list and populate the corresponding lists
    for value in lithology:
        for key in lists_dict.keys():
            if value == key:
                lists_dict[key].append(1)
            else:
                lists_dict[key].append(0)
    
    for _l in lists_dict:
        plt.fill_betweenx(depth, lists_dict[_l], facecolor=colors[_l], linewidth = linewidth)