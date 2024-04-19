import matplotlib.pyplot as plt

# based on "PythonParaGeofisicos" from Sep 24, 2020

def wellplot(well, depth, curves, colors, units, size = (12,10)):

    n_tracks = len(curves)

    fig, ax = plt.subplots(1, n_tracks, sharey=True)
    fig.set_size_inches(size)

    ax[0].set_ylabel(depth + ' (' + units[0] + ')')
    ax[0].invert_yaxis()
    ii = 0
    for c in curves:
        ax[ii].plot(well[c], well[depth], color=colors[ii])
        ax[ii].set_title(c)
        ax[ii].set_xlabel(units[ii])
        ax[ii].grid()
        ii += 1

    plt.show()