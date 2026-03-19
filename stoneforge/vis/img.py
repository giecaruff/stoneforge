import matplotlib.pyplot as plt
import numpy as np

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
        
import matplotlib.pyplot as plt
import numpy as np

class plotwell:

    def __init__(self, well, depth, curves, colors, units, d_unit='m', size=(12,10)):
        
        self.well = well
        self.depth = depth
        self.curves = curves
        self.colors = colors
        self.units = units
        self.d_unit = d_unit
        self.size = size

        self.extra_tracks = []  # store facies or future tracks

    # -------------------------
    # Add lithology track
    # -------------------------
    def facies(self, lithology, colors, linewidth=0):
        self.extra_tracks.append({
            "type": "facies",
            "lithology": lithology,
            "colors": colors,
            "linewidth": linewidth
        })

    # -------------------------
    # Render everything
    # -------------------------
    def show(self):

        n_tracks = len(self.curves) + len(self.extra_tracks)

        fig, ax = plt.subplots(1, n_tracks, sharey=True)
        fig.set_size_inches(self.size)

        if n_tracks == 1:
            ax = [ax]

        # Depth axis
        ax[0].set_ylabel(f"{self.depth} ({self.d_unit})")
        ax[0].invert_yaxis()

        track_idx = 0

        # -------------------------
        # Extra tracks (facies first)
        # -------------------------
        for track in self.extra_tracks:

            if track["type"] == "facies":
                lith = self.well[track["lithology"]].values
                d = self.well[self.depth].values
                colors = track["colors"]

                unique_lith = np.unique(lith)

                for lith_value in unique_lith:
                    mask = (lith == lith_value)

                    ax[track_idx].fill_betweenx(
                        d,
                        0,
                        1,
                        where=mask,
                        facecolor=colors.get(lith_value, 'gray'),
                        linewidth=track["linewidth"]
                    )

                ax[track_idx].set_title(track["lithology"])
                ax[track_idx].set_xlim(0, 1)
                ax[track_idx].set_xticks([])
                ax[track_idx].grid()

                track_idx += 1

        # -------------------------
        # Curve tracks
        # -------------------------
        for i, c in enumerate(self.curves):
            ax[track_idx].plot(self.well[c], self.well[self.depth], color=self.colors[i])
            ax[track_idx].set_title(c)
            ax[track_idx].set_xlabel(self.units[i])
            ax[track_idx].grid()

            track_idx += 1

        plt.tight_layout()
        plt.show()
        