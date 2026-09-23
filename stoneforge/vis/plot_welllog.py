import os
import numpy as np
import matplotlib
if os.environ.get("DISPLAY", "") == "":
    backend = matplotlib.get_backend().lower()
    if backend in {"tkagg", "qt5agg", "qtagg", "wxagg", "gtk3agg", "gtkagg"}:
        matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
from typing import Annotated

class LogPlot:

    def __init__(self,
    size: Annotated[tuple, "image size"] = (27.7, 40.0),
    top: Annotated[float, "top depth"] = None,
    bot: Annotated[float, "bottom depth"] = None,
    title: Annotated[tuple, "image title"] =('Well',2),
    res: Annotated[tuple, "resolution"] = 1000,
    dpi: Annotated[int, "dots per inch"] = 200):
        """ Initialize the LogPlot class.
        
        Parameters
        ----------
        size : tuple
            Size of the figure in cm (width, height), standard is (27.7, 40.0).
        top : float
            Top depth value, if None it will be set to min depth in set_depth, standard is None.
        bot : float
            Bottom depth value, if None it will be set to max depth in set_depth, standard is None.
        title : tuple
            Title of the plot and its position (text, x-position), standard is ('Well', 2).
        res : tuple
            Resolution of the depth axis (min, max), standard is 1000, meaning '1/1000'.
        dpi : int
            Dots per inch for the figure, standard is 200 dpi.

        Example
        -------
        >>> plot = LogPlot(size=(29.7, 300.0), top=2000, bot=4000, title=('Well A', 1.12))

        Notes
        -----
        The class provides methods to create various types of well log plots including normal, logarithmic, fill, colormap, crossover, color, and matrix plots.
        Each plot type can be added as a track to the figure with customizable parameters.
        The depth axis must be set using the set_depth method before adding any tracks.
        """

        self.cm = 0.3937
        
        
        self.ax = None
        self.xzeros = []
        self.yzeros = []
        self.widths = []
        self.heights = []
        self._first_track = True
        self.title = title

        self._subs = 0
        self._bar = (int(0.9167 * size[0] + 0.833))*'─'

        self.y = None
        self.depth_description = "Depth"
        self.top = top
        self.bot = bot

        self.res = res
        self.dpi = dpi
        self.fig = plt.figure(figsize=(size[0]*self.cm, size[1]*self.cm))

        if top != None and bot != None:
            self.range = bot - top
            altura_inch = self.range / res * 100 / 2.54
            print(f"Figure size (depths and scales detected): {size[0]*self.cm:.2f} x {altura_inch:.2f} polegadas")
            self.fig.set_size_inches(size[0]*self.cm, altura_inch)
        else:
            self.range = None


    def _format_bar(self, min_val, max_val, label):
        """ Format the title bar with min and max values centered around the label."""
        bar_str = self._bar
        total_length = len(bar_str)
        min_str = str(min_val)
        max_str = str(max_val)
        
        fixed_space = len(min_str) + len(max_str) + 2
        
        remaining_space = total_length - fixed_space
        
        if remaining_space < 0:
            raise ValueError("Bar is too short to fit min and max values")
        
        bar_line = f"{min_str} " + "─" * remaining_space + f" {max_str}"
        visual_length = len(bar_line)
        centered_label = label.center(visual_length)
        formatted_bar = f"{centered_label}\n{bar_line}"
        
        return formatted_bar
    

    def _set_colormapped_title(self, ax, text, cmap_name="viridis", fontsize=10, y=1.04):
        """
        Set a title on the axis where each character is colored according to a colormap.
        
        Parameters:
            ax         : matplotlib axis
            text       : string to display
            cmap_name  : name of the matplotlib colormap (default 'viridis')
            fontsize   : font size
            y          : vertical position relative to axis (default 1.02)
        """
        # Use the new registry access instead of get_cmap
        cmap = mpl.colormaps[cmap_name] 
        n = len(text)

        # Normalize character positions into [0,1]
        norm_positions = np.linspace(0, 1, n)

        # Clear old title
        ax.set_title("")

        # Place each character
        for i, (ch, pos) in enumerate(zip(text, norm_positions)):
            ax.text(
                -1.0 + 2*(i/(n)),   # approximate x-position
                y,
                ch,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=fontsize,
                color=cmap(pos)
            )
    

    def _addtrack(self, w=0.2, track=False):
        """ Add a new track to the figure. """
        if self.xzeros:
            if track:
                self.xzeros.append(self.xzeros[-1])
                self._subs += 1
            else:
                self.xzeros.append(self.xzeros[-1] + self.widths[-1])
                self._subs = 0
        else:
            self.xzeros.append(0.1)

        self.yzeros.append(0.1)
        self.widths.append(w)
        self.heights.append(1.0)
        
        self.ax = self.fig.add_axes([self.xzeros[-1], self.yzeros[-1], self.widths[-1], self.heights[-1]])
        self.ax.patch.set_alpha(0)
        self.ax.set_xticklabels([])
        
        if not self._first_track:
            self.ax.set_yticklabels([])
        else:
            self.ax.set_ylabel(self.depth_description)
        
        self._first_track = False


    def _mask_depth_range(self, x, y, dmin, dmax):
        """ Mask values outside the specified depth range with NaN. """
        x = np.asarray(x)
        y = np.asarray(y)
        xm = np.copy(x)
        xm[(y < dmin) | (y > dmax)] = np.nan
        return xm


    def set_depth(self,
                  y: Annotated[float, "depth values"],
                  d: Annotated[str, "depth description"] = "depth"):
        """ Set the depth axis for the log plots.
        
        Parameters
        ----------
        y : np.ndarray
            1D array of depth values.
        d : str
            Description for the depth axis, standard is "depth".
        res : tuple
            Resolution of the depth axis (min, max), standard is (1, 1000).
        dpi : int
            Dots per inch for the figure, standard is 200.
        
        Example
        -------
        >>> depths # depth array [1000, 1001, ..., 3000]
        >>> plot.set_depth(depths, d="Depth (m)", res=(1, 1000), dpi=300)
        """

        self.y = y
        self.depth_description = d
        
        if self.top == None:
            self.top = np.nanmin(y)
            
        if self.bot == None:
            self.bot = np.nanmax(y)
            
        self.range = self.bot - self.top
        self.depth_range = np.arange(self.top,self.bot,50)


    def normal_plot(self,
                    x: Annotated[np.array, "log values"],
                    track: Annotated[bool, "new track or not"]=False,
                    c: Annotated[str, "log color"]='black',
                    s: Annotated[str, "log line style"]='-',
                    m: Annotated[str, "marker"]='',
                    lw: Annotated[float, "line width"]=.5,
                    w: Annotated[float, "track proportion"]=.2,
                    vmin: Annotated[float, "log minimum value"]=None,
                    vmax: Annotated[float, "log maximum value"]=None,
                    step: Annotated[int, "grid step"]=10,
                    label: Annotated[str, "log label"]='',
                    grid: Annotated[bool, "grid"] = True,
                    ylim: Annotated[tuple, "depth range"] = False):
        """ Create a normal (linear) log plot. 
        
        Parameters
        ----------
        x : np.ndarray
            1D array of log values.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        c : str
            Color of the log line, standard is 'black'.
        s : str
            Line style of the log line, standard is '-'.
        m : str
            Line style marker of the log line, standard is ''.
        lw : float
            Line width of the log line, standard is 0.5.
        w : float
            Track width as a proportion of figure width, standard is 0.2.
        vmin : float
            Minimum value for x-axis, if None it will be set to min of x.
        vmax : float
            Maximum value for x-axis, if None it will be set to max of x.
        step : int
            Number of grid steps on x-axis, standard is 10.
        label : str
            Label for the log, shown in title, standard is '' (no label).
        grid : bool
            If True, will show grid lines (standard is True).
        ylim : tuple
            Depth range to display (min_depth, max_depth), if False will use full depth range.

        Example
        -------
        >>> plot.normal_plot(log_values, track=False, c='blue', s='-', lw=0.5, w=0.2, vmin=0, vmax=100, step=10, label='Gamma Ray', grid=True, ylim=(1500, 2500))
        """
        
        self._addtrack(w=w, track=track)
        if not track and grid == True:
            self.ax.minorticks_on()
            self.ax.grid(which='major',axis = 'y', linewidth=1.8)
            self.ax.grid(which='major',axis = 'x', linewidth=0.4)
            self.ax.grid(which='minor',axis = 'y', linewidth=0.4)
        if vmin == None:
            vmin = np.nanmin(x)
        if vmax == None:
            vmax = np.nanmax(x)

        if ylim:
            x = self._mask_depth_range(x, self.y, ylim[0], ylim[1])
            
        _title = self._format_bar(vmin, vmax, label)
        self.ax.set_xticks(np.linspace(vmin,vmax,step+1))
        self.ax.set_yticks(self.depth_range)
        self.ax.plot(x, self.y, color=c, linestyle=s, linewidth=lw, marker = m)
        self.ax.set_xlim(vmin,vmax)
        self.ax.set_ylim(self.bot,self.top)
        if label:
            self.ax.set_title(_title+'\n'*self._subs*2, color = c, fontsize=10)


    def logarithm_plot(self,
                       x: Annotated[np.array, "log values"],
                       track: Annotated[bool, "new track or not"]=False,
                       c: Annotated[str, "log color"]='black',
                       s: Annotated[str ,"log line style"]='-',
                       lw: Annotated[float, "linewidth"]=.5,
                       w: Annotated[float, "track proportion"]=.2,
                       vmin: Annotated[float, "log minimum value"]=None,
                       vmax: Annotated[float, "log maximum value"]=None,
                       step: Annotated[int, "grid step"]=10,
                       label: Annotated[str, "log label"]='',
                       grid: Annotated[bool, "grid"] = False,
                       ylim: Annotated[tuple, "depth range"] = False):
        """ Create a logarithmic log plot.
        
        Parameters
        ----------
        x : np.ndarray
            1D array of log values.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        c : str
            Color of the log line, standard is 'black'.
        s : str
            Line style of the log line, standard is '-'.
        lw : float
            Line width of the log line, standard is 0.5.
        w : float
            Track width as a proportion of figure width, standard is 0.2.
        vmin : float
            Minimum value for x-axis, if None it will be set to min of x.
        vmax : float
            Maximum value for x-axis, if None it will be set to max of x.
        step : int
            Number of grid steps on x-axis, standard is 10.
        label : str
            Label for the log, shown in title, standard is '' (no label).
        grid : bool
            If False, will remove grids (logarithm plots don’t show grids naturally).
        ylim : tuple
            Depth range to display (min_depth, max_depth), if False will use full depth range.

        Example
        -------
        >>> plot.logarithm_plot(log_values, track=False, c='red', s='--', lw=0.5, w=0.2, vmin=0.1, vmax=1000, step=10, label='Resistivity', grid=False, ylim=(1500, 2500))
        """
        self._addtrack(w=w, track=track)
        if not track and grid == False:
            self.ax.minorticks_on()
            self.ax.grid(which='major',axis = 'y', linewidth=1.8)
            self.ax.grid(which='major',axis = 'x', linewidth=0.8)
            self.ax.grid(which='minor',axis = 'y', linewidth=0.4)
            self.ax.grid(which='minor',axis = 'x', linewidth=0.4)
        if vmin == None:
            vmin = np.nanmin(x)
        if vmax == None:
            vmax = np.nanmax(x)

        if ylim:
            x = self._mask_depth_range(x, self.y, ylim[0], ylim[1])
            
        _title = self._format_bar(vmin, vmax, label)
        self.ax.set_xticks(np.linspace(vmin,vmax,step+1))
        self.ax.set_yticks(self.depth_range)
        self.ax.semilogx(x, self.y, color=c, linestyle=s, linewidth=lw)
        self.ax.set_xlim(vmin,vmax)     
        self.ax.set_xticklabels([])
        self.ax.set_ylim(self.bot,self.top)
        if label:
            self.ax.set_title(_title+'\n'*self._subs*2, color = c, fontsize=10)


    def fill_plot(self,
                  x: Annotated[np.array, "log values"],
                  track: Annotated[bool, "new track or not"]=False,
                  c: Annotated[str, "log color"]='black',
                  s: Annotated[str, "log line style"]='-',
                  lw: Annotated[float, "line width"]=.5,
                  w: Annotated[float, "track proportion"]=0.2,
                  vmin: Annotated[float, "log minimum value"]=None,
                  vmax: Annotated[float, "log maximum value"]=None,
                  step: Annotated[int, "grid step"]=10,
                  label: Annotated[str, "log label"]='',
                  grid: Annotated[bool, "grid"]=False):
        """ Create a filled log plot.
        
        Parameters
        ----------
        x : np.ndarray
            1D array of log values.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        c : str
            Color of the log line and fill, standard is 'black'.
        s : str
            Line style of the log line, standard is '-'.
        lw : float
            Line width of the log line, standard is 0.5.
        w : float
            Track width as a proportion of figure width, standard is 0.2.
        vmin : float
            Minimum value for x-axis, if None it will be set to min of x.
        vmax : float
            Maximum value for x-axis, if None it will be set to max of x.
        step : int
            Number of grid steps on x-axis, standard is 10.
        label : str
            Label for the log, shown in title, standard is '' (no label).
        grid : bool
            If False, will remove grids (fill plots don’t show grids naturally).

        Example
        -------
        >>> plot.fill_plot(log_values, track=False, c='green', s='-', lw=0.5, w=0.2, vmin=0, vmax=200, step=10, label='Porosity', grid=False)
        """

        self._addtrack(w=w, track=track)
        if not track and grid == False:
            self.ax.minorticks_on()
            self.ax.grid(which='major',axis = 'y', linewidth=1.8)
            self.ax.grid(which='major',axis = 'x', linewidth=0.4)
            self.ax.grid(which='minor',axis = 'y', linewidth=0.4)
        if vmin == None:
            vmin = np.nanmin(x)
        if vmax == None:
            vmax = np.nanmax(x)
        _title = self._format_bar(vmin, vmax, label)
        self.ax.set_xticks(np.linspace(vmin,vmax,step+1))
        self.ax.set_yticks(self.depth_range)
        self.ax.plot(x, self.y, color=c, linestyle=s, linewidth=lw)
        self.ax.fill_betweenx(self.y, x, vmax, color=c)
        self.ax.set_xlim(vmin,vmax)
        self.ax.set_ylim(self.bot,self.top)
        if label:
            self.ax.set_title(_title+'\n'*self._subs*2, color = c, fontsize=10)
    

    def fill_cmap_plot(self,
                    x: Annotated[np.ndarray, "log values"],
                    track: Annotated[bool, "new track or not"]=False,
                    cmap: Annotated[str, "matplotlib colormap"]='viridis',
                    w: Annotated[float, "track proportion"]=0.2,
                    vmin: Annotated[float, "log minimum value"]=None,
                    vmax: Annotated[float, "log maximum value"]=None,
                    step: Annotated[int, "grid step"]=10,
                    label: Annotated[str, "log label"]=True,
                    grid: Annotated[bool, "grid"]=False):
        """ Create a filled log plot with a colormap.

        Parameters
        ----------
        x : np.ndarray
            1D array of log values.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        cmap : str
            Colormap name (default 'viridis').
        w : float
            Track width as a proportion of figure width, standard is 0.2.
        vmin : float
            Minimum value for color scaling (default: min of x).
        vmax : float
            Maximum value for color scaling (default: max of x).
        step : int
            Number of grid steps on x-axis, standard is 10.
        label : str
            Label for the log, shown in title, standard is '' (no label).
        grid : bool
            If False, will remove grids.
        """

        self._addtrack(w=w, track=track)
        if not track and grid == False:
            self.ax.minorticks_on()
            self.ax.grid(which='major', axis='y', linewidth=1.8)
            self.ax.grid(which='major', axis='x', linewidth=0.4)
            self.ax.grid(which='minor', axis='y', linewidth=0.4)
    
        # Handle vmin/vmax
        if vmin is None:
            vmin = np.nanmin(x)
        if vmax is None:
            vmax = np.nanmax(x)

        y = self.y
    
        #_title = self._format_bar(vmin, vmax, label)
        self.ax.set_xticks(np.linspace(vmin, vmax, step+1))
        self.ax.set_yticks(self.depth_range)
    
        # ============================= #
        # Create colored stripe using imshow
    
        # Define grid
        x_grid = np.linspace(vmin, vmax, 500)  # 200 horizontal samples
        
        # Normalize x values for colormap
        norm = mcolors.Normalize(vmin, vmax)
        cmap = plt.get_cmap(cmap)
        
        color_array = np.full((len(self.y), len(x_grid)), np.nan)
        
        for i in range(len(self.y)):
            x_val = x[i]
            idx = np.where(x_grid <= x_val)[0]
            if len(idx) > 0:
                color_array[i, idx] = norm(x_val)  # Same normalized value across row
        
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        
        im = self.ax.imshow(
            color_array,
            cmap=cmap,
            aspect='auto',
            origin='upper',
            extent=[vmin, vmax, ymax, ymin]
        )
        # ============================= #
    
        # Plot the black line on top
        self.ax.plot(x, self.y, color='black', linewidth=0.1)
    
        self.ax.set_xlim(vmin, vmax)
        self.ax.set_ylim(self.bot, self.top)
    
        if label:
            # create inset axis for colorbar above this track
            self.ax.set_title("---", color = 'k', fontsize=10)
            cax = self.ax.inset_axes([0, 1.005, 1, 0.017])  # [x0, y0, width, height] in axis coords # h1, 1.035, 1.065
            cbar = self.ax.figure.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=0, length=2)


    def crossover_plot(self,
                       x1: Annotated[np.array, "first log values"],
                       x2: Annotated[np.array, "second log values"],
                       track: Annotated[bool, "new track or not"]=False,
                       cmap: Annotated[str, "color map"]='seismic',
                       s: Annotated[str, "log line style"]='-',
                       lw: Annotated[float, "line width"]=.5,
                       w: Annotated[float, "track proportion"]=.2,
                       vmin: Annotated[float, "log minimum value"]=0,
                       vmax: Annotated[float, "log maximum value"]=1,
                       label: Annotated[str, "log label"]='',
                       grid: Annotated[bool, "grid"]=False):
        """ Create a crossover log plot between two logs.

        Parameters
        ----------
        x1 : np.ndarray
            1D array of first log values.
        x2 : np.ndarray
            1D array of second log values.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        cmap : str
            Colormap name for the filled area, standard is 'seismic'.
        s : str
            Line style of the log lines, standard is '-'.
        lw : float
            Line width of the log lines, standard is 0.5.
        w : float
            Track width as a proportion of figure width, standard is 0.2.
        vmin : float
            Minimum value for color scale, standard is 0.
        vmax : float
            Maximum value for color scale, standard is 1.
        label : str
            Label for the log, shown in title, standard is '' (no label).
        grid : bool
            If False, will remove grids (crossover plots don’t show grids naturally).

        Example
        -------
        >>> plot.crossover_plot(log1_values, log2_values, track=False, cmap='seismic', s='-', lw=0.5, w=0.2, vmin=0, vmax=1, label='Crossover Plot', grid=False)
        """
    
        self._addtrack(w=w, track=track)
    
        if not track and grid == False:
            self.ax.minorticks_on()
            self.ax.grid(which='major', axis='y', linewidth=1.8)
            self.ax.grid(which='major', axis='x', linewidth=0.4)
            self.ax.grid(which='minor', axis='y', linewidth=0.4)

        x3 = x2 - x1
        x3 = (x3 + 0.2) / ( 0.5 + 0.2)
        cmap = plt.get_cmap('seismic')
        
        width = 100
        height = len(self.y)
        
        xmin = 0
        xmax = 1

        y = self.y
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
        
        color_array = np.full((height, width), np.nan)
        x_bins = np.linspace(xmin, xmax, width)
    
        for i in range(height):
        
            if np.isnan(x1[i]) and np.isnan(x2[i]):
                x_start = 0.
                x_end = 0.
            else:
                x_start = np.nanmin(np.array([x1[i], x2[i]]))
                x_end = np.nanmax(np.array([x1[i], x2[i]]))
        
            col_indices = np.where((x_bins >= x_start) & (x_bins <= x_end))[0]
            color_array[i, col_indices] = x3[i]
        
        plt.imshow(
            color_array,
            cmap=cmap,
            aspect='auto',
            origin='upper',
            vmin = vmin,
            vmax = vmax,
            extent=[0., 1., ymax, ymin]
        )

        self.ax.plot(x1, self.y, color='black', linestyle=s, linewidth=lw)
        self.ax.plot(x2, self.y, color='black', linestyle=s, linewidth=lw)
    
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(self.bot, self.top)
        self.ax.set_xticks([])
        self.ax.set_yticks(self.depth_range)
    
        if label:
            self.ax.set_title(self.title + '\n' * self._subs * 2, color='black', fontsize=10)


    def color_plot(self,
                   x_color: Annotated[np.array, "color values"],
                   track: Annotated[bool, "new track or not"]=False,
                   w: Annotated[float, "track proportion"]=.2,
                   rule: Annotated[str, "filling order"]="down"
                   ):
        """
        Plots a 2D color (RGB) matrix along depth axis using fill between.
    
        Parameters
        ----------
        x_color : np.ndarray
            2D matrix of shape (n_depths, 3 columns [R ,G ,B]) representing color data.
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        w : float
            Track width.
        rule : str
            Filling rule; 'mean' fill between provided depths; 'up' fill between up range; 'down' (standard) fill between down range. 

        Example
        -------
        >>> plot.color_plot(color_values, track=False, w=0.2)
        """
        self._addtrack(w=w, track=track)

        _intr = []
        for i in range(len(self.y) - 1):
            _intr.append(self.y[i+1] - self.y[i])
        _intr = np.mean(np.array(_intr))

        if rule == 'mean':
            _depth = np.array(self.y, float)

        elif rule == 'up':
            _value = float(self.y[0] - _intr)
            _depth = np.insert(self.y, 0, _value)

        elif rule == 'down':
            _value = float(self.y[-1] + _intr)
            _depth = np.append(self.y, _value)

        for i in range(len(_depth) - 1):
            self.ax.fill_betweenx(
                [_depth[i], _depth[i+1]],
                0, 1,
                color=x_color[i],
                linewidth=0
            )
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(self.bot, self.top)


    def compositional_plot(self,
                           data: Annotated[dict, "data dictionary"],
                           colors: Annotated[dict, "colors dictionary"],
                           track: Annotated[bool, "new track or not"]=False,
                           w: Annotated[float, "track proportion"]=.2,
                           spacing: Annotated[float, "spacing between bars"]=1.):
        """
        Plots a compositional bar plot along depth axis.

        data should be structure in:
        {'Comp1': array([0.      , 0.       ... ,0.002002, 0.002002]),
         'Comp2': array([0.0179, 0.01798202, 0.01798202, ... , 0.        ,        0.        ]),
         ...
         'Compn': array([0.0959041 , 0.0959041 , ..., 0.04504505, 0.04504505])

        colors ahould be structured in:
        {'Comp1': (1.0, 0.2, 0.8),
         'Comp2': (0.6196078431372549, 0.49019607843137253, 0.3254901960784314),
         ...
         'Compn': (0.6901960784313725, 0.6901960784313725, 0.9411764705882353)

        Parameters
        ----------
        data : dict
            data dictionary
        colors: dict
            colors dictionary (should contain the same keys as in 'data')
        track : bool
            Whether to overlay this track on the previous or not (standard is 'False').
        w : float
            Track width.
        spacing : float
            Spacing between bars (0-1), standard is 1.0 (no spacing).
        """

        self._addtrack(w=w, track=track)

        bottom = [0] * len(self.y)
        for comp in data:
            self.ax.barh(
                self.y,
                data[comp],
                left=bottom,
                color=colors[comp],
                edgecolor="none",
                height=(self.y[1] - self.y[0]) * spacing
            )
            bottom = [i + j for i, j in zip(bottom, data[comp])]
        self.ax.set_ylim(self.bot, self.top)
        self.ax.set_xlim(0, 1)


    def matrix_plot(self, M, track=False, cmap='viridis', w=0.2, label=True,vmin=None, vmax=None):
        """
        Plots a 2D matrix (M) vertically along depth axis using imshow.
    
        Parameters
        ----------
        M : np.ndarray
            2D matrix of shape (n_depths, n_columns) representing log/image data.
        y : np.ndarray
            1D array of depth values (length = n_depths).
        track : bool
            Ignored for now.
        cmap : str
            Colormap name for imshow.
        w : float
            Track width.
        pos : bool
            Whether to overlay this track on the previous.
        label : str
            Label for title.
        grid : bool
            If False, will remove grids (imshow doesn’t show grids naturally).
        """
        self._addtrack(w=w, track=track)
    
        M = np.asarray(M)
        y = self.y

        if vmin is None:
            vmin = np.nanmin(M)
        if vmin is None:
            vmax = np.nanmax(M)
    
        if M.shape[0] != len(y):
            raise ValueError(f"Matrix row count {M.shape[0]} must match y length {len(y)}")
        
        ymin = np.nanmin(y)
        ymax = np.nanmax(y)
    
        xmin, xmax = 0, 1
        #_title = self._format_bar(-10., 2000., label, matrix=True)
    
        im = self.ax.imshow(
            M,
            aspect='auto',
            extent=[xmin, xmax, ymax, ymin],
            cmap=cmap,
            interpolation='none',
            origin='upper',
            vmin=vmin,
            vmax=vmax
        )
    
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(self.bot, self.top)
        self.ax.set_xticks([])
        self.ax.set_yticks(self.depth_range)
    
        if label:
            # create inset axis for colorbar above this track
            cax = self.ax.inset_axes([0, 1.035, 1, 0.017])  # [x0, y0, width, height] in axis coords
            cbar = self.ax.figure.colorbar(im, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=0, length=2)
            self.ax.set_title("---", color = 'k', fontsize=10)


    def show(self, title = False):
        """ Display the figure """
        print("Displaying figure... depth range:{self.range}")
        self.fig.suptitle(self.title[0], fontsize=40, x=0.65, y=self.title[1])
        plt.show()


    def save(self,filetype = 'pdf'):
        print("saving figure... depth range:",self.range,"dpi:",self.dpi)
        plt.savefig(filetype, bbox_inches='tight', dpi=self.dpi)