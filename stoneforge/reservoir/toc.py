# -*- coding: utf-8 -*-
# Made By Fernando Vizeu for GIECAR/UFF in 2017

from __future__ import print_function
from ..io import las2, tabr

# Python 2 compatibility from: https://github.com/oxplot/fysom/issues/1
try:
    unicode = unicode
except NameError:
    # 'unicode' is undefined, must be Python 3
    str = str
    unicode = str
    bytes = bytes
    basestring = (str,bytes)
    raw_input = input
else:
    # 'unicode' exists, must be Python 2
    str = str
    unicode = unicode
    bytes = str
    basestring = basestring
    raw_input = input

import matplotlib
matplotlib.use('TkAgg')

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import NullFormatter, NullLocator, MaxNLocator
import codecs
import json
import pandas as pd

from ._lithocodes import codes
from ._lithopatterns import patterns

LITHOCODES = codes
LITHOPATTERNS = patterns

from ._mplwidgets import BaselinePicker, LiveLine, LOMPicker, DepthController

CONFIGURATION = {
"labdata":
    {
    "filename": "data.csv",
    "delimiter": ",",
    "decimal": ".",
    "headerlines": 1,
    "nullstr": "-",
    "rows":
        {
        "well": 0,
        "top": 1,
        "base": 2,
        "toc": 3
        }
    },
"logdata":
    {
    "lasfilesdir": "DADOS",
    "mnemoics":
        {
            "depth": "DEPTH",
            "dt": "DT",
            "rt": "RT90",
            "gr": "GR",
            "cali": "CAL"
        }
    },
"visualization":
    {
    "limits":
        {
            "dt": [200.0, 0.0],
            "logrt": [-1.0, 3.0],
            "gr": [0.0, 150.0],
            "dtlogr": [-1.0, 3.0],
            "toc": [0.0, 10.0],
            "cali": [6.0, 16.0]
        },
    "colors":
        {
            "dt": "gray",
            "logrt": "red",
            "gr": "green",
            "toc": "blue",
            "labtoc": "red",
            "baseline": "blue",
            "cali": "gray"
        }
    },
"initialparameters":
    {
        "lom": 10.0,
        "dtbaseline": 100.0,
        "logrtbaseline": 0.0
    },
"others":
    {
        "smoothing":
        {
            "smooth": True,
            "show": False,
            "windowsize": 101,
            "window": ["gaussian", 20.0]
        },
        "resampling":
        {
            "resample": True,
            "npoints": 1000
        },
        "nolito": -1
    }
}

if int(matplotlib.__version__[0]) >= 2:
    colorsdict = dict(blue="C0", orange="C1", green="C2", red="C3", purple="C4",
                      brown="C5", pink="C6", gray="C7", yellow="C8", cyan="C9")
else:
    colorsdict = dict(blue="b", orange="orange", green="g", red="r", purple="purple",
                      brown="brow", pink="m", gray="k", yellow="y", cyan="c")

#with open('litocodes.json', 'r') as f:
#    litocodes = json.load(f)

#with open('litopatterns.json', 'r') as f:
#    litopatterns = json.load(f)

# TODO: parametrizar linewidht, fontsize, etc...
_FONTSIZE = 20.0
_DEPTHSHIFT = 345.0

### BEGIN: FUNCTIONS ###

def getstdwellname(wellname):
    return " ".join(wellname.split())

def getdisplayname(name, unit):
    if unit is None:
        return name.strip()
    unit = unit.strip()
    if unit:
        return "{} ({})".format(name.strip(), unit)
    else:
        return name.strip()

def mergelogs(logs):
    mergedlog = np.empty(len(logs[0]))
    mergedlog[:] = np.nan
    
    for log in logs:
        w = np.isfinite(log)
        mergedlog[w] = log[w]
    
    return mergedlog

def baselinedatatolog(depth, baselinex, baseliney):
    log = np.empty(depth.shape[0], dtype=float)
    log[:] = np.nan
    
    for x, y in zip(baselinex, baseliney):
        ymin, ymax = sorted(y)
        where = (depth >= ymin)*(depth <= ymax)
        log[where] = x
    
    return log

def passeymethod(dt, logrt, dtbaseline, logrtbaseline, lom):
    dlogrt = (logrt - logrtbaseline) + 0.02*(dt - dtbaseline)
    toc = dlogrt*10**(2.297 - 0.1688*lom)
    return np.clip(toc, 0.0, 100.0)

def logplot(ax, depth, log, color="C0", xlim=None, ylim=None, style='-'):
    if style in ('-', '--', '-.', ':'):
        linestyle = style
        marker = None
    else:
        linestyle = ''
        marker = style
    line, = ax.plot(log, depth, c=color, ls=linestyle, marker=marker)
    ax.grid(True)
    
    if xlim:
        ax.set_xlim(xlim)
    else:
        xlim = ax.get_xlim()
    
    if ylim:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()
    
    ax.tick_params(labelbottom=False)
    ax.tick_params(labelleft=False)
    ax.tick_params(width=0.0)
    
    return line

def getdepthrect(ax, distance, width):
    bottom = ax.get_position().y0
    top = ax.get_position().y1
    left = ax.get_position().x0 - width + distance
    return [left, bottom, width, top-bottom]

def getlegendrect(ax, distance, height):
    left = ax.get_position().x0
    right = ax.get_position().x1
    bottom = ax.get_position().y1 + distance
    return [left, bottom, right-left, height]

def loglegend(ax, label, xlim, color, style, fontsize, linewidth=None):
    ax.text(0.01, 0.01, str(xlim[0]).replace('.', ','), fontsize=fontsize, ha='left', va='bottom')
    ax.text(0.99, 0.01, str(xlim[1]).replace('.', ','), fontsize=fontsize, ha='right', va='bottom')
    ax.text(0.5, 0.56, label, fontsize=fontsize, ha='center', va='bottom')
    
    if style in ('-', '--', '-.', ':'):
        linestyle = style
        marker = None
        x = [0.25, 0.75]
        y = [0.5, 0.5]
    else:
        linestyle = ''
        marker = style
        x = [1.0/3.0, 0.5, 2.0/3.0]
        y = [0.5, 0.5, 0.5]
    
    if linewidth is None:
        linewidth = 1.0

    ax.plot(x, y, c=color, ls=linestyle, marker=marker, lw=linewidth)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    ax.xaxis.set_major_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

def depthlegend(ax, label, fontsize, rotation=90.0):
    ax.text(0.5, 0.5, label, fontsize=fontsize, ha='center', va='center', rotation=rotation)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    
    ax.xaxis.set_major_locator(NullLocator())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())

def emptytrack(ax, depthlim):
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim(depthlim)
    
    ax.tick_params(labelbottom=False)
    ax.tick_params(labelleft=False)
    ax.tick_params(width=0.0)

def classificationplot(ax, depth, classification, ylim=None):
    if ylim:
        ax.set_ylim(ylim)
    
    ax.set_xlim([0.0, 1.0])
    ax.tick_params(labelbottom=False)
    ax.tick_params(labelleft=False)
    ax.tick_params(width=0.0)

    classes = np.unique(classification)
    
    for cls in classes:
        where = classification == cls
        name = LITHOCODES['codigo']['{:0>3}'.format(cls)]['nome'].lower()
        color = LITHOPATTERNS[name]['color']
        color = [a/255.0 for a in color]
        hatch = LITHOPATTERNS[name]['hatch']
        ax.fill_betweenx(depth, 0.0, 1.0, where, color=color, hatch=hatch, edgecolor='k', linewidth=0.0)

### END: FUNCTIONS ###

class content():

    def __init__(self, config = False):
        config = {}
        if not config:
            config = CONFIGURATION
        else:
            config = config

        self.config = config

        print("Loading configuration file.")

        print("")

        print("Reading CSV file.")

        self.csvfilename = config["labdata"].pop("filename")
        self.csvrows = config["labdata"].pop("rows")
        self.csvpars = config.pop("labdata")
        
        self.lasfilesdir = config["logdata"].pop("lasfilesdir")
        self.mnemoics = config["logdata"].pop("mnemoics")
        self.nolito = config["others"].pop("nolito")
        self.depthmnem = self.mnemoics.pop("depth")

    def load_lab_csv(self, csvfilename):
        
        csvfile = tabr.TABParser(csvfilename)

        csvheader = []
        csvdata = []
        for i in csvfile.data.keys():
            csvheader.append([i])
            csvdata.append(csvfile.data[i]['values'])

        wells = np.array(csvdata[self.csvrows.pop("well")], dtype=unicode)
        wellnames = np.unique(wells)

        labdata = {}
        for wellname in wellnames:
            
            labdata[wellname] = {}
            where = wells == wellname
            for key, index in self.csvrows.items():
                labdata[wellname][key] = np.array(csvdata[index], dtype=float)[where]

        self.labdata = labdata

    def load_lab_las(self, lasfilesdir):

        print("Starting LAS files reading.")
        self.lasfilenames = os.listdir(lasfilesdir)

        litomnem = False #mnemoics.pop("lito")
        logdata = {}

        for lasfilename in self.lasfilenames:
            print("Reading {} ...".format(lasfilename), end=" ")
            
            #lasfile = LAS.open(os.path.join(lasfilesdir, lasfilename), 'r')
            #lasfile.read()

            lasfile = las2.LAS2Parser(os.path.join(lasfilesdir, lasfilename))
            #print(lasfile.data)
            
            wellname = os.path.splitext(lasfilename)[0]
            welldata = {}
            
            #depthidx = lasfile.data[depthmnem]['values']
            
            depthdata = {}
            depthdata["name"] = self.depthmnem
            depthdata["unit"] = lasfile.data[self.depthmnem]['unit']
            depthdata["data"] = lasfile.data[self.depthmnem]['values']
            depthdata["displayname"] = getdisplayname(depthdata["name"], depthdata["unit"])
            
            welldata["depth"] = depthdata
            
            if litomnem in lasfile.data.keys():
                litoidx = lasfile.curvesnames.index(litomnem)
                litodata = {}
                litodata["name"] = litomnem
                litodata["unit"] = ''
                
                litoarray = lasfile.data[litoidx]
                isnan = np.isnan(litoarray)
                litoarray[isnan] = self.nolito
                litoarray = litoarray.astype(int)
                
                litodata["data"] = litoarray
                litodata["displayname"] = litomnem
                
                welldata["lito"] = litodata
            
            for curvekey, mnem in self.mnemoics.items():
                curvedata = {}
                curvedata["name"] = mnem
                curvedata["unit"] = None
                curvedata["data"] = []
                curvedata["displayname"] = mnem
                
                for idx, curvename in enumerate(lasfile.data.keys()):
                    if not curvename.startswith(mnem):
                        continue
                    if curvedata["unit"] is None:
                        curvedata["unit"] = lasfile.data[curvename]['unit']
                        #curvedata["displayname"] = getdisplayname(curvedata["name"], curvedata["unit"])
                        curvedata["displayname"] = curvename
                    
                    curvedata["data"].append(lasfile.data[curvename]['values'])
                
                if len(curvedata["data"]) > 1:
                    curvedata["data"] = mergelogs(curvedata["data"])
                elif len(curvedata["data"]) == 1:
                    curvedata["data"] = curvedata["data"][0]
                
                welldata[curvekey] = curvedata
                
            logdata[wellname] = welldata
            self.logdata = logdata
            print("Done!")

    def view(self):

        print("Begining interactive Passey Method.")

        wellnameslist = list(sorted(self.logdata.keys()))
        choosewellprompt = "Choose well:\n{}\n> ".format('\n'.join(["{}. {}".format(a, b) for a, b in enumerate(["Close"] + wellnameslist)]))

        limits = self.config["visualization"].pop("limits")
        colors = self.config["visualization"].pop("colors")

        initialparameters = self.config.pop("initialparameters")

        smoothing = self.config["others"].pop("smoothing")
        resampling = self.config["others"].pop("resampling")


        while True:
            resp = raw_input(choosewellprompt)
            print("")
            try:
                resp = int(resp)
            except:
                print("Invalid choice!")
                continue
            
            if resp == 0:
                print("Closing program.")
                break
            elif resp > len(wellnameslist) or resp < 0:
                print("Invalid choice!")
                continue
            else:
                wellname = wellnameslist[resp-1]
            
            if wellname not in self.labdata:
                print("Skipping well {}: no lab data.".format(wellname))
                continue
            
            print("Well {}.".format(wellname))
            
            labdepth = self.labdata[wellname]['top']
            labtoc = self.labdata[wellname]['toc']
            
            depth = self.logdata[wellname]["depth"]['data']
            dt = self.logdata[wellname]["dt"]['data']
            logrt = np.log10(self.logdata[wellname]["rt"]['data'])
            gr = self.logdata[wellname]["gr"]['data']
            cali = self.logdata[wellname]["cali"]['data']
            
            #if 'lito' in logdata[wellname]:
            #    uselito = True
            #    lito = logdata[wellname]["lito"]['data']
            #else:
            uselito = False
            
            depth -= _DEPTHSHIFT
            labdepth -= _DEPTHSHIFT
            
            w = np.isfinite(dt)*np.isfinite(logrt)*np.isfinite(gr)*np.isfinite(cali)
            
            depthlim = np.max(depth[w]), np.min(depth[w])
            
            if smoothing["smooth"]:
                windowtype = smoothing["window"]
                if isinstance(windowtype, list):
                    windowtype = tuple(windowtype)
                windowdata = signal.windows.get_window(windowtype, smoothing["windowsize"], False)
                windowdata /= np.sum(windowdata)
                dt2 = np.convolve(dt, windowdata, 'same')
                logrt2 = np.convolve(logrt, windowdata, 'same')
            else:
                dt2 = dt
                logrt2 = logrt
            
            if resampling["resample"]:
                depth2 = np.linspace(depthlim[0], depthlim[1], resampling["npoints"])
                dt2 = np.interp(depth2, depth, dt2)
                logrt2 = np.interp(depth2, depth, logrt2)
            else:
                depth2 = depth
            
            dtbli = initialparameters["dtbaseline"]
            logrtbli = initialparameters["logrtbaseline"]
            lomi = initialparameters["lom"]
            
            toci = passeymethod(dt2, logrt2, dtbli, logrtbli, lomi)
            
            mainfigure = plt.figure(figsize=(16, 9))
            
            legendheight = 0.07
            depthwidth = 0.04
            
            left = 0.005 + depthwidth
            
            plt.subplots_adjust(left=left, right=0.995, bottom=0.005, top=0.995-2*legendheight, wspace=0.0)
            
            ###
            
            ax1 = plt.subplot(1, 5, 1)
            logplot(ax1, depth, gr, colorsdict[colors["gr"]], limits["gr"], depthlim)
            caliplot = (limits['gr'][1] - limits['gr'][0])*(cali - limits['cali'][0])/(limits['cali'][1] - limits['cali'][0]) + limits['gr'][0]
            logplot(ax1, depth, caliplot, colorsdict[colors["cali"]], limits["gr"], depthlim)
            
            ax1.set_xticks([30.0, 60.0, 90.0, 120.0])
            
            rect = getlegendrect(ax1, 0.0, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, self.logdata[wellname]["gr"]['displayname'], limits["gr"], colorsdict[colors["gr"]], '-', fontsize=_FONTSIZE)
            
            rect = getlegendrect(ax1, legendheight, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, self.logdata[wellname]["cali"]['displayname'], limits["cali"], colorsdict[colors["cali"]], '-', fontsize=_FONTSIZE)
            
            ###
            
            ax2 = plt.subplot(1, 5, 2)
            logplot(ax2, depth, dt, colorsdict[colors["dt"]], limits["dt"], depthlim)
            
            ax2.set_xticks([150.0, 100.0, 50.0])
            
            rect = getlegendrect(ax2, 0.0, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, self.logdata[wellname]["dt"]['displayname'], limits["dt"], colorsdict[colors["dt"]], '-', fontsize=_FONTSIZE)
            
            rect = getlegendrect(ax2, legendheight, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "Baseline", limits["dt"], colorsdict[colors["baseline"]], '-', linewidth=3.0, fontsize=_FONTSIZE)
            
            if smoothing["smooth"] and smoothing["show"]:
                logplot(ax2, depth2, dt2, colorsdict[colors["logrt"]], limits["dt"], depthlim)
                
                # rect = getlegendrect(ax2, legendheight, legendheight)
                # aux = plt.axes(rect)
                # loglegend(aux, logdata[wellname]["dt"]['displayname'] + " (Smooth)", limits["dt"], colorsdict[colors["logrt"]], '-', fontsize=_FONTSIZE)
            
            blpdt = BaselinePicker(ax2, color=colorsdict[colors["baseline"]], x0=dtbli, linewidth=3.0)
            blpdt.connect()
            
            ###
            
            ax3 = plt.subplot(1, 5, 3)
            logplot(ax3, depth, logrt, colorsdict[colors["logrt"]], limits["logrt"], depthlim)
            
            ax3.set_xticks([0.0, 1.0, 2.0])
            
            rect = getlegendrect(ax3, 0.0, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "log({})".format(self.logdata[wellname]["rt"]['displayname']), limits["logrt"], colorsdict[colors["logrt"]], '-', fontsize=_FONTSIZE)
            
            rect = getlegendrect(ax3, legendheight, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "Baseline", limits["logrt"], colorsdict[colors["baseline"]], '-', linewidth=3.0, fontsize=_FONTSIZE)
            
            if smoothing["smooth"] and smoothing["show"]:
                logplot(ax3, depth2, logrt2, colorsdict[colors["dt"]], limits["logrt"], depthlim)
                
                # rect = getlegendrect(ax3, legendheight, legendheight)
                # aux = plt.axes(rect)
                # loglegend(aux, "log({})".format(logdata[wellname]["rt"]['displayname']) + " (Smooth)", limits["logrt"], colorsdict[colors["dt"]], '-', fontsize=_FONTSIZE)
            
            blplogrt = BaselinePicker(ax3, color=colorsdict[colors["baseline"]], x0=logrtbli, linewidth=3.0)
            blplogrt.connect()
            
            ###
            
            ax4 = plt.subplot(1, 5, 4)
            dtline = logplot(ax4, depth, -(dt - dtbli)*0.02, colorsdict[colors["dt"]], limits["dtlogr"], depthlim)
            logrtline = logplot(ax4, depth, (logrt - logrtbli), colorsdict[colors["logrt"]], limits["dtlogr"], depthlim)
            
            ax4.set_xticks([0.0, 1.0, 2.0])
            
            rect = getlegendrect(ax4, 0.0, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "-0.02x({} - baseline)".format(self.logdata[wellname]["dt"]['name']), limits["dtlogr"], colorsdict[colors["dt"]], '-', fontsize=_FONTSIZE)
            
            rect = getlegendrect(ax4, legendheight, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "log({}) - baseline".format(self.logdata[wellname]["rt"]['name']), limits["dtlogr"], colorsdict[colors["logrt"]], '-', fontsize=_FONTSIZE)
            
            ###

            to_save = {
                'DEPT':np.flip(depth2),
                'TOC':np.flip(toci)
            }

            df = pd.DataFrame(to_save)
            df.to_csv('toc.csv', index=False)
            
            ax5 = plt.subplot(1, 5, 5)
            tocline = logplot(ax5, depth2, toci, colorsdict[colors["toc"]], limits["toc"], depthlim)
            logplot(ax5, labdepth, labtoc, colorsdict[colors["labtoc"]], limits["toc"], depthlim, style="o")
            
            ax5.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            
            rect = getlegendrect(ax5, 0.0, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "COT calculado (%)", limits["toc"], colorsdict[colors["toc"]], '-', fontsize=_FONTSIZE)
            
            rect = getlegendrect(ax5, legendheight, legendheight)
            aux = plt.axes(rect)
            loglegend(aux, "COT medido (%)", limits["toc"], colorsdict[colors["labtoc"]], 'o', fontsize=_FONTSIZE)
            
            dtll = LiveLine(dtline)
            logrtll = LiveLine(logrtline)
            tocll = LiveLine(tocline)

            print(limits)
            
            ###
            
            rect = getdepthrect(ax1, -depthwidth*int(uselito), depthwidth)
            depthax = plt.axes(rect)
            emptytrack(depthax, depthlim)
            
            rect = getlegendrect(depthax, 0.0, 2*legendheight)
            aux = plt.axes(rect)
            depthlegend(aux, self.logdata[wellname]["depth"]['displayname'], fontsize=_FONTSIZE)
            
            litoax = None
            
            if uselito:
                rect = getdepthrect(ax1, 0.0, depthwidth)
                litoax = plt.axes(rect)
                classificationplot(litoax, depth, self.lito, depthlim)
                
                rect = getlegendrect(litoax, 0.0, 2*legendheight)
                aux = plt.axes(rect)
                depthlegend(aux, self.logdata[wellname]["lito"]['displayname'], fontsize=_FONTSIZE)
            
            ###
            
            mainaxes = [ax1, ax2, ax3, ax4, ax5]
            
            if uselito:
                mainaxes.append(litoax)
            
            ###
            
            # ax = plt.axes([0.2, 0.95, 0.6, 0.025])
            lomfigure = plt.figure(figsize=(5, 1))
            ax = plt.axes([0.025, 0.5, 0.95, 0.475])
            
            plt.xlabel("LOM")
            plt.xticks(range(21))
            plt.yticks([])
            lompicker = LOMPicker(ax, color=colorsdict[colors["baseline"]], x0=lomi, linewidth=3.0)
            lompicker.connect()
            
            ###
            
            depthfigure = plt.figure(figsize=(3, 6))
            ax = plt.axes([0.5, 0.025, 0.475, 0.95])
            
            plt.ylabel(self.logdata[wellname]["depth"]['displayname'])
            # plt.yticks(range(21))
            plt.xticks([])
            dptcntr = DepthController(ax, depthlim, color=colorsdict[colors["baseline"]], linewidth=3.0)
            dptcntr.connect()
            
            ###
            
            def callbacklom(event):
                if event == 'start_moving':
                    tocll.start_life()
                elif event == 'moving':
                    dtbl = baselinedatatolog(depth2, *blpdt.getdata())
                    logrtbl = baselinedatatolog(depth2, *blplogrt.getdata())
                    lom = lompicker.getdata()
                    
                    tocdata = passeymethod(dt2, logrt2, dtbl, logrtbl, lom)
                    tocll.set_data(tocdata, depth2)
                else:
                    tocll.end_life()
            
            def callbackdt(event):
                if event == 'start_moving':
                    dtll.start_life()
                    callbacklom(event)
                elif event == 'moving':
                    dtbl = baselinedatatolog(depth, *blpdt.getdata())
                    dtll.set_data(-(dt - dtbl)*0.02, depth)
                    callbacklom(event)
                else:
                    dtll.end_life()
                    callbacklom(event)
            
            def callbacklogrt(event):
                if event == 'start_moving':
                    logrtll.start_life()
                    callbacklom(event)
                elif event == 'moving':
                    logrtbl = baselinedatatolog(depth, *blplogrt.getdata())
                    logrtll.set_data(logrt - logrtbl, depth)
                    callbacklom(event)
                else:
                    logrtll.end_life()
                    callbacklom(event)
            
            def callbackdepth(event):
                if event == 'end_moving':
                    ylim = dptcntr.getdata()
                    for ax in mainaxes:
                        ax.set_ylim(ylim)
                    
                    for i in reversed(range(len(depthax.texts))):
                        depthax.texts[i].remove()
                    
                    depthax.set_ylim(ylim)
                    
                    yticks = depthax.get_yticks()
                    for tick in yticks:
                        if tick < ylim[0] and tick > ylim[1]:
                            depthax.text(0.5, tick, "X{:g}".format(tick % 1000).replace('.', ','), ha='center', va='center', fontsize=_FONTSIZE)
                    
                    mainfigure.canvas.draw() 
            
            def callbackclose(event):
                plt.close('all')
            
            mainfigure.canvas.mpl_connect('close_event', callbackclose)
            
            blpdt.callback = callbackdt
            blplogrt.callback = callbacklogrt
            lompicker.callback = callbacklom
            dptcntr.callback = callbackdepth
            
            callbackdepth("end_moving")
            
            plt.show()