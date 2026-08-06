
class BaselinePicker(object):
    
    def __init__(self, ax, xlim=None, ylim=None, x0=None, color="C0", linestyle="-", linewidth=1, pickradius=5.0, callback=None):
        self.ax = ax
        self.background = None
        self.pickedline = None
        self.button = None
        self.vlines = []
        self.hline = None
        
        self.callback = callback
        
        self.xlim = xlim
        if self.xlim is None:
            self.xlim = list(sorted(ax.get_xlim()))
        
        self.ylim = ylim
        if self.ylim is None:
            self.ylim = list(sorted(ax.get_ylim()))
        
        self.color = color
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.pickradius = pickradius
        
        if x0 is None:
            x0 = sum(self.xlim)/2.0
        
        initialline = self.createvline(x0, min(self.ylim), max(self.ylim))
        
        self.vlines.append(initialline)

    def createvline(self, x, ymin, ymax):
        return self.ax.plot([x, x], [ymin, ymax], color=self.color, linestyle=self.linestyle, linewidth=self.linewidth, pickradius=self.pickradius, zorder=-1000)[0]
    
    def createhline(self, y):
        return self.ax.plot([min(self.xlim), max(self.xlim)], [y, y], color=self.color, linestyle=self.linestyle, linewidth=self.linewidth, pickradius=self.pickradius)[0]
    
    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        pickedline = None
        
        for line in self.vlines:
            contains, attrd = line.contains(event)
            if contains:
                pickedline = line
                break
        
        if pickedline is None:
            return
        
        self.pickedline = pickedline
        
        if event.button == 1:
            self.pickedline.set_animated(True)
            self.ax.figure.canvas.draw()
            self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

            self.ax.draw_artist(self.pickedline)

            self.ax.figure.canvas.blit(self.ax.bbox)
            
            self.button = 1
            
            if self.callback:
                self.callback("start_moving")
        
        elif event.button == 3:
            self.hline = self.createhline(event.ydata)
            
            self.hline.set_animated(True)
            self.ax.figure.canvas.draw()
            self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)

            self.ax.draw_artist(self.hline)

            self.ax.figure.canvas.blit(self.ax.bbox)
            
            self.button = 3
            

    def on_motion(self, event):
        if event.inaxes != self.ax:
            return
                
        if self.button == 1:
            self.pickedline.set_xdata([event.xdata, event.xdata])

            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.pickedline)
            self.ax.figure.canvas.blit(self.ax.bbox)
            
            if self.callback:
                self.callback("moving")
        
        elif self.button == 3:
            self.hline.set_ydata([event.ydata, event.ydata])
            
            self.ax.figure.canvas.restore_region(self.background)
            self.ax.draw_artist(self.hline)
            self.ax.figure.canvas.blit(self.ax.bbox)

    def on_release(self, event):
        if self.button == 1:
            self.pickedline.set_animated(False)
            self.pickedline = None
            self.button = None
            self.background = None
            self.ax.figure.canvas.draw()
            
            if self.callback:
                self.callback("end_moving")
        
        elif self.button == 3:
            self.hline.set_animated(False)
            self.ax.lines.remove(self.hline)
            del self.hline
            self.hline = None
            self.button = None
            self.background = None
            
            y = event.ydata
            
            ymin, ymax = sorted(self.pickedline.get_ydata())
            x = self.pickedline.get_xdata()[0]
            
            if y <= ymin or y >= ymax:
                return
            
            self.pickedline.set_ydata([ymin, y])
            
            newline = self.createvline(x, y, ymax)
            self.vlines.append(newline)
            
            self.ax.figure.canvas.draw()

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
    
    def getdata(self):
        x = []
        y = []
        for line in self.vlines:
            x.append(line.get_xdata()[0])
            y.append(list(sorted(line.get_ydata())))
        
        return x, y


class LiveLine(object):
    
    def __init__(self, line):
        self.line = line
        self.background = None

    def start_life(self):
        self.line.set_animated(True)
        self.line.figure.canvas.draw()
        self.background = self.line.figure.canvas.copy_from_bbox(self.line.axes.bbox)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def set_data(self, xdata, ydata):
        self.line.set_xdata(xdata)
        self.line.set_ydata(ydata)
        self.line.figure.canvas.restore_region(self.background)
        self.line.axes.draw_artist(self.line)
        self.line.figure.canvas.blit(self.line.axes.bbox)

    def end_life(self):
        self.background = None
        self.line.set_animated(False)
        self.line.figure.canvas.draw()


class LOMPicker(object):
    
    def __init__(self, ax, xlim=(0, 20), x0=10, color="C0", linestyle="-", linewidth=1, pickradius=5.0, callback=None):
        self.ax = ax
        self.background = None
        self.moving = False
        self.callback = callback
        
        self.ax.set_xlim(xlim)
        self.ax.set_ylim([0.0, 1.0])
        
        self.lomline, = self.ax.plot([x0, x0], [0.0, 1.0], color=color, linestyle=linestyle, linewidth=linewidth, pickradius=pickradius)
    
    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button != 1:
            return
        
        contains, attrd = self.lomline.contains(event)
        if not contains:
            return
        
        self.lomline.set_animated(True)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.lomline)
        self.ax.figure.canvas.blit(self.ax.bbox)
        
        self.moving = True
        
        if self.callback:
            self.callback("start_moving")

    def on_motion(self, event):
        if not self.moving:
            return
        
        if event.inaxes != self.ax:
            return
        
        self.lomline.set_xdata([event.xdata, event.xdata])

        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.lomline)
        self.ax.figure.canvas.blit(self.ax.bbox)
        
        if self.callback:
            self.callback("moving")

    def on_release(self, event):
        self.lomline.set_animated(False)
        self.moving = False
        self.background = None
        self.ax.figure.canvas.draw()
        
        if self.callback:
            self.callback("end_moving")

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
    
    def getdata(self):
        return self.lomline.get_xdata()[0]


class DepthController(object):
    def __init__(self, ax, ylim, color="C0", linestyle="-", linewidth=1, pickradius=5.0, callback=None):
        self.ax = ax
        self.background = None
        self.pickedline = None
        self.callback = callback
        self.ylim = ylim
        
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim([0.0, 1.0])
        
        self.topline, = self.ax.plot([0.0, 1.0], [self.ylim[1], self.ylim[1]], color=color, linestyle=linestyle, linewidth=linewidth, pickradius=pickradius)
        self.bottomline, = self.ax.plot([0.0, 1.0], [self.ylim[0], self.ylim[0]], color=color, linestyle=linestyle, linewidth=linewidth, pickradius=pickradius)
    
    def connect(self):
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.button != 1:
            return
        
        contains, attrd = self.topline.contains(event)
        if contains:
            self.pickedline = self.topline
        else:
            contains, attrd = self.bottomline.contains(event)
            if contains:
                self.pickedline = self.bottomline
            else:
                return
        
        self.pickedline.set_animated(True)
        self.ax.figure.canvas.draw()
        self.background = self.ax.figure.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.pickedline)
        self.ax.figure.canvas.blit(self.ax.bbox)
        
        if self.callback:
            self.callback("start_moving")

    def on_motion(self, event):
        if self.pickedline is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        self.pickedline.set_ydata([event.ydata, event.ydata])

        self.ax.figure.canvas.restore_region(self.background)
        self.ax.draw_artist(self.pickedline)
        self.ax.figure.canvas.blit(self.ax.bbox)
        
        if self.callback:
            self.callback("moving")

    def on_release(self, event):
        if self.pickedline is None:
            return
        
        if event.inaxes != self.ax:
            return
        
        self.pickedline.set_animated(False)
        self.pickedline = None
        self.background = None
        self.ax.figure.canvas.draw()
        
        if self.callback:
            self.callback("end_moving")

    def disconnect(self):
        self.rect.figure.canvas.mpl_disconnect(self.cidpress)
        self.rect.figure.canvas.mpl_disconnect(self.cidrelease)
        self.rect.figure.canvas.mpl_disconnect(self.cidmotion)
    
    def getdata(self):
        return self.bottomline.get_ydata()[0], self.topline.get_ydata()[0]
        