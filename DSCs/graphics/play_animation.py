import matplotlib;matplotlib.use("TkAgg")
import imageio
from HGPLVM.graphics.stick_graphic import StickGraphic
from HGPLVM.graphics.interactive_stick_figures import InteractiveFigures
from matplotlib import pyplot as plt
import glob
import matplotlib.animation as animation
import numpy as np

##
num_tps = 100
location = 'C:\\Users\\Jesse\\Documents\\Pivot Animator\\Animations\\images\\backspring_l\\bsl1'
interval = 15

def play_animation(location,num_tps=100):
    graphic1 = StickGraphic('gs')
    stick_mat_angles1 = graphic1.location_to_angles_PV(location,num_time_points=num_tps)
    graphic1.HGPLVM_angles_PV_to_stick_dicts_CCs(stick_mat_angles1)

    ##
    fileset = [file for file in glob.glob(location + "**/*.png", recursive=True)]
    im = imageio.imread(fileset[0])
    rows,cols,depth = im.shape
    x_dim = cols / 2
    y_dim = rows / 2
    IF = InteractiveFigures([graphic1.stick_dicts])
    return IF.plot_animation_all_figures(x_dim, y_dim)


fig, animate = play_animation(location,num_tps=100)
ani = animation.FuncAnimation(fig, animate, num_tps, interval=interval)
plt.show()