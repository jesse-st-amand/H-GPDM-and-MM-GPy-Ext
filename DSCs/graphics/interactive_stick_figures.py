import numpy as np
import matplotlib#;matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class InteractiveFigures():
    def __init__(self,list_of_stick_dicts):
        self.list_of_stick_dicts = list_of_stick_dicts
        self.num_figures = len(list_of_stick_dicts)

    def plot_animation_all_figures(self):
        x_dim = [0,0]
        y_dim = [0, 0]
        for i, stick_dicts in enumerate(self.list_of_stick_dicts):
            x_dim_temp = np.min(stick_dicts['pelvis']['CC'][:,0]) - 75
            y_dim_temp = np.min(stick_dicts['pelvis']['CC'][:, 1]) - 150
            if x_dim[0] > x_dim_temp:
                x_dim[0] = x_dim_temp
            if y_dim[0] > y_dim_temp:
                y_dim[0] = y_dim_temp

            x_dim_temp = np.max(stick_dicts['pelvis']['CC'][:, 0]) + 75
            y_dim_temp = np.max(stick_dicts['pelvis']['CC'][:, 1]) + 150
            if x_dim[1] < x_dim_temp:
                x_dim[1] = x_dim_temp
            if y_dim[1] < y_dim_temp:
                y_dim[1] = y_dim_temp

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(autoscale_on=False, xlim=(x_dim[0], x_dim[1]), ylim=(y_dim[0], y_dim[1]))
        ax.set_aspect('equal')
        ax.grid()
        colors = ['b','r','g','y','m','c']
        lws = [8,7.5,7,6.5,6,5.5]
        markersizes = [20,19,18,17,16,15]
        line = '-'
        num_figures = self.num_figures
        bodies = []
        larms = []
        rarms = []
        llegs = []
        rlegs = []
        heads = []
        for i,stick_dicts in enumerate(self.list_of_stick_dicts):
            lw = lws[i]
            color = colors[i]
            markersize = markersizes[i]
            bodies.append(ax.plot([], [], line, lw=lw, color=color))
            larms.append(ax.plot([], [], line, lw=lw, color=color))
            rarms.append(ax.plot([], [], line, lw=lw, color=color))
            llegs.append(ax.plot([], [], line, lw=lw, color=color))
            rlegs.append(ax.plot([], [], line, lw=lw, color=color))
            heads.append(ax.plot([], [], "o", lw=lw,markersize=markersize, color=color))

        def animate(t):
            for i,stick_dicts in enumerate(self.list_of_stick_dicts):

                bodyx = [stick_dicts["head"]["CC"][t, 0], stick_dicts["sternum"]["CC"][t, 0],
                         stick_dicts["torso"]["CC"][t, 0], stick_dicts["pelvis"]["CC"][t, 0]]
                bodyy = [stick_dicts["head"]["CC"][t, 1], stick_dicts["sternum"]["CC"][t, 1],
                         stick_dicts["torso"]["CC"][t, 1], stick_dicts["pelvis"]["CC"][t, 1]]
                bodies
                exec('body' + str(i) + '= bodies[' + str(i) + '][0]')
                exec('body' + str(i) + '.set_data(bodyx, bodyy)')

                larmx = [stick_dicts["sternum"]["CC"][t, 0], stick_dicts["left shoulder"]["CC"][t, 0],
                         stick_dicts["left hand"]["CC"][t, 0]]
                larmy = [stick_dicts["sternum"]["CC"][t, 1], stick_dicts["left shoulder"]["CC"][t, 1],
                         stick_dicts["left hand"]["CC"][t, 1]]
                larms
                exec('larm' + str(i) + '= larms[' + str(i) + '][0]')
                exec('larm' + str(i) + '.set_data(larmx, larmy)')

                rarmx = [stick_dicts["sternum"]["CC"][t, 0], stick_dicts["right shoulder"]["CC"][t, 0],
                         stick_dicts["right hand"]["CC"][t, 0]]
                rarmy = [stick_dicts["sternum"]["CC"][t, 1], stick_dicts["right shoulder"]["CC"][t, 1],
                         stick_dicts["right hand"]["CC"][t, 1]]
                rarms
                exec('rarm' + str(i) + '= rarms[' + str(i) + '][0]')
                exec('rarm' + str(i) + '.set_data(rarmx, rarmy)')

                llegx = [stick_dicts["pelvis"]["CC"][t, 0], stick_dicts["left knee"]["CC"][t, 0],
                         stick_dicts["left foot"]["CC"][t, 0]]
                llegy = [stick_dicts["pelvis"]["CC"][t, 1], stick_dicts["left knee"]["CC"][t, 1],
                         stick_dicts["left foot"]["CC"][t, 1]]
                llegs
                exec('lleg' + str(i) + '= llegs[' + str(i) + '][0]')
                exec('lleg' + str(i) + '.set_data(llegx, llegy)')

                rlegx = [stick_dicts["pelvis"]["CC"][t, 0], stick_dicts["right knee"]["CC"][t, 0],
                         stick_dicts["right foot"]["CC"][t, 0]]
                rlegy = [stick_dicts["pelvis"]["CC"][t, 1], stick_dicts["right knee"]["CC"][t, 1],
                         stick_dicts["right foot"]["CC"][t, 1]]
                rlegs
                exec('rleg' + str(i) + '= rlegs[' + str(i) + '][0]')
                exec('rleg' + str(i) + '.set_data(rlegx, rlegy)')

                headx = [stick_dicts["head"]["CC"][t, 0]]
                heady = [stick_dicts["head"]["CC"][t, 1]]
                heads
                exec('head' + str(i) + '= heads[' + str(i) + '][0]')
                exec('head' + str(i) + '.set_data(headx, heady)')

            return exec(','.join(['body'+str(i)+', larm'+str(i)+', rarm'+str(i)+', lleg'+str(i)+', rleg'+str(i)+', head'+str(i) for i in range(num_figures)]))

        return fig, animate
