import matplotlib;matplotlib.use("TkAgg")
import imageio
from HGPLVM.graphics.stick_graphic import StickGraphic
from HGPLVM.graphics.interactive_stick_figures import InteractiveFigures
from matplotlib import pyplot as plt
from glob import glob
import matplotlib.animation as animation
import numpy as np
import os
##
num_tps = 200
interval = 40
action_selection = [0,1,2]
num_seqs_per_set = 5

subject_selection = [0]
action_selection = [0]
actions = ['walking_r']#'cartwheels_r','jumping_jacks','walking_r'
subjects = ['subject_joint']
stick_dir = 'D:\\2D_stick_figures\\'
locations = []
##
locations.append('D:\\2D_stick_figures\\cartwheels_r\\')#0
locations.append('D:\\2D_stick_figures\\jumping_jacks\\')#1
locations.append('D:\\2D_stick_figures\\walking_r\\')#2

'''actions = []
for i in action_selection:
    seqs = []
    for subdir_path in glob(locations[i]+'*'):
        subdir = subdir_path.split('\\')[-1]
        if subdir == 'creation':
            continue
        if not len(seqs) < num_seqs_per_set:
            break
        seqs.append(subdir_path)
    actions.append(seqs)'''

def save_anim_numpy(actions,num_tps):
    graphic1 = StickGraphic()
    for action in actions:
        for subject in subjects:
            location = stick_dir+action+'\\'+subject+'\\'
            for i,trial in enumerate(glob(location+'*')):
                if trial.split('\\')[-1].split('_')[-1] == 'gs':
                    graphic1.set_color_key('gs')
                else:
                    graphic1.set_color_key('rgb')
                print(trial)
                stick_mat_angles = graphic1.location_to_angles_PV(trial, num_time_points=num_tps)

                np.save('.\\npy\\2D\\' + action + '\\PCs_' + subject + '_trial_' + str(i) + '_tps_' + str(num_tps),stick_mat_angles)
                print('.\\npy\\2D\\' + action + '\\PCs_' + subject + '_trial_' + str(i) + '_tps_' + str(num_tps))

def play_animation(actions,num_tps=100):
    graphic1 = StickGraphic()
    stick_dicts = []
    x_dim = 0
    y_dim = 0
    graphic1 = StickGraphic()
    for action in actions:
        for subject in subjects:
            location = 'C:\\Users\\Jesse\\Documents\\Python\\GPy\\HGPLVM\\graphics\\npy\\2D\\' + action + '\\'
            for i, trial in enumerate(glob(location +'PCs_'+ subject +'*')):
                print(trial)
                stick_mat = np.load(trial)
                stick_mat_CC = graphic1.SAPV_SAPCC(stick_mat)
                stick_dicts.append(graphic1.HGPLVM_angles_PV_to_stick_dicts_CCs(stick_mat))

                if x_dim < np.max(stick_mat_CC[:, 0]):
                    x_dim = np.max(stick_mat_CC[:, 0])
                #if x_dim[1] < np.max(stick_mat[:, :, 0]):
                    #x_dim[1] = np.max(stick_mat[:, :, 0])
                c = 150
                if y_dim < np.max(stick_mat_CC[:, 1]) + c:
                    y_dim = np.max(stick_mat_CC[:, 1]) +c
                #if y_dim[1] < np.max(stick_mat[:, :, 1]):
                    #y_dim[1] = np.max(stick_mat[:, :, 1])

    IF = InteractiveFigures(stick_dicts)
    return IF.plot_animation_all_figures()


save_anim_numpy(actions,num_tps)
fig, animate = play_animation(actions,num_tps=num_tps)
ani = animation.FuncAnimation(fig, animate, num_tps, interval=interval)
plt.show()