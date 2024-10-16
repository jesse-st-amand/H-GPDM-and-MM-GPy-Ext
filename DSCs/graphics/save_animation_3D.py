import matplotlib;matplotlib.use("TkAgg")
import imageio
from HGPLVM.graphics.stick_graphic_3D import StickGraphic
from HGPLVM.graphics.interactive_stick_figures_3D import InteractiveFigures
from matplotlib import pyplot as plt
from glob import glob
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
##
num_tps = 100
interval = 50
subject_selection = [0]
action_selection = [0]#np.arange(0,12) [0,1,2,3,4,5,6,7,8,9,10,11]
save_npy = False
play_ani = True
names = ['Lucas','Jana','Jens','Alex','Lisa']
people = []
action_types = []
sub_interval = []
timing_files = []
##
people.append('D:\\bimanual_experiment_bvh\\Lucas\\')#0
people.append('D:\\bimanual_experiment_bvh\\Jana\\')#1
people.append('D:\\bimanual_experiment_bvh\\Jens\\')#2
people.append('D:\\bimanual_experiment_bvh\\Alex\\')#3
people.append('D:\\bimanual_experiment_bvh\\Lisa\\')#4


action_types.append('box_half_lift')#0
sub_interval.append([''])
action_types.append('box_half_lift_2kg')#1
sub_interval.append([''])
action_types.append('box_lateral')#2  _left _right _general
sub_interval.append(['_left', '_right', '_general'])
action_types.append('box_lateral_2kg')#3  _left _right _general
sub_interval.append(['_left', '_right', '_general'])
action_types.append('box_lift')#4
sub_interval.append([''])
action_types.append('box_lift_2kg')#5
sub_interval.append([''])
action_types.append('box_turn_cw')#6 _back _forward _general
sub_interval.append(['_back', '_forward', '_general'])
action_types.append('box_turn_ccw')#7 _back _forward _general
sub_interval.append(['_back', '_forward', '_general'])
action_types.append('bread_cutting_left')#8
sub_interval.append([''])
action_types.append('bread_cutting_right')#9
sub_interval.append([''])
action_types.append('jar_opening_left')#10 _close _open _general
sub_interval.append(['_close', '_open', '_general'])
action_types.append('jar_opening_right')#11 _close _open _general
sub_interval.append(['_close', '_open', '_general'])


sequences = []
stick_mats = []
SM_PCs = []
for action_i in action_selection:
    for person_i in subject_selection:
        for sub_i in sub_interval[action_i]:
            graphic1 = StickGraphic()
            action = action_types[action_i]
            person = people[person_i]
            file = person + 'BVH\\' + action + '_pos.csv'
            timing_file = person + 'Timing\\' + action + '_trial_times' + sub_i + '.csv'
            print(timing_file)
            timing_arr = pd.read_csv(timing_file).to_numpy()
            df = pd.read_csv(file)
            df = df.drop('time', axis=1)
            pos_arr = df.to_numpy()


            #line = np.linspace(0, numpy_array.shape[0] - 1, num_tps, dtype=int)

            sequences.append(pos_arr)

            N, D3 = df.shape
            N_D_3 = np.zeros([N, int((D3) / 3), 3])
            axis_num = {'x': int(0), 'y': int(2), 'z': int(1)}
            for i, header in enumerate(df.columns):
                name, axis = header.split('.')
                N_D_3[:, int(np.trunc(i / 3)), axis_num[axis]] = df[header].to_numpy()
                D_N_3 = np.moveaxis(N_D_3,0,1) #D_N_3

            D_N_3 -= np.mean(D_N_3[0,:,:],0)#np.mean(np.mean(stick_mat,0),0)

            for i,start_stop in enumerate(timing_arr):
                start = start_stop[0]
                stop = start_stop[1]
                line = np.linspace(start,stop, num_tps, dtype=int)
                stick_mat = D_N_3[:,line,:]
                stick_mats.append(stick_mat)
                SM_angles = graphic1.HGPLVM_stick_mat_CCs_to_angles_PV(stick_mat)
                #SM_angles[:,65:]= np.zeros(SM_angles[:,65:].shape)+np.pi/4
                SM_PCs.append(SM_angles)

                if save_npy == True:
                    np.save('.\\npy\\3D\\' + action + sub_i + '\\stick_mat_CC_' + names[person_i] + '_trial_' + str(i) + '_tps_' + str(num_tps), stick_mat)
                    np.save('.\\npy\\3D\\' + action + sub_i + '\\PCs_' + names[person_i] + '_trial_' + str(i) + '_tps_' + str(num_tps), SM_angles)
d_dict = {}
for i, header in enumerate(df.columns):
    d_dict[header] = i


def play_animation(stick_mats,num_tps=100):
    graphic1 = StickGraphic()
    stick_dicts = []
    x_dim = [np.min(stick_mats[0][:, :, 0]), np.max(stick_mats[0][:, :, 0])]
    y_dim = [np.min(stick_mats[0][:, :, 1]), np.max(stick_mats[0][:, :, 1])]
    z_dim = [np.min(stick_mats[0][:, :, 2]), np.max(stick_mats[0][:, :, 2])]
    for i,stick_mat in enumerate(stick_mats):
        #stick_dicts.append(graphic1.stick_mat_to_dict(stick_mat))
        #SM_angles = graphic1.HGPLVM_stick_mat_CCs_to_angles_PV(stick_mat)
        stick_dicts.append(graphic1.HGPLVM_angles_PV_to_stick_dicts_CCs(SM_PCs[i]))

        '''stick_mat2 = graphic1.HGPLVM_angles_PV_to_stick_mat_CCs(SM_angles)
        SM_angles2 = graphic1.HGPLVM_stick_mat_CCs_to_angles_PV(stick_mat2)
        stick_dicts.append(graphic1.HGPLVM_angles_PV_to_stick_dicts_CCs(SM_angles2))'''

        if x_dim[0] > np.min(stick_mat[:, :, 0]):
            x_dim[0] = np.min(stick_mat[:, :, 0])
        if x_dim[1] < np.max(stick_mat[:, :, 0]):
            x_dim[1] = np.max(stick_mat[:, :, 0])

        if y_dim[0] > np.min(stick_mat[:, :, 1]):
            y_dim[0] = np.min(stick_mat[:, :, 1])
        if y_dim[1] < np.max(stick_mat[:, :, 1]):
            y_dim[1] = np.max(stick_mat[:, :, 1])

        if z_dim[0] > np.min(stick_mat[:, :, 2]):
            z_dim[0] = np.min(stick_mat[:, :, 2])
        if z_dim[1] < np.max(stick_mat[:, :, 2]):
            z_dim[1] = np.max(stick_mat[:, :, 2])

    IF = InteractiveFigures(stick_dicts)
    return IF.plot_animation_all_figures()

if play_ani:
    fig, animate = play_animation(stick_mats,num_tps=num_tps)
    ani = animation.FuncAnimation(fig, animate, num_tps, interval=interval)
    plt.show()