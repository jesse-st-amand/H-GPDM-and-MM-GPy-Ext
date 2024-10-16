import matplotlib;matplotlib.use("TkAgg")
import imageio
from HGPLVM.graphics.stick_graphic_3D import StickGraphicCMU
from HGPLVM.graphics.interactive_stick_figures_3D import InteractiveFiguresCMU
from matplotlib import pyplot as plt
from glob import glob
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
##
num_tps = 100
interval = 50
subject_selection = [0,1,2,3,4,5,6,7,8,9]
action_selection = [7]#np.arange(0,12) [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]0,1,7,9,10
save_npy = False
play_ani = True
names = ['02','10','13','14','113','118','126','136','143','144']
people = []
action_types = []
sub_interval = ['01','02','03','04','05','06','07','08','09','10']
timing_files = []
##
people.append('D:\\CMU_bvh\\02\\')#0
people.append('D:\\CMU_bvh\\10\\')#1
people.append('D:\\CMU_bvh\\13\\')#2
people.append('D:\\CMU_bvh\\14\\')#3
people.append('D:\\CMU_bvh\\113\\')#4
people.append('D:\\CMU_bvh\\118\\')#5
people.append('D:\\CMU_bvh\\126\\')#6
people.append('D:\\CMU_bvh\\136\\')#7
people.append('D:\\CMU_bvh\\143\\')#8
people.append('D:\\CMU_bvh\\144\\')#9


action_types.append('bend_down')#0
action_types.append('soccer_kick')#1
action_types.append('elbow_to_knee')#2
action_types.append('jumping_jacks')#3
action_types.append('squats')#4
action_types.append('twists')#5
action_types.append('run_single_cycle')#6
action_types.append('jump')#7
action_types.append('backstroke')#8
action_types.append('breaststroke')#9
action_types.append('flystroke')#10
action_types.append('normal_walk')#11
action_types.append('jump_distance')#12
action_types.append('jump_side')#13
action_types.append('jump_up')#14
action_types.append('sit_get_up')#15
action_types.append('cartwheels')#16
action_types.append('left_front_kick')#17
action_types.append('left_lunges')#18
action_types.append('left_punches')#19





sequences = []
stick_mats = []
SM_PCs = []
headers = False
for action_i in action_selection:
    for person_i in subject_selection:
        for sub_i in sub_interval:
            graphic1 = StickGraphicCMU()
            action = action_types[action_i]
            file = 'D:\\CMU_bvh\\' + names[person_i] + '\\BVH\\' + action + '_' + sub_i + '_pos.csv'
            timing_file = 'D:\\CMU_bvh\\' + names[person_i] + '\\Timing\\' + action + '_' + sub_i + '.csv'

            if os.path.exists(timing_file):
                print(timing_file)
            else:
                continue

            timing_arr = pd.read_csv(timing_file).to_numpy()
            df = pd.read_csv(file)
            df = df.drop('time', axis=1)
            pos_arr = df.to_numpy()

            if headers == False:
                d_dict = {}
                for i, header in enumerate(df.columns):
                    d_dict[header] = i
                headers = True

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
                    folder_path = '.\\npy\\CMU\\' + action

                    # Check if the folder exists
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)  # Create the folder
                        print(f"Folder '{folder_path}' created.")
                    print(folder_path + '\\stick_mat_CC_01_trial_' + str(sub_i) + '_tps_' + str(num_tps))
                    np.save(folder_path + '\\stick_mat_CC_01_trial_' + str(sub_i) + '_tps_' + str(num_tps), stick_mat)
                    np.save(folder_path + '\\PCs_01_trial_' + str(sub_i) + '_tps_' + str(num_tps), SM_angles)



def play_animation(stick_mats,num_tps=100):
    graphic1 = StickGraphicCMU()
    stick_dicts = []
    x_dim = [np.min(stick_mats[0][:, :, 0]), np.max(stick_mats[0][:, :, 0])]
    y_dim = [np.min(stick_mats[0][:, :, 1]), np.max(stick_mats[0][:, :, 1])]
    z_dim = [np.min(stick_mats[0][:, :, 2]), np.max(stick_mats[0][:, :, 2])]
    for i,stick_mat in enumerate(stick_mats):
        stick_dicts.append(graphic1.HGPLVM_angles_PV_to_stick_dicts_CCs(SM_PCs[i]))


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

    IF = InteractiveFiguresCMU(stick_dicts)
    return IF.plot_animation_all_figures()

if play_ani:
    fig, animate = play_animation(stick_mats,num_tps=num_tps)
    ani = animation.FuncAnimation(fig, animate, num_tps, interval=interval)
    plt.show()