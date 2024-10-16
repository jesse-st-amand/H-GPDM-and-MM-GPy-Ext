from ..graphics.stick_figure import StickFigure
import imageio
import glob
import numpy as np
import matplotlib#;matplotlib.use("TkAgg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
class StickGraphic():
    def __init__(self, img_color_scale='rgb'):
        self.file_set = None
        self.node_color_list = []
        self.set_color_key(img_color_scale)
        self.num_nodes = len(self.node_color_list)
        self.num_images = None
        self.stick_dicts = None
        self.pelvis_init = None
        self.x_dim = None
        self.y_dim = None

    def images_to_stick_mat_CC(self,image_directory):
        self.file_set = [file for file in glob.glob(image_directory + "**/*.png", recursive=True)]
        self.num_images = len(self.file_set)
        im = imageio.imread(self.file_set[0])
        rows, cols, depth = im.shape
        self.x_dim = cols / 2
        self.y_dim = rows / 2

        xs_mat = np.zeros([self.num_nodes, self.num_images])
        ys_mat = np.zeros([self.num_nodes, self.num_images])

        for i, file in enumerate(self.file_set):
            im = imageio.imread(file)
            xs = np.zeros(12)
            ys = np.zeros(12)
            for j, color in enumerate(self.node_color_list):
                c0, c1, c2 = color

                tally = np.zeros(im[:, :, 0].shape)
                tally[np.where(im[:, :, 0] == c0)] += 1
                tally[np.where(im[:, :, 1] == c1)] += 1
                tally[np.where(im[:, :, 2] == c2)] += 1

                indices = np.where(tally == 3)
                xs[j] = np.mean(indices[0])
                ys[j] = np.mean(indices[1])

            # x and y swapped, x flipped
            ys_new = -1 * xs
            xs_new = ys

            xs_mat[:, i] = xs_new
            ys_mat[:, i] = ys_new

        xs_mat = xs_mat - np.mean(np.mean(xs_mat))
        ys_mat = ys_mat - np.mean(np.mean(ys_mat))

        stick_mat = np.concatenate(
            [xs_mat.reshape([self.num_nodes, self.num_images, 1]), ys_mat.reshape([self.num_nodes, self.num_images, 1])],
            2)
        self.pelvis_init = stick_mat[0,0,0:2]
        return stick_mat

    def images_to_stick_dicts_CC(self,image_directory):
        stick_mat = self.images_to_stick_mat_CC(image_directory)
        self.stick_dicts = self.stick_mat_to_dict(stick_mat)

    def get_2D_CC_mat(self):
        num_coors = self.stick_dicts["pelvis"]['CC'].shape[1]
        CC_mat_2D = np.zeros([self.num_nodes*num_coors,self.num_images])
        for i,key in enumerate(self.stick_dicts.keys()):
            CC_mat_2D[(((i+1)*num_coors)-2):(i+1)*num_coors,:] = self.stick_dicts[key]['CC'].T
        return CC_mat_2D

    def get_3D_CC_mat_from_2D(self,CC_mat_2D):
        num_coors = int(CC_mat_2D.shape[0]/self.num_nodes)
        T = CC_mat_2D.shape[1]
        stick_mat = np.zeros([self.num_nodes,T,num_coors])
        for i in range(self.num_nodes):
            stick_mat[i,:,:] = CC_mat_2D[(((i+1)*num_coors)-2):(i+1)*num_coors,:].T
        return stick_mat

    def get_3D_CC_mat(self):
        num_coors = list(self.stick_dicts["pelvis"]['CC'].values).shape[1]
        CC_mat_3D = np.zeros([self.num_nodes,self.num_images,num_coors])
        for i,key in enumerate(self.stick_dicts.keys()):
            CC_mat_3D[i,:,:] = self.stick_dicts[key]['CC'].T
        return CC_mat_3D

    def interp(self,T):
        """
        T: the number of timesteps to be interpolated
        :param T:
        :return:
        """
        stick_mat_CC_2D = self.get_2D_CC_mat()
        start_pt_pv = stick_mat_CC_2D[:2, 0].copy()
        N = stick_mat_CC_2D.shape[1]
        d = T/N
        #stick_mat_CC_2D[:2,:]=self.PV_CC_position2velocity(stick_mat_CC_2D[:2, :])
        x = np.linspace(0, self.num_images - 1, num=self.num_images, endpoint=True)
        x2 = np.linspace(0, self.num_images - 1, num=T, endpoint=True)
        f = interp1d(x, stick_mat_CC_2D, kind='quadratic')
        stick_mat_CC_interp_2D = f(x2)
        #stick_mat_CC_interp_2D[:2, :] = self.PV_CC_velocity2position(stick_mat_CC_interp_2D[:2, :]/d,start_pt_pv)
        stick_mat_CC_interp_3D = self.get_3D_CC_mat_from_2D(stick_mat_CC_interp_2D)
        return stick_mat_CC_interp_3D

    def PV_CC_position2velocity(self,PV_array):
        temp = np.zeros(PV_array.shape)
        temp[:, 1:] = PV_array[:, 1:] - PV_array[:, :-1]
        temp[:, 0] = temp[:, 1]
        return temp

    def PV_CC_velocity2position(self,PV_array,start_pt_pv):
        temp = np.zeros(PV_array.shape)
        temp[:,0]=start_pt_pv
        j = 1
        for i in range(PV_array.shape[1]-1):
            temp[:, j] = PV_array[:,j] + temp[:,i]
            j += 1
        return temp

    def stick_mat_to_dict(self,stick_mat,CCs=True):
        if CCs == True:
            stick_dict = {
                "pelvis": {"CC": stick_mat[0, :, :].squeeze()},
                "left knee": {"CC": stick_mat[1, :, :].squeeze()},
                "right knee": {"CC": stick_mat[2, :, :].squeeze()},
                "torso": {"CC": stick_mat[3, :, :].squeeze()},
                "left foot": {"CC": stick_mat[4, :, :].squeeze()},
                "right foot": {"CC": stick_mat[5, :, :].squeeze()},
                "sternum": {"CC": stick_mat[6, :, :].squeeze()},
                "left shoulder": {"CC": stick_mat[7, :, :].squeeze()},
                "head": {"CC": stick_mat[8, :, :].squeeze()},
                "right shoulder": {"CC": stick_mat[9, :, :].squeeze()},
                "left hand": {"CC": stick_mat[10, :, :].squeeze()},
                "right hand": {"CC": stick_mat[11, :, :].squeeze()}
            }
        else: #input in angles
            stick_dict = {
                "pelvis": {"angle": None, "CC": np.concatenate([stick_mat[0, :, :].squeeze(1),stick_mat[1, :, :].squeeze(1)],0)},
                "left knee": {"angle": stick_mat[2, :, :].squeeze()},
                "right knee": {"angle": stick_mat[3, :, :].squeeze()},
                "torso": {"angle": stick_mat[4, :, :].squeeze()},
                "left foot": {"angle": stick_mat[5, :, :].squeeze()},
                "right foot": {"angle": stick_mat[6, :, :].squeeze()},
                "sternum": {"angle": stick_mat[7, :, :].squeeze()},
                "left shoulder": {"angle": stick_mat[8, :, :].squeeze()},
                "head": {"angle": stick_mat[9, :, :].squeeze()},
                "right shoulder": {"angle": stick_mat[10, :, :].squeeze()},
                "left hand": {"angle": stick_mat[11, :, :].squeeze()},
                "right hand": {"angle": stick_mat[12, :, :].squeeze()}
            }
        return stick_dict
    def stick_mat_CC_to_angles(self,stick_mat_CCs):
        self.num_nodes, self.num_images, DoFs = stick_mat_CCs.shape
        stick_mat_angles = np.array([], dtype=np.int64)
        for i in range(self.num_images):
            stick_dict = self.stick_mat_to_dict(stick_mat_CCs[:,i,:].reshape([self.num_nodes, 1, DoFs]))
            SF = StickFigure(stick_dict)
            if i == 0:
                stick_mat_angles = SF.angles.reshape([self.num_nodes+1, 1, DoFs-1])
            else:
                stick_mat_angles = np.concatenate([stick_mat_angles, SF.angles.reshape([self.num_nodes+1, 1, DoFs-1])],1)
        return stick_mat_angles

    def stick_mat_angles_to_CC(self,stick_mat_angles):
        rows,self.num_images,DoFs = stick_mat_angles.shape
        self.num_nodes = rows-1
        for i in range(self.num_images):
            stick_dict = self.stick_mat_to_dict(stick_mat_angles[:,i,:].reshape([self.num_nodes+1, 1, DoFs]), CCs=False)
            SF = StickFigure(stick_dict)
            if i == 0:
                stick_mat_CCs = SF.CCs.reshape([self.num_nodes,1,DoFs+1])
            else:
                stick_mat_CCs = np.concatenate([stick_mat_CCs,SF.CCs.reshape([self.num_nodes,1,DoFs+1])],1)
        return stick_mat_CCs

    def SAPCC_to_SAPV(self, stick_mat_angles):
        '''
        stick angles w/ pelvis in CC to stick angles w/ pelvis in velocity
        First time point is 0
        '''
        N, D = stick_mat_angles.shape
        #stick_mat_angles_PV = stick_mat_angles
        stick_mat_angles_PV = np.zeros([N, D])
        stick_mat_angles_PV[:, 2:] = stick_mat_angles[:, 2:]
        stick_mat_angles_PV[1:, :2] = stick_mat_angles[1:N, 0:2] - stick_mat_angles[0:(N - 1), 0:2]
        stick_mat_angles_PV[0, :2] = stick_mat_angles_PV[1, :2]
        return stick_mat_angles_PV

    def SAPV_SAPCC(self, stick_mat_angles_PV):
        '''
        stick angles w/ pelvis in velocity to stick angles w/ pelvis in CC
        First time point is 0
        '''
        N, D = stick_mat_angles_PV.shape
        stick_mat_angles = np.zeros([N,D])
        stick_mat_angles[:,2:] = stick_mat_angles_PV[:,2:]
        for n in range(N):
            if n == 0:
                stick_mat_angles[0, 0:2] = sum(stick_mat_angles_PV[:,:2])/-2
            else:
                stick_mat_angles[n,0:2] = stick_mat_angles[n-1,0:2] + stick_mat_angles_PV[n,0:2]
        return stick_mat_angles



    def stick_mat_angles_to_stick_dicts_CCs(self,stick_mat_angles):
        self.stick_dicts = self.stick_mat_to_dict(self.stick_mat_angles_to_CC(stick_mat_angles))

    def HGPLVM_angles_to_stick_dicts_CCs(self,stick_mat_angles):
        stick_mat_angles = stick_mat_angles.T
        stick_mat_angles = stick_mat_angles.reshape([stick_mat_angles.shape[0], stick_mat_angles.shape[1], 1])
        self.stick_dicts = self.stick_mat_to_dict(self.stick_mat_angles_to_CC(stick_mat_angles))
        return self.stick_dicts

    def HGPLVM_angles_PV_to_stick_dicts_CCs(self, stick_mat_angles_PV):
        stick_mat_angles = self.SAPV_SAPCC(stick_mat_angles_PV)
        return self.HGPLVM_angles_to_stick_dicts_CCs(stick_mat_angles)

    def plot_animation_from_angles(self,stick_mat_angles):
        self.stick_mat_angles_to_stick_dicts_CCs(stick_mat_angles)
        fig, animate = self.plot_animation()
        return fig, animate

    def plot_animation_from_directory(self,image_directory):
        self.images_to_stick_dicts_CC(image_directory)
        fig, animate = self.plot_animation()
        return fig, animate

    def plot_animation(self):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-self.x_dim, self.x_dim), ylim=(-self.y_dim, self.y_dim))
        ax.set_aspect('equal')
        ax.grid()
        color = 'b'
        lw = 3
        line = '-'

        stick_dicts = self.stick_dicts
        body, = ax.plot([], [], line, lw=lw, color=color)
        larm, = ax.plot([], [], line, lw=lw, color=color)
        rarm, = ax.plot([], [], line, lw=lw, color=color)
        lleg, = ax.plot([], [], line, lw=lw, color=color)
        rleg, = ax.plot([], [], line, lw=lw, color=color)

        def animate(t):
            bodyx = [stick_dicts["head"]["CC"][t, 0], stick_dicts["sternum"]["CC"][t, 0],
                     stick_dicts["torso"]["CC"][t, 0], stick_dicts["pelvis"]["CC"][t, 0]]
            bodyy = [stick_dicts["head"]["CC"][t, 1], stick_dicts["sternum"]["CC"][t, 1],
                     stick_dicts["torso"]["CC"][t, 1], stick_dicts["pelvis"]["CC"][t, 1]]
            body.set_data(bodyx, bodyy)

            larmx = [stick_dicts["sternum"]["CC"][t, 0], stick_dicts["left shoulder"]["CC"][t, 0],
                     stick_dicts["left hand"]["CC"][t, 0]]
            larmy = [stick_dicts["sternum"]["CC"][t, 1], stick_dicts["left shoulder"]["CC"][t, 1],
                     stick_dicts["left hand"]["CC"][t, 1]]
            larm.set_data(larmx, larmy)

            rarmx = [stick_dicts["sternum"]["CC"][t, 0], stick_dicts["right shoulder"]["CC"][t, 0],
                     stick_dicts["right hand"]["CC"][t, 0]]
            rarmy = [stick_dicts["sternum"]["CC"][t, 1], stick_dicts["right shoulder"]["CC"][t, 1],
                     stick_dicts["right hand"]["CC"][t, 1]]
            rarm.set_data(rarmx, rarmy)

            llegx = [stick_dicts["pelvis"]["CC"][t, 0], stick_dicts["left knee"]["CC"][t, 0],
                     stick_dicts["left foot"]["CC"][t, 0]]
            llegy = [stick_dicts["pelvis"]["CC"][t, 1], stick_dicts["left knee"]["CC"][t, 1],
                     stick_dicts["left foot"]["CC"][t, 1]]
            lleg.set_data(llegx, llegy)

            rlegx = [stick_dicts["pelvis"]["CC"][t, 0], stick_dicts["right knee"]["CC"][t, 0],
                     stick_dicts["right foot"]["CC"][t, 0]]
            rlegy = [stick_dicts["pelvis"]["CC"][t, 1], stick_dicts["right knee"]["CC"][t, 1],
                     stick_dicts["right foot"]["CC"][t, 1]]
            rleg.set_data(rlegx, rlegy)

            head, = ax.plot(stick_dicts["head"]["CC"][t, 0], stick_dicts["head"]["CC"][t, 1], 'o', lw=2,
                            markersize=20, color=color)

            return body, larm, rarm, lleg, rleg, head

        return fig, animate

    def set_color_key(self, img_color_scale):
        self.node_color_list = []
        if img_color_scale.lower() == 'rgb':
            self.node_color_list.append(np.array([255, 128, 255]))  # pelvis
            self.node_color_list.append(np.array([64, 0, 0]))  # lknee
            self.node_color_list.append(np.array([192, 192, 192]))  # rknee
            self.node_color_list.append(np.array([255, 255, 0]))  # torso
            self.node_color_list.append(np.array([115, 0, 128]))  # lfoot
            self.node_color_list.append(np.array([0, 128, 192]))  # rfoot
            self.node_color_list.append(np.array([255, 0, 0]))  # str
            self.node_color_list.append(np.array([0, 255, 0]))  # lsho
            self.node_color_list.append(np.array([255, 128, 0]))  # head
            self.node_color_list.append(np.array([128, 0, 255]))  # rsho
            self.node_color_list.append(np.array([0, 0, 255]))  # lhand
            self.node_color_list.append(np.array([64, 128, 128]))  # rhand
        elif img_color_scale.lower() == 'gs':
            self.node_color_list.append(np.array([255, 128, 255]))  # pelvis
            self.node_color_list.append(np.array([64, 0, 0]))  # lknee
            self.node_color_list.append(np.array([255, 0, 128]))  # rknee
            self.node_color_list.append(np.array([255, 255, 0]))  # torso
            self.node_color_list.append(np.array([115, 0, 128]))  # lfoot
            self.node_color_list.append(np.array([0, 128, 192]))  # rfoot
            self.node_color_list.append(np.array([255, 0, 0]))  # str
            self.node_color_list.append(np.array([0, 255, 0]))  # lsho
            self.node_color_list.append(np.array([255, 128, 0]))  # head
            self.node_color_list.append(np.array([128, 0, 255]))  # rsho
            self.node_color_list.append(np.array([0, 0, 255]))  # lhand
            self.node_color_list.append(np.array([64, 128, 128]))  # rhand
        else:
            raise NotImplementedError

    def location_to_angles(self,location,num_time_points=100):
        self.images_to_stick_dicts_CC(location)
        stick_mat_CC = self.get_2D_CC_mat()
        stick_mat_CC = self.interp(num_time_points)
        return self.stick_mat_CC_to_angles(stick_mat_CC).squeeze().T

    def location_to_angles_PV(self, location, num_time_points=100):
        stick_mat_angles = self.location_to_angles(location, num_time_points)
        return self.SAPCC_to_SAPV(stick_mat_angles)
"""
        for key in self.stick_dicts.keys():
            exec(str(key).replace(" ", "_") + ", = ax.plot([], [], line, lw=lw, color=color)")


        def animate(t):
            for key in self.stick_dicts.keys():
                exec(str(key).replace(" ", "_") + ".set_data(stick_dict[\"" + str(
                    key) + "\"][\"CC\"][:,0], stick_dict[\"" + str(key) + "\"][\"CC\"][:,1])")

            return pelvis, left_knee
"""