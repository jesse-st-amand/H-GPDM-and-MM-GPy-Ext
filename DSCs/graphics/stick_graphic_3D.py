from ..graphics.stick_figure_3D import StickFigure
from ..graphics.stick_figure_3D import StickFigureCMU
import numpy as np
from scipy.interpolate import interp1d

class StickGraphicBase():
    def __init__(self, img_color_scale='rgb'):
        self.num_tps = None
        self.stick_dict = None
        self.pelvis_init = None
        self.x_dim = None
        self.y_dim = None

    def get_2D_CC_mat(self):
        num_coors = self.stick_dict["Hips"]['CC'].shape[1]
        CC_mat_2D = np.zeros([self.num_nodes*num_coors,self.num_tps])
        for i,key in enumerate(self.stick_dict.keys()):
            CC_mat_2D[(((i+1)*num_coors)-2):(i+1)*num_coors,:] = self.stick_dict[key]['CC'].T
        return CC_mat_2D

    def get_3D_CC_mat_from_2D(self,CC_mat_2D):
        num_coors = int(CC_mat_2D.shape[0]/self.num_nodes)
        T = CC_mat_2D.shape[1]
        stick_mat = np.zeros([self.num_nodes,T,num_coors])
        for i in range(self.num_nodes):
            stick_mat[i,:,:] = CC_mat_2D[(((i+1)*num_coors)-2):(i+1)*num_coors,:].T
        return stick_mat

    def get_3D_CC_mat(self):
        num_coors = list(self.stick_dict["Hips"]['CC'].values).shape[1]
        CC_mat_3D = np.zeros([self.num_nodes,self.num_tps,num_coors])
        for i,key in enumerate(self.stick_dict.keys()):
            CC_mat_3D[i,:,:] = self.stick_dict[key]['CC'].T
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
        x = np.linspace(0, self.num_tps - 1, num=self.num_tps, endpoint=True)
        x2 = np.linspace(0, self.num_tps - 1, num=T, endpoint=True)
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
    def SAPV_list_2_SAPCC_list(self, stick_mat_angles_PV_list):
        SAPCC_list = []
        for stick_mat_angles in stick_mat_angles_PV_list:
            SAPCC_list.append(self.SAPV_SAPCC(stick_mat_angles))
        return SAPCC_list
    def SAPCC_to_SAPV(self, stick_mat_angles):
        '''
        stick angles w/ pelvis in CC to stick angles w/ pelvis in velocity
        First time point is 0
        '''
        N, D = stick_mat_angles.shape
        #stick_mat_angles_PV = stick_mat_angles
        stick_mat_angles_PV = np.zeros([N, D])
        stick_mat_angles_PV[:, 3:] = stick_mat_angles[:, 3:]
        stick_mat_angles_PV[1:, :3] = stick_mat_angles[1:N, 0:3] - stick_mat_angles[0:(N - 1), 0:3]
        stick_mat_angles_PV[0, :3] = stick_mat_angles_PV[1, :3]
        return stick_mat_angles_PV

    def SAPV_SAPCC(self, stick_mat_angles_PV):
        '''
        stick angles w/ pelvis in velocity to stick angles w/ pelvis in CC
        First time point is 0
        '''
        N, D = stick_mat_angles_PV.shape
        stick_mat_angles = np.zeros([N,D])
        stick_mat_angles[:,3:] = stick_mat_angles_PV[:,3:]
        for n in range(N):
            if n == 0:
                stick_mat_angles[0, 0:3] = sum(stick_mat_angles_PV[:,:3])/-2
            else:
                stick_mat_angles[n,0:3] = stick_mat_angles[n-1,0:3] + stick_mat_angles_PV[n,0:3]
        return stick_mat_angles

    def stick_mat_angles_to_stick_dicts_CCs(self,stick_mat_angles):
        self.stick_dict = self.stick_mat_to_dict(self.stick_mat_angles_to_CC(stick_mat_angles))

    def HGPLVM_angles_to_stick_dicts_CCs(self,stick_mat_angles):
        stick_mat_angles = stick_mat_angles.T
        stick_mat_angles = stick_mat_angles.reshape([stick_mat_angles.shape[0], stick_mat_angles.shape[1], 1])
        self.stick_dict = self.stick_mat_to_dict(self.stick_mat_angles_to_CC(stick_mat_angles))
        return self.stick_dict

    def HGPLVM_angles_PV_to_stick_dicts_CCs(self, stick_mat_angles_PV):
        #stick_mat_angles = self.SAPV_SAPCC(stick_mat_angles_PV)
        return self.HGPLVM_angles_to_stick_dicts_CCs(stick_mat_angles_PV)

    def HGPLVM_stick_mat_CCs_to_angles_PV(self, stick_mat):
        stick_mat_angles = self.stick_mat_CC_to_angles(stick_mat)
        return stick_mat_angles#self.SAPCC_to_SAPV(stick_mat_angles) #Old format had PV in velocity. Might be relevant later

    def HGPLVM_angles_PV_to_stick_mat_CCs(self, stick_mat_angles_PV):
        #stick_mat_angles = self.SAPV_SAPCC(stick_mat_angles_PV) #Old format had PV in velocity. Might be relevant later
        stick_mat_angles = stick_mat_angles_PV.T
        stick_mat_angles = stick_mat_angles.reshape([stick_mat_angles.shape[0], stick_mat_angles.shape[1], 1])
        return self.stick_mat_angles_to_CC(stick_mat_angles)

class StickGraphic(StickGraphicBase):
    def __init__(self, img_color_scale='rgb'):
        self.num_nodes = 58
        super().__init__(img_color_scale)

    def stick_mat_CC_to_angles(self,stick_mat_CCs):
        self.num_nodes, self.num_tps, DoFs = stick_mat_CCs.shape
        stick_mat_angles = np.array([], dtype=np.int64)
        for i in range(self.num_tps):
            stick_dict = self.stick_mat_to_dict(stick_mat_CCs[:,i,:].reshape([self.num_nodes, 1, DoFs]))
            SF = StickFigure(stick_dict)
            if i == 0:
                stick_mat_angles = SF.angles.reshape([1, -1])
            else:
                stick_mat_angles = np.concatenate([stick_mat_angles, SF.angles.reshape([1, -1])],0)
        return stick_mat_angles

    def stick_mat_angles_to_CC(self,stick_mat_angles):
        rows,self.num_tps,DoFs = stick_mat_angles.shape

        for i in range(self.num_tps):
            stick_dict = self.stick_mat_to_dict(stick_mat_angles[:,i,:], CCs=False)
            SF = StickFigure(stick_dict)
            if i == 0:
                stick_mat_CCs = SF.CCs.reshape([self.num_nodes,1,3])
            else:
                stick_mat_CCs = np.concatenate([stick_mat_CCs,SF.CCs.reshape([self.num_nodes,1,3])],1)
        return stick_mat_CCs

    def stick_mat_CC_list_to_CC_dict_list(self, stick_mat_list):
        dict_list = []
        for stick_mat in stick_mat_list:
            dict_list.append(self.stick_mat_to_dict(stick_mat,CCs=False))
        return dict_list

    def CC_2D_list_to_stick_dict_list(self, stick_mat_list):
        return [self.CC_2D_array_to_stick_dict(stick_mat) for stick_mat in stick_mat_list]

    def CC_2D_array_to_stick_dict(self, sequences):
        # Define the keys in the order they appear in the flattened array
        keys = [
            "Hips", "Spine", "Head", "Head_End",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
            "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle3_End",
            "LeftHandRing", "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing3_End",
            "LeftHandPinky", "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky3_End",
            "LeftHandIndex", "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex3_End",
            "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumb3_End",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand",
            "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle3_End",
            "RightHandRing", "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing3_End",
            "RightHandPinky", "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky3_End",
            "RightHandIndex", "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex3_End",
            "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumb3_End"
        ]

        # Calculate the number of columns for each key
        columns_per_key = sequences.shape[1] // len(keys)

        # Create the dictionary
        stick_dict = {}
        start = 0
        for key in keys:
            end = start + columns_per_key
            stick_dict[key] = {"CC": sequences[:, start:end]}
            start = end

        return stick_dict

    def stick_mat_to_dict(self,stick_mat,CCs=True):
        if CCs == True:
            stick_dict = {
                "Hips": {"CC": stick_mat[0, :, :].squeeze()},
                "Spine": {"CC": stick_mat[1, :, :].squeeze()},
                "Head": {"CC": stick_mat[2, :, :].squeeze()},
                "Head_End": {"CC": stick_mat[3, :, :].squeeze()},

                "LeftShoulder": {"CC": stick_mat[4, :, :].squeeze()},
                "LeftArm": {"CC": stick_mat[5, :, :].squeeze()},
                "LeftForeArm": {"CC": stick_mat[6, :, :].squeeze()},
                "LeftHand": {"CC": stick_mat[7, :, :].squeeze()},
                "LeftHandMiddle1": {"CC": stick_mat[8, :, :].squeeze()},
                "LeftHandMiddle2": {"CC": stick_mat[9, :, :].squeeze()},
                "LeftHandMiddle3": {"CC": stick_mat[10, :, :].squeeze()},
                "LeftHandMiddle3_End": {"CC": stick_mat[11, :, :].squeeze()},
                "LeftHandRing": {"CC": stick_mat[12, :, :].squeeze()},
                "LeftHandRing1": {"CC": stick_mat[13, :, :].squeeze()},
                "LeftHandRing2": {"CC": stick_mat[14, :, :].squeeze()},
                "LeftHandRing3": {"CC": stick_mat[15, :, :].squeeze()},
                "LeftHandRing3_End": {"CC": stick_mat[16, :, :].squeeze()},
                "LeftHandPinky": {"CC": stick_mat[17, :, :].squeeze()},
                "LeftHandPinky1": {"CC": stick_mat[18, :, :].squeeze()},
                "LeftHandPinky2": {"CC": stick_mat[19, :, :].squeeze()},
                "LeftHandPinky3": {"CC": stick_mat[20, :, :].squeeze()},
                "LeftHandPinky3_End": {"CC": stick_mat[21, :, :].squeeze()},
                "LeftHandIndex": {"CC": stick_mat[22, :, :].squeeze()},
                "LeftHandIndex1": {"CC": stick_mat[23, :, :].squeeze()},
                "LeftHandIndex2": {"CC": stick_mat[24, :, :].squeeze()},
                "LeftHandIndex3": {"CC": stick_mat[25, :, :].squeeze()},
                "LeftHandIndex3_End": {"CC": stick_mat[26, :, :].squeeze()},
                "LeftHandThumb1": {"CC": stick_mat[27, :, :].squeeze()},
                "LeftHandThumb2": {"CC": stick_mat[28, :, :].squeeze()},
                "LeftHandThumb3": {"CC": stick_mat[29, :, :].squeeze()},
                "LeftHandThumb3_End": {"CC": stick_mat[30, :, :].squeeze()},

                "RightShoulder": {"CC": stick_mat[31, :, :].squeeze()},
                "RightArm": {"CC": stick_mat[32, :, :].squeeze()},
                "RightForeArm": {"CC": stick_mat[33, :, :].squeeze()},
                "RightHand": {"CC": stick_mat[34, :, :].squeeze()},
                "RightHandMiddle1": {"CC": stick_mat[35, :, :].squeeze()},
                "RightHandMiddle2": {"CC": stick_mat[36, :, :].squeeze()},
                "RightHandMiddle3": {"CC": stick_mat[37, :, :].squeeze()},
                "RightHandMiddle3_End": {"CC": stick_mat[38, :, :].squeeze()},
                "RightHandRing": {"CC": stick_mat[39, :, :].squeeze()},
                "RightHandRing1": {"CC": stick_mat[40, :, :].squeeze()},
                "RightHandRing2": {"CC": stick_mat[41, :, :].squeeze()},
                "RightHandRing3": {"CC": stick_mat[42, :, :].squeeze()},
                "RightHandRing3_End": {"CC": stick_mat[43, :, :].squeeze()},
                "RightHandPinky": {"CC": stick_mat[44, :, :].squeeze()},
                "RightHandPinky1": {"CC": stick_mat[45, :, :].squeeze()},
                "RightHandPinky2": {"CC": stick_mat[46, :, :].squeeze()},
                "RightHandPinky3": {"CC": stick_mat[47, :, :].squeeze()},
                "RightHandPinky3_End": {"CC": stick_mat[48, :, :].squeeze()},
                "RightHandIndex": {"CC": stick_mat[49, :, :].squeeze()},
                "RightHandIndex1": {"CC": stick_mat[50, :, :].squeeze()},
                "RightHandIndex2": {"CC": stick_mat[51, :, :].squeeze()},
                "RightHandIndex3": {"CC": stick_mat[52, :, :].squeeze()},
                "RightHandIndex3_End": {"CC": stick_mat[53, :, :].squeeze()},
                "RightHandThumb1": {"CC": stick_mat[54, :, :].squeeze()},
                "RightHandThumb2": {"CC": stick_mat[55, :, :].squeeze()},
                "RightHandThumb3": {"CC": stick_mat[56, :, :].squeeze()},
                "RightHandThumb3_End": {"CC": stick_mat[57, :, :].squeeze()}
            }
        else: #input in angles
            index_list = []
            index_list.append(np.array([0,1,2]))
            for i in range(self.num_nodes):
                index_list.append(np.array([(i*2)+3,(i*2)+4]))
            stick_dict = {
                "Hips": {"angle": None, "CC": stick_mat[index_list[0], :].squeeze()},

                "Spine": {"angle": stick_mat[index_list[1], :].squeeze()},
                "Head": {"angle": stick_mat[index_list[2], :].squeeze()},
                "Head_End": {"angle": stick_mat[index_list[3], :].squeeze()},

                "LeftShoulder": {"angle": stick_mat[index_list[4], :].squeeze()},
                "LeftArm": {"angle": stick_mat[index_list[5], :].squeeze()},
                "LeftForeArm": {"angle": stick_mat[index_list[6], :].squeeze()},
                "LeftHand": {"angle": stick_mat[index_list[7], :].squeeze()},
                "LeftHandMiddle1": {"angle": stick_mat[index_list[8], :].squeeze()},
                "LeftHandMiddle2": {"angle": stick_mat[index_list[9], :].squeeze()},
                "LeftHandMiddle3": {"angle": stick_mat[index_list[10], :].squeeze()},
                "LeftHandMiddle3_End": {"angle": stick_mat[index_list[11], :].squeeze()},
                "LeftHandRing": {"angle": stick_mat[index_list[12], :].squeeze()},
                "LeftHandRing1": {"angle": stick_mat[index_list[13], :].squeeze()},
                "LeftHandRing2": {"angle": stick_mat[index_list[14], :].squeeze()},
                "LeftHandRing3": {"angle": stick_mat[index_list[15], :].squeeze()},
                "LeftHandRing3_End": {"angle": stick_mat[index_list[16], :].squeeze()},
                "LeftHandPinky": {"angle": stick_mat[index_list[17], :].squeeze()},
                "LeftHandPinky1": {"angle": stick_mat[index_list[18], :].squeeze()},
                "LeftHandPinky2": {"angle": stick_mat[index_list[19], :].squeeze()},
                "LeftHandPinky3": {"angle": stick_mat[index_list[20], :].squeeze()},
                "LeftHandPinky3_End": {"angle": stick_mat[index_list[21], :].squeeze()},
                "LeftHandIndex": {"angle": stick_mat[index_list[22], :].squeeze()},
                "LeftHandIndex1": {"angle": stick_mat[index_list[23], :].squeeze()},
                "LeftHandIndex2": {"angle": stick_mat[index_list[24], :].squeeze()},
                "LeftHandIndex3": {"angle": stick_mat[index_list[25], :].squeeze()},
                "LeftHandIndex3_End": {"angle": stick_mat[index_list[26], :].squeeze()},
                "LeftHandThumb1": {"angle": stick_mat[index_list[27], :].squeeze()},
                "LeftHandThumb2": {"angle": stick_mat[index_list[28], :].squeeze()},
                "LeftHandThumb3": {"angle": stick_mat[index_list[29], :].squeeze()},
                "LeftHandThumb3_End": {"angle": stick_mat[index_list[30], :].squeeze()},

                "RightShoulder": {"angle": stick_mat[index_list[31], :].squeeze()},
                "RightArm": {"angle": stick_mat[index_list[32], :].squeeze()},
                "RightForeArm": {"angle": stick_mat[index_list[33], :].squeeze()},
                "RightHand": {"angle": stick_mat[index_list[34], :].squeeze()},
                "RightHandMiddle1": {"angle": stick_mat[index_list[35], :].squeeze()},
                "RightHandMiddle2": {"angle": stick_mat[index_list[36], :].squeeze()},
                "RightHandMiddle3": {"angle": stick_mat[index_list[37], :].squeeze()},
                "RightHandMiddle3_End": {"angle": stick_mat[index_list[38], :].squeeze()},
                "RightHandRing": {"angle": stick_mat[index_list[39], :].squeeze()},
                "RightHandRing1": {"angle": stick_mat[index_list[40], :].squeeze()},
                "RightHandRing2": {"angle": stick_mat[index_list[41], :].squeeze()},
                "RightHandRing3": {"angle": stick_mat[index_list[42], :].squeeze()},
                "RightHandRing3_End": {"angle": stick_mat[index_list[43], :].squeeze()},
                "RightHandPinky": {"angle": stick_mat[index_list[44], :].squeeze()},
                "RightHandPinky1": {"angle": stick_mat[index_list[45], :].squeeze()},
                "RightHandPinky2": {"angle": stick_mat[index_list[46], :].squeeze()},
                "RightHandPinky3": {"angle": stick_mat[index_list[47], :].squeeze()},
                "RightHandPinky3_End": {"angle": stick_mat[index_list[48], :].squeeze()},
                "RightHandIndex": {"angle": stick_mat[index_list[49], :].squeeze()},
                "RightHandIndex1": {"angle": stick_mat[index_list[50], :].squeeze()},
                "RightHandIndex2": {"angle": stick_mat[index_list[51], :].squeeze()},
                "RightHandIndex3": {"angle": stick_mat[index_list[52], :].squeeze()},
                "RightHandIndex3_End": {"angle": stick_mat[index_list[53], :].squeeze()},
                "RightHandThumb1": {"angle": stick_mat[index_list[54], :].squeeze()},
                "RightHandThumb2": {"angle": stick_mat[index_list[55], :].squeeze()},
                "RightHandThumb3": {"angle": stick_mat[index_list[56], :].squeeze()},
                "RightHandThumb3_End": {"angle": stick_mat[index_list[57], :].squeeze()}
            }
        self.stick_dict = stick_dict
        return self.stick_dict


class StickGraphicCMU(StickGraphicBase):
    def __init__(self, img_color_scale='rgb'):
        self.num_nodes = 38
        super().__init__(img_color_scale)

    def stick_mat_CC_to_angles(self,stick_mat_CCs):
        self.num_nodes, self.num_tps, DoFs = stick_mat_CCs.shape
        stick_mat_angles = np.array([], dtype=np.int64)
        for i in range(self.num_tps):
            stick_dict = self.stick_mat_to_dict(stick_mat_CCs[:,i,:].reshape([self.num_nodes, 1, DoFs]))
            SF = StickFigureCMU(stick_dict)
            if i == 0:
                stick_mat_angles = SF.angles.reshape([1, -1])
            else:
                stick_mat_angles = np.concatenate([stick_mat_angles, SF.angles.reshape([1, -1])],0)
        return stick_mat_angles

    def stick_mat_angles_to_CC(self,stick_mat_angles):
        rows,self.num_tps,DoFs = stick_mat_angles.shape

        for i in range(self.num_tps):
            stick_dict = self.stick_mat_to_dict(stick_mat_angles[:,i,:], CCs=False)
            SF = StickFigureCMU(stick_dict)
            if i == 0:
                stick_mat_CCs = SF.CCs.reshape([self.num_nodes,1,3])
            else:
                stick_mat_CCs = np.concatenate([stick_mat_CCs,SF.CCs.reshape([self.num_nodes,1,3])],1)
        return stick_mat_CCs

    def CC_2D_array_to_stick_dict(self, sequences):
        # Define the keys in the order they appear in the flattened array
        keys = [
            "Hips",
            "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_End",
            "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_End",
            "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head", "Head_End",
            "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1",
            "LeftHandIndex1_End",
            "LThumb", "LThumb_End",
            "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightFingerBase", "RightHandIndex1",
            "RightHandIndex1_End",
            "RThumb", "RThumb_End"
        ]

        # Calculate the number of columns for each key
        columns_per_key = sequences.shape[1] // len(keys)

        # Create the dictionary
        stick_dict = {}
        start = 0
        for key in keys:
            end = start + columns_per_key
            if key == 'Hips':
                stick_dict[key] = {"CC": sequences[:, start:end]}
            else:
                stick_dict[key] = {"CC": sequences[:, start:end] + stick_dict['Hips']['CC']}
            start = end

        return stick_dict

    def stick_mat_to_dict(self,stick_mat,CCs=True):
        if CCs == True:
            stick_dict = {
                "Hips": {"CC": stick_mat[0, :, :].squeeze()},

                "LHipJoint": {"CC": stick_mat[1, :, :].squeeze()},
                "LeftUpLeg": {"CC": stick_mat[2, :, :].squeeze()},
                "LeftLeg": {"CC": stick_mat[3, :, :].squeeze()},
                "LeftFoot": {"CC": stick_mat[4, :, :].squeeze()},
                "LeftToeBase": {"CC": stick_mat[5, :, :].squeeze()},
                "LeftToeBase_End": {"CC": stick_mat[6, :, :].squeeze()},

                "RHipJoint": {"CC": stick_mat[7, :, :].squeeze()},
                "RightUpLeg": {"CC": stick_mat[8, :, :].squeeze()},
                "RightLeg": {"CC": stick_mat[9, :, :].squeeze()},
                "RightFoot": {"CC": stick_mat[10, :, :].squeeze()},
                "RightToeBase": {"CC": stick_mat[11, :, :].squeeze()},
                "RightToeBase_End": {"CC": stick_mat[12, :, :].squeeze()},

                "LowerBack": {"CC": stick_mat[13, :, :].squeeze()},
                "Spine": {"CC": stick_mat[14, :, :].squeeze()},
                "Spine1": {"CC": stick_mat[15, :, :].squeeze()},
                "Neck": {"CC": stick_mat[16, :, :].squeeze()},
                "Neck1": {"CC": stick_mat[17, :, :].squeeze()},
                "Head": {"CC": stick_mat[18, :, :].squeeze()},
                "Head_End": {"CC": stick_mat[19, :, :].squeeze()},

                "LeftShoulder": {"CC": stick_mat[20, :, :].squeeze()},
                "LeftArm": {"CC": stick_mat[21, :, :].squeeze()},
                "LeftForeArm": {"CC": stick_mat[22, :, :].squeeze()},
                "LeftHand": {"CC": stick_mat[23, :, :].squeeze()},
                "LeftFingerBase": {"CC": stick_mat[24, :, :].squeeze()},
                "LeftHandIndex1": {"CC": stick_mat[25, :, :].squeeze()},
                "LeftHandIndex1_End": {"CC": stick_mat[26, :, :].squeeze()},

                "LThumb": {"CC": stick_mat[27, :, :].squeeze()},
                "LThumb_End": {"CC": stick_mat[28, :, :].squeeze()},

                "RightShoulder": {"CC": stick_mat[29, :, :].squeeze()},
                "RightArm": {"CC": stick_mat[30, :, :].squeeze()},
                "RightForeArm": {"CC": stick_mat[31, :, :].squeeze()},
                "RightHand": {"CC": stick_mat[32, :, :].squeeze()},
                "RightFingerBase": {"CC": stick_mat[33, :, :].squeeze()},
                "RightHandIndex1": {"CC": stick_mat[34, :, :].squeeze()},
                "RightHandIndex1_End": {"CC": stick_mat[35, :, :].squeeze()},

                "RThumb": {"CC": stick_mat[36, :, :].squeeze()},
                "RThumb_End": {"CC": stick_mat[37, :, :].squeeze()},

            }
        else: #input in angles
            index_list = []
            index_list.append(np.array([0,1,2]))
            for i in range(self.num_nodes):
                index_list.append(np.array([(i*2)+3,(i*2)+4]))
            stick_dict = {
                "Hips": {"angle": None, "CC": stick_mat[index_list[0], :].squeeze()},

                "LHipJoint": {"angle": stick_mat[index_list[1], :].squeeze()},
                "LeftUpLeg": {"angle": stick_mat[index_list[2], :].squeeze()},
                "LeftLeg": {"angle": stick_mat[index_list[3], :].squeeze()},
                "LeftFoot": {"angle": stick_mat[index_list[4], :].squeeze()},
                "LeftToeBase": {"angle": stick_mat[index_list[5], :].squeeze()},
                "LeftToeBase_End": {"angle": stick_mat[index_list[6], :].squeeze()},

                "RHipJoint": {"angle": stick_mat[index_list[7], :].squeeze()},
                "RightUpLeg": {"angle": stick_mat[index_list[8], :].squeeze()},
                "RightLeg": {"angle": stick_mat[index_list[9], :].squeeze()},
                "RightFoot": {"angle": stick_mat[index_list[10], :].squeeze()},
                "RightToeBase": {"angle": stick_mat[index_list[11], :].squeeze()},
                "RightToeBase_End": {"angle": stick_mat[index_list[12], :].squeeze()},

                "LowerBack": {"angle": stick_mat[index_list[13], :].squeeze()},
                "Spine": {"angle": stick_mat[index_list[14], :].squeeze()},
                "Spine1": {"angle": stick_mat[index_list[15], :].squeeze()},
                "Neck": {"angle": stick_mat[index_list[16], :].squeeze()},
                "Neck1": {"angle": stick_mat[index_list[17], :].squeeze()},
                "Head": {"angle": stick_mat[index_list[18], :].squeeze()},
                "Head_End": {"angle": stick_mat[index_list[19], :].squeeze()},

                "LeftShoulder": {"angle": stick_mat[index_list[20], :].squeeze()},
                "LeftArm": {"angle": stick_mat[index_list[21], :].squeeze()},
                "LeftForeArm": {"angle": stick_mat[index_list[22], :].squeeze()},
                "LeftHand": {"angle": stick_mat[index_list[23], :].squeeze()},
                "LeftFingerBase": {"angle": stick_mat[index_list[24], :].squeeze()},
                "LeftHandIndex1": {"angle": stick_mat[index_list[25], :].squeeze()},
                "LeftHandIndex1_End": {"angle": stick_mat[index_list[26], :].squeeze()},
                "LThumb": {"angle": stick_mat[index_list[27], :].squeeze()},
                "LThumb_End": {"angle": stick_mat[index_list[28], :].squeeze()},

                "RightShoulder": {"angle": stick_mat[index_list[29], :].squeeze()},
                "RightArm": {"angle": stick_mat[index_list[30], :].squeeze()},
                "RightForeArm": {"angle": stick_mat[index_list[31], :].squeeze()},
                "RightHand": {"angle": stick_mat[index_list[32], :].squeeze()},
                "RightFingerBase": {"angle": stick_mat[index_list[33], :].squeeze()},
                "RightHandIndex1": {"angle": stick_mat[index_list[34], :].squeeze()},
                "RightHandIndex1_End": {"angle": stick_mat[index_list[35], :].squeeze()},
                "RThumb": {"angle": stick_mat[index_list[36], :].squeeze()},
                "RThumb_End": {"angle": stick_mat[index_list[37], :].squeeze()},
            }
        self.stick_dict = stick_dict
        return self.stick_dict