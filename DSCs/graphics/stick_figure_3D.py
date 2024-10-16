import numpy as np
from ..graphics.stick_node_3D import StickNode
import matplotlib.pyplot as plt
class StickFigureBase():
    def __init__(self,stick_dict):
        self.stick_dict = stick_dict
        self.input = list(stick_dict["Hips"].items())[0][0]
        self.Hips = None
        self.top_node = None
        self.num_nodes = 0
        self.create_hierarchy()
        self.CCs = np.zeros([self.num_nodes,3])
        self.angles = np.zeros((self.num_nodes*2)+1)
        self.lengths = np.zeros(self.num_nodes)
        self.setup_stick_figure()

    def get_angles_mat(self,node_mat):
        pass

    def setup_stick_figure(self, StickNode = None):
        if StickNode is None:
            StickNode = self.Hips
        self.set_node_parameters(StickNode)
        for child in StickNode.mChild:
            self.setup_stick_figure(child)

    def set_node_parameters(self, StickNode):
        StickNode.set_stick_figure(self)
        if self.input == "CC":
            StickNode.set_thetas_from_CCs()
        elif self.input == "angle":
            StickNode.set_CCs_from_thetas()
        else:
            print("Invalid input.")
        if StickNode.name == "Hips":
            self.angles[StickNode.ID:StickNode.ID + 3] = StickNode.CCs
        else:
            self.angles[(StickNode.ID*2)+1:(StickNode.ID*2)+3] = StickNode.thetas
        self.CCs[StickNode.ID, :] = StickNode.CCs
        self.lengths[StickNode.ID] = StickNode.length

    def get_pelvis_CC(self):
        if self.stick_dict["Hips"]["CC"] is None:
            return np.zeros(3)
        else:
            return self.stick_dict["Hips"]["CC"]

    def get_node(self,name,StickNode=None,node=None):
        if StickNode is None:
            StickNode = self.Hips
        if StickNode.name == name:
            node = StickNode
        if node is not None:
            return node
        for child in StickNode.mChild:
            node = self.get_node(name,child,node)
        return node

    def print_all_nodes(self):
        for key in list(self.stick_dict.keys()):
            self.get_node(key).print_attributes()

    def plot(self,x_dim,y_dim):
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(autoscale_on=False, xlim=(-x_dim, x_dim), ylim=(-y_dim, y_dim))
        ax.set_aspect('equal')
        ax.grid()
        for CC in self.CCs:
            plt.scatter(CC[0], CC[1])
        plt.show()
class StickFigure(StickFigureBase):
    def __init__(self,stick_dict):
        super().__init__(stick_dict)


    def create_hierarchy(self):
        '''
        ID numbers are created in the order in which the nodes are created, from 0 up.
        This defines the absolute, "1D" order of seniority, and thus, acts as the index for each node in any generated matrix

        Network hierarchy is defined by the SetChild function and will determine how the class traverses the network.
        :return:
        '''

        self.Hips = StickNode(self, "Hips")
        self.top_node = self.Hips

        Spine = StickNode(self, "Spine")
        Head = StickNode(self, "Head")
        Head_End = StickNode(self, "Head_End")
        #left arm
        LeftShoulder = StickNode(self, "LeftShoulder")
        LeftArm = StickNode(self, "LeftArm")
        LeftForeArm = StickNode(self, "LeftForeArm")
        #left hand
        LeftHand = StickNode(self, "LeftHand")
        LeftHandMiddle1 = StickNode(self, "LeftHandMiddle1")
        LeftHandMiddle2 = StickNode(self, "LeftHandMiddle2")
        LeftHandMiddle3 = StickNode(self, "LeftHandMiddle3")
        LeftHandMiddle3_End = StickNode(self, "LeftHandMiddle3_End")

        LeftHandRing = StickNode(self, "LeftHandRing")
        LeftHandRing1 = StickNode(self, "LeftHandRing1")
        LeftHandRing2 = StickNode(self, "LeftHandRing2")
        LeftHandRing3 = StickNode(self, "LeftHandRing3")
        LeftHandRing3_End = StickNode(self, "LeftHandRing3_End")

        LeftHandPinky = StickNode(self, "LeftHandPinky")
        LeftHandPinky1 = StickNode(self, "LeftHandPinky1")
        LeftHandPinky2 = StickNode(self, "LeftHandPinky2")
        LeftHandPinky3 = StickNode(self, "LeftHandPinky3")
        LeftHandPinky3_End = StickNode(self, "LeftHandPinky3_End")

        LeftHandIndex = StickNode(self, "LeftHandIndex")
        LeftHandIndex1 = StickNode(self, "LeftHandIndex1")
        LeftHandIndex2 = StickNode(self, "LeftHandIndex2")
        LeftHandIndex3 = StickNode(self, "LeftHandIndex3")
        LeftHandIndex3_End = StickNode(self, "LeftHandIndex3_End")

        LeftHandThumb1 = StickNode(self, "LeftHandThumb1")
        LeftHandThumb2 = StickNode(self, "LeftHandThumb2")
        LeftHandThumb3 = StickNode(self, "LeftHandThumb3")
        LeftHandThumb3_End = StickNode(self, "LeftHandThumb3_End")

        # Right arm
        RightShoulder = StickNode(self, "RightShoulder")
        RightArm = StickNode(self, "RightArm")
        RightForeArm = StickNode(self, "RightForeArm")
        # Right hand
        RightHand = StickNode(self, "RightHand")
        RightHandMiddle1 = StickNode(self, "RightHandMiddle1")
        RightHandMiddle2 = StickNode(self, "RightHandMiddle2")
        RightHandMiddle3 = StickNode(self, "RightHandMiddle3")
        RightHandMiddle3_End = StickNode(self, "RightHandMiddle3_End")

        RightHandRing = StickNode(self, "RightHandRing")
        RightHandRing1 = StickNode(self, "RightHandRing1")
        RightHandRing2 = StickNode(self, "RightHandRing2")
        RightHandRing3 = StickNode(self, "RightHandRing3")
        RightHandRing3_End = StickNode(self, "RightHandRing3_End")

        RightHandPinky = StickNode(self, "RightHandPinky")
        RightHandPinky1 = StickNode(self, "RightHandPinky1")
        RightHandPinky2 = StickNode(self, "RightHandPinky2")
        RightHandPinky3 = StickNode(self, "RightHandPinky3")
        RightHandPinky3_End = StickNode(self, "RightHandPinky3_End")

        RightHandIndex = StickNode(self, "RightHandIndex")
        RightHandIndex1 = StickNode(self, "RightHandIndex1")
        RightHandIndex2 = StickNode(self, "RightHandIndex2")
        RightHandIndex3 = StickNode(self, "RightHandIndex3")
        RightHandIndex3_End = StickNode(self, "RightHandIndex3_End")

        RightHandThumb1 = StickNode(self, "RightHandThumb1")
        RightHandThumb2 = StickNode(self, "RightHandThumb2")
        RightHandThumb3 = StickNode(self, "RightHandThumb3")
        RightHandThumb3_End = StickNode(self, "RightHandThumb3_End")


        self.Hips.SetChild(0, Spine)

        Head.SetChild(0, Head_End)
        Spine.SetChild(0, LeftShoulder)
        Spine.SetChild(1, Head)
        Spine.SetChild(2, RightShoulder)


        LeftShoulder.SetChild(0,LeftArm)
        LeftArm.SetChild(0, LeftForeArm)

        LeftForeArm.SetChild(0, LeftHand)

        LeftHand.SetChild(0, LeftHandMiddle1)
        LeftHand.SetChild(1, LeftHandRing)
        LeftHand.SetChild(2, LeftHandIndex)

        LeftHandMiddle1.SetChild(0, LeftHandMiddle2)
        LeftHandMiddle2.SetChild(0, LeftHandMiddle3)
        LeftHandMiddle3.SetChild(0, LeftHandMiddle3_End)

        LeftHandIndex.SetChild(0, LeftHandIndex1)
        LeftHandIndex1.SetChild(0, LeftHandIndex2)
        LeftHandIndex2.SetChild(0, LeftHandIndex3)
        LeftHandIndex3.SetChild(0, LeftHandIndex3_End)
        LeftHandIndex.SetChild(1, LeftHandThumb1)

        LeftHandThumb1.SetChild(0, LeftHandThumb2)
        LeftHandThumb2.SetChild(0, LeftHandThumb3)
        LeftHandThumb3.SetChild(0, LeftHandThumb3_End)

        LeftHandRing.SetChild(0, LeftHandRing1)
        LeftHandRing1.SetChild(0, LeftHandRing2)
        LeftHandRing2.SetChild(0, LeftHandRing3)
        LeftHandRing3.SetChild(0, LeftHandRing3_End)
        LeftHandRing.SetChild(1, LeftHandPinky)

        LeftHandPinky.SetChild(0, LeftHandPinky1)
        LeftHandPinky1.SetChild(0, LeftHandPinky2)
        LeftHandPinky2.SetChild(0, LeftHandPinky3)
        LeftHandPinky3.SetChild(0, LeftHandPinky3_End)

        RightShoulder.SetChild(0, RightArm)
        RightArm.SetChild(0, RightForeArm)

        RightForeArm.SetChild(0, RightHand)

        RightHand.SetChild(0, RightHandMiddle1)
        RightHand.SetChild(1, RightHandRing)
        RightHand.SetChild(2, RightHandIndex)

        RightHandMiddle1.SetChild(0, RightHandMiddle2)
        RightHandMiddle2.SetChild(0, RightHandMiddle3)
        RightHandMiddle3.SetChild(0, RightHandMiddle3_End)

        RightHandIndex.SetChild(0, RightHandIndex1)
        RightHandIndex1.SetChild(0, RightHandIndex2)
        RightHandIndex2.SetChild(0, RightHandIndex3)
        RightHandIndex3.SetChild(0, RightHandIndex3_End)
        RightHandIndex.SetChild(1, RightHandThumb1)

        RightHandThumb1.SetChild(0, RightHandThumb2)
        RightHandThumb2.SetChild(0, RightHandThumb3)
        RightHandThumb3.SetChild(0, RightHandThumb3_End)


        RightHandRing.SetChild(0, RightHandRing1)
        RightHandRing1.SetChild(0, RightHandRing2)
        RightHandRing2.SetChild(0, RightHandRing3)
        RightHandRing3.SetChild(0, RightHandRing3_End)
        RightHandRing.SetChild(1, RightHandPinky)

        RightHandPinky.SetChild(0, RightHandPinky1)
        RightHandPinky1.SetChild(0, RightHandPinky2)
        RightHandPinky2.SetChild(0, RightHandPinky3)
        RightHandPinky3.SetChild(0, RightHandPinky3_End)


    def get_generic_length(self, name):
        lengths_dict = {
         'Hips':0,
         'Spine':7.62,
         'Head':12.78,
         'Head_End':4.51,

         'LeftShoulder':8.38,
         'LeftArm':8.48,
         'LeftForeArm':10.3,
         'LeftHand': 7.16,
         'LeftHandMiddle1':3.42,
         'LeftHandMiddle2':1.83,
         'LeftHandMiddle3':1.12,
         'LeftHandMiddle3_End':0.59,
         'LeftHandRing':.36,
         'LeftHandRing1':3.04,
         'LeftHandRing2':1.7,
         'LeftHandRing3':1.02,
         'LeftHandRing3_End':.55,
         'LeftHandPinky':.34,
         'LeftHandPinky1':2.57,
         'LeftHandPinky2':1.35,
         'LeftHandPinky3':.86,
         'LeftHandPinky3_End':.58,
         'LeftHandIndex':.35,
         'LeftHandIndex1':3.29,
         'LeftHandIndex2':1.08,
         'LeftHandIndex3':1.05,
         'LeftHandIndex3_End':.64,
         'LeftHandThumb1':.42,
         'LeftHandThumb2':2.18,
         'LeftHandThumb3':1.45,
         'LeftHandThumb3_End':.82,

         'RightShoulder':8.38,
         'RightArm':8.48,
         'RightForeArm':10.3,
         'RightHand': 7.16,
         'RightHandMiddle1':3.42,
         'RightHandMiddle2':1.83,
         'RightHandMiddle3':1.12,
         'RightHandMiddle3_End':0.59,
         'RightHandRing':.36,
         'RightHandRing1':3.04,
         'RightHandRing2':1.7,
         'RightHandRing3':1.02,
         'RightHandRing3_End':.55,
         'RightHandPinky':.34,
         'RightHandPinky1':2.57,
         'RightHandPinky2':1.35,
         'RightHandPinky3':.86,
         'RightHandPinky3_End':.58,
         'RightHandIndex':.35,
         'RightHandIndex1':3.29,
         'RightHandIndex2':1.08,
         'RightHandIndex3':1.05,
         'RightHandIndex3_End':.64,
         'RightHandThumb1':.42,
         'RightHandThumb2':2.18,
         'RightHandThumb3':1.45,
         'RightHandThumb3_End':.82,
        }
        return lengths_dict[name]

class StickFigureCMU(StickFigureBase):
    def __init__(self,stick_dict):
        super().__init__(stick_dict)


    def create_hierarchy(self):
        '''
        ID numbers are created in the order in which the nodes are created, from 0 up.
        This defines the absolute, "1D" order of seniority, and thus, acts as the index for each node in any generated matrix

        Network hierarchy is defined by the SetChild function and will determine how the class traverses the network.
        :return:
        '''

        self.Hips = StickNode(self, "Hips")
        self.top_node = self.Hips

        LHipJoint = StickNode(self, "LHipJoint")
        LeftUpLeg = StickNode(self, "LeftUpLeg")
        LeftLeg = StickNode(self, "LeftLeg")
        LeftFoot = StickNode(self, "LeftFoot")
        LeftToeBase = StickNode(self, "LeftToeBase")
        LeftToeBase_End = StickNode(self, "LeftToeBase_End")

        RHipJoint = StickNode(self, "RHipJoint")
        RightUpLeg = StickNode(self, "RightUpLeg")
        RightLeg = StickNode(self, "RightLeg")
        RightFoot = StickNode(self, "RightFoot")
        RightToeBase = StickNode(self, "RightToeBase")
        RightToeBase_End = StickNode(self, "RightToeBase_End")

        LowerBack = StickNode(self, "LowerBack")
        Spine = StickNode(self, "Spine")
        Spine1 = StickNode(self, "Spine1")
        Neck = StickNode(self, "Neck")
        Neck1 = StickNode(self, "Neck1")
        Head = StickNode(self, "Head")
        Head_End = StickNode(self, "Head_End")

        LeftShoulder = StickNode(self, "LeftShoulder")
        LeftArm = StickNode(self, "LeftArm")
        LeftForeArm = StickNode(self, "LeftForeArm")
        LeftHand = StickNode(self, "LeftHand")
        LeftFingerBase = StickNode(self, "LeftFingerBase")
        LeftHandIndex1 = StickNode(self, "LeftHandIndex1")
        LeftHandIndex1_End = StickNode(self, "LeftHandIndex1_End")
        LThumb = StickNode(self, "LThumb")
        LThumb_End = StickNode(self, "LThumb_End")

        RightShoulder = StickNode(self, "RightShoulder")
        RightArm = StickNode(self, "RightArm")
        RightForeArm = StickNode(self, "RightForeArm")
        RightHand = StickNode(self, "RightHand")
        RightFingerBase = StickNode(self, "RightFingerBase")
        RightHandIndex1 = StickNode(self, "RightHandIndex1")
        RightHandIndex1_End = StickNode(self, "RightHandIndex1_End")
        RThumb = StickNode(self, "RThumb")
        RThumb_End = StickNode(self, "RThumb_End")




        self.Hips.SetChild(0, LowerBack)
        self.Hips.SetChild(1, LHipJoint)
        self.Hips.SetChild(2, RHipJoint)

        LHipJoint.SetChild(0, LeftUpLeg)
        LeftUpLeg.SetChild(0, LeftLeg)
        LeftLeg.SetChild(0, LeftFoot)
        LeftFoot.SetChild(0, LeftToeBase)
        LeftToeBase.SetChild(0, LeftToeBase_End)

        RHipJoint.SetChild(0, RightUpLeg)
        RightUpLeg.SetChild(0, RightLeg)
        RightLeg.SetChild(0, RightFoot)
        RightFoot.SetChild(0, RightToeBase)
        RightToeBase.SetChild(0, RightToeBase_End)

        LowerBack.SetChild(0, Spine)
        Spine.SetChild(0, Spine1)
        Spine1.SetChild(0, Neck)
        Neck.SetChild(0, Neck1)
        Neck1.SetChild(0, Head)
        Head.SetChild(0, Head_End)

        Spine1.SetChild(1, LeftShoulder)
        LeftShoulder.SetChild(0, LeftArm)
        LeftArm.SetChild(0, LeftForeArm)
        LeftForeArm.SetChild(0, LeftHand)
        LeftHand.SetChild(0, LeftFingerBase)
        LeftFingerBase.SetChild(0, LeftHandIndex1)
        LeftHandIndex1.SetChild(0, LeftHandIndex1_End)

        LeftHand.SetChild(1, LThumb)
        LThumb.SetChild(0, LThumb_End)

        Spine1.SetChild(2, RightShoulder)
        RightShoulder.SetChild(0, RightArm)
        RightArm.SetChild(0, RightForeArm)
        RightForeArm.SetChild(0, RightHand)
        RightHand.SetChild(0, RightFingerBase)
        RightFingerBase.SetChild(0, RightHandIndex1)
        RightHandIndex1.SetChild(0, RightHandIndex1_End)

        RightHand.SetChild(1, RThumb)
        RThumb.SetChild(0, RThumb_End)




    def get_generic_length(self, name):
        lengths_dict = {
            'Hips':0,
            'LowerBack': 0,
            'Spine':2,
            'Spine1':2,
            'Neck': 0,
            'Neck1': 1.6,
            'Head':1.7,
            'Head_End':0,

            'LeftShoulder':0,
            'LeftArm':3.6,
            'LeftForeArm':5.3,
            'LeftHand': 3.4,
            'LeftFingerBase':0,
            'LeftHandIndex1':.46,
            'LeftHandIndex1_End':0,
            'LThumb':.66,
            'LThumb_End':0,

            'RightShoulder':0,
            'RightArm':3.6,
            'RightForeArm':5.3,
            'RightHand':3.4,
            'RightFingerBase':0,
            'RightHandIndex1':.46,
            'RightHandIndex1_End':0,
            'RThumb':.66,
            'RThumb_End':0,

            "LHipJoint":0,
            "LeftUpLeg":2.5,
            "LeftLeg":7.5,
            "LeftFoot":7.3,
            "LeftToeBase":1.1,
            "LeftToeBase_End":0,

            "RHipJoint":0,
            "RightUpLeg":2.5,
            "RightLeg":7.5,
            "RightFoot":7.3,
            "RightToeBase":1.1,
            "RightToeBase_End":0,
        }


        return lengths_dict[name]

