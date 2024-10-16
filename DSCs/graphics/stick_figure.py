import numpy as np
from ..graphics.stick_node import StickNode
import matplotlib.pyplot as plt

class StickFigure():
    def __init__(self,stick_dict):
        self.stick_dict = stick_dict
        self.input = list(stick_dict["pelvis"].items())[0][0]
        self.pelvis = None
        self.top_node = None
        self.num_nodes = 0
        self.create_hierarchy()
        self.CCs = np.zeros([self.num_nodes,2])
        self.angles = np.zeros(self.num_nodes+1)
        self.lengths = np.zeros(self.num_nodes)
        self.setup_stick_figure()

    def get_angles_mat(self,node_mat):
        pass

    def create_hierarchy(self):
        '''
        ID numbers are created in the order in which the nodes are created, from 0 up.
        This defines the absolute, "1D" order of seniority, and thus, acts as the index for each node in any generated matrix

        Network hierarchy is defined by the SetChild function and will determine how the class traverses the network.
        :return:
        '''
        self.pelvis = StickNode(self, "pelvis")
        self.top_node = self.pelvis
        left_knee = StickNode(self, "left knee")
        right_knee = StickNode(self, "right knee")
        torso = StickNode(self, "torso")
        left_foot = StickNode(self, "left foot")
        right_foot = StickNode(self, "right foot")
        sternum = StickNode(self, "sternum")
        left_shoulder = StickNode(self, "left shoulder")
        head = StickNode(self, "head")
        right_shoulder = StickNode(self, "right shoulder")
        left_hand = StickNode(self, "left hand")
        right_hand = StickNode(self, "right hand")

        self.pelvis.SetChild(0,left_knee)
        self.pelvis.SetChild(1, right_knee)
        self.pelvis.SetChild(2, torso)


        left_knee.SetChild(0,left_foot)

        right_knee.SetChild(0, right_foot)

        torso.SetChild(0, sternum)

        sternum.SetChild(0, left_shoulder)
        sternum.SetChild(1, head)
        sternum.SetChild(2, right_shoulder)

        left_shoulder.SetChild(0, left_hand)

        right_shoulder.SetChild(0, right_hand)

    def setup_stick_figure(self, StickNode = None):
        if StickNode is None:
            StickNode = self.pelvis
        self.set_node_parameters(StickNode)
        for child in StickNode.mChild:
            self.setup_stick_figure(child)

    def set_node_parameters(self, StickNode):
        StickNode.set_stick_figure(self)
        if self.input == "CC":
            StickNode.set_theta_from_CC()
        elif self.input == "angle":
            StickNode.set_CC_from_theta()
        else:
            print("Invalid input.")
        if StickNode.name == "pelvis":
            self.angles[StickNode.ID:StickNode.ID + 2] = StickNode.CC
        else:
            self.angles[StickNode.ID+1] = StickNode.theta
        self.CCs[StickNode.ID, :] = StickNode.CC
        self.lengths[StickNode.ID] = StickNode.length

    def get_pelvis_CC(self):
        if self.stick_dict["pelvis"]["CC"] is None:
            return np.zeros(2)
        else:
            return self.stick_dict["pelvis"]["CC"]


    def get_generic_lengths(self, name):
        lengths_dict = {
            "pelvis": 0,
            "left knee": 50,
            "right knee": 50,
            "torso": 32,
            "left foot": 50,
            "right foot": 50,
            "sternum": 32,
            "left shoulder": 40,
            "head": 45,
            "right shoulder": 40,
            "left hand": 40,
            "right hand": 40
        }
        return lengths_dict[name]

    def get_node(self,name,StickNode=None,node=None):
        if StickNode is None:
            StickNode = self.pelvis
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


    '''
    def print_all_nodes(self,StickNode=None):
        """
        prints the attributes of all nodes, from bottom up
        :param StickNode: 
        """
        if StickNode is None:
            StickNode = self.pelvis
        for child in StickNode.mChild[::-1]:
            StickNode.print_all_nodes(child)
        StickNode.print_attributes()
    '''




