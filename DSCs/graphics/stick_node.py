from ..graphics.NodeStruct import NodeStruct
import numpy as np
class StickNode(NodeStruct):
    def __init__(self, StickFigure, name):
        super(StickNode, self).__init__()
        self.name = name
        self.stick_figure = StickFigure
        self.ID = self.stick_figure.num_nodes
        self.stick_figure.num_nodes += 1

        self.CC = None
        self.CC_centered = None
        self.length = None
        self.quadrant = None
        self.theta = None  # angle in polar coordinates

        self.set_dict_input()

    def set_dict_input(self):
        for key,value in self.stick_figure.stick_dict[self.name].items():
            if key == 'CC':
                self.CC = value
            if key == 'angle':
                self.theta = value

    def set_CC_from_theta(self):
        self.length = self.stick_figure.get_generic_lengths(self.name)
        if self.mParent is None:
            self.CC_centered = np.zeros(2)
            self.CC = self.stick_figure.get_pelvis_CC()
        else:
            x = np.cos(self.theta)*self.length
            y = np.sin(self.theta)*self.length
            self.CC_centered = np.array([x, y]).flatten()
            self.CC = self.CC_centered + self.mParent.CC



    def set_CC_centered(self):
        if self.mParent is None:
            self.CC_centered = np.zeros(self.CC.shape)
        else:
            self.CC_centered = self.CC - self.mParent.CC

    def set_length(self):
        self.length = np.sqrt(np.sum((self.CC_centered)**2))

    def set_quadrant(self):
        if self.mParent is None:
            return
        if self.CC_centered[0] >= 0 and self.CC_centered[1] >= 0:
            self.quadrant = 1
        elif self.CC_centered[0] < 0 and self.CC_centered[1] >= 0:
            self.quadrant = 2
        elif self.CC_centered[0] < 0 and self.CC_centered[1] < 0:
            self.quadrant = 3
        elif self.CC_centered[0] >= 0 and self.CC_centered[1] < 0:
            self.quadrant = 4

    def set_theta_from_CC(self):
        self.set_CC_centered()
        self.set_length()
        self.set_quadrant()
        if self.mParent is None:
            return
        elif self.CC_centered[0] == 0:
            theta_prime = np.pi/2
        else:
            theta_prime = np.arctan(np.abs(self.CC_centered[1]/self.CC_centered[0]))
        if self.quadrant == 1:
            self.theta = theta_prime
        elif self.quadrant == 2:
            self.theta = np.pi-theta_prime
        elif self.quadrant == 3:
            self.theta = np.pi+theta_prime
        elif self.quadrant == 4:
            self.theta = 2*np.pi-theta_prime

    def set_theta_from_dict(self):
        pass

    def set_stick_figure(self, StickFigure):
        self.stick_figure = StickFigure

    def print_attributes(self):
        print(
            "Name: " + self.name + "\n"
            "ID: ", self.ID, "\n"
            "Stick figure: ", self.stick_figure, "\n"                                       
            "CC: ", self.CC, "\n"
            "CC centered: ", self.CC_centered , "\n"
            "Length: " + str(self.length) + "\n"
            "quadrant: " + str(self.quadrant) + "\n"
            "theta: " + str(self.theta) + "\n"
        )
