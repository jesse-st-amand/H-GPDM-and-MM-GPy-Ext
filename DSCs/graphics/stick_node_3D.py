from ..graphics.NodeStruct import NodeStruct
import numpy as np
class StickNode(NodeStruct):
    def __init__(self, StickFigure, name):
        super(StickNode, self).__init__()
        self.name = name
        self.stick_figure = StickFigure
        self.ID = self.stick_figure.num_nodes
        self.stick_figure.num_nodes += 1

        self.CCs = None
        self.CCs_centered = None
        self.length = None
        self.quadrants = [None,None]
        self.thetas = [None,None]  # angle in polar coordinates

        self.set_dict_input()

    def set_dict_input(self):
        for key,value in self.stick_figure.stick_dict[self.name].items():
            if key == 'CC':
                self.CCs = value
            if key == 'angle':
                self.thetas = value

    def set_CCs_from_thetas(self):
        self.set_length()
        if self.mParent is None:
            self.CCs_centered = np.zeros(3)
            self.CCs = self.stick_figure.get_pelvis_CC()
        else:
            x = np.cos(self.thetas[0]) * np.sin(self.thetas[1]) * self.length
            y = np.sin(self.thetas[0]) * np.sin(self.thetas[1]) * self.length
            z = np.cos(self.thetas[1]) * self.length
            self.CCs_centered = np.array([x, y, z]).flatten()
            self.CCs = self.CCs_centered + self.mParent.CCs

    def set_CCs_centered(self):
        if self.mParent is None:
            self.CCs_centered = np.zeros(self.CCs.shape)
        else:
            self.CCs_centered = self.CCs - self.mParent.CCs

    def set_length(self):
        #self.length = np.sqrt(np.sum((self.CCs_centered)**2))
        self.length = self.stick_figure.get_generic_length(self.name)

    def set_quadrants(self):
        if self.mParent is None:
            return
        self.quadrants[0] = self.get_quadrants_from_dims(0,1)
        #self.quadrants[1] = self.get_quadrants_from_dims(2,1)
            
    def get_quadrants_from_dims(self,d1,d2):
        if self.CCs_centered[d1] >= 0 and self.CCs_centered[d2] >= 0:
            return 1
        elif self.CCs_centered[d1] < 0 and self.CCs_centered[d2] >= 0:
            return 2
        elif self.CCs_centered[d1] < 0 and self.CCs_centered[d2] < 0:
            return 3
        elif self.CCs_centered[d1] >= 0 and self.CCs_centered[d2] < 0:
            return 4

    def set_thetas_from_CCs(self):
        self.set_CCs_centered()
        self.set_length()
        self.set_quadrants()
        if self.mParent is None:
            return

        self.get_thetas_from_dims(0,0,1)
        self.thetas[1] = np.arctan2(
            np.abs(np.sqrt(self.CCs_centered[0] ** 2 + self.CCs_centered[1] ** 2)), self.CCs_centered[2])
    def get_thetas_from_dims(self,theta_i,d1,d2):
        if self.CCs_centered[d1] == 0:
            theta_prime = np.pi/2
        else:
            theta_prime = np.arctan(np.abs(self.CCs_centered[d2]/self.CCs_centered[d1]))
        if self.quadrants[theta_i] == 1:
            self.thetas[theta_i] = theta_prime
        elif self.quadrants[theta_i] == 2:
            self.thetas[theta_i] = np.pi-theta_prime
        elif self.quadrants[theta_i] == 3:
            self.thetas[theta_i] = np.pi+theta_prime
        elif self.quadrants[theta_i] == 4:
            self.thetas[theta_i] = 2*np.pi-theta_prime

    def set_theta_from_dict(self):
        pass

    def set_stick_figure(self, StickFigure):
        self.stick_figure = StickFigure

    def print_attributes(self):
        print(
            "Name: " + self.name + "\n"
            "ID: ", self.ID, "\n"
            "Stick figure: ", self.stick_figure, "\n"                                       
            "CC: ", self.CCs, "\n"
            "CC centered: ", self.CCs_centered , "\n"
            "Length: " + str(self.length) + "\n"
            "quadrant: " + str(self.quadrants) + "\n"
            "theta: " + str(self.thetas) + "\n"
        )
