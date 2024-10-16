import numpy as np
from GPy.core.parameterization import Parameterized
from GPy.core import Param

class geometry_base(Parameterized):
    def __init__(self, BC, name="geometry_base"):
        super(geometry_base, self).__init__(name=name)
        self.BC = BC

    def map(self, params, Y = None, phases = None):
        seq_len = self.BC.GPNode.seq_eps[0] - self.BC.GPNode.seq_x0s[0] + 1
        if phases is None:
            Y_KE = 1 / np.sum((.5 * Y ** 2), 1)
            if Y.shape[0] < seq_len:
                ratio = 2 * np.pi / np.sum(Y_KE)
                phases = np.cumsum(Y_KE * ratio)
            else:
                seq_x0s = np.arange(0, Y.shape[0], seq_len)
                seq_eps = np.arange(seq_len - 1, Y.shape[0], seq_len)
                phases_list = []
                for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):
                    ratio = 2 * np.pi / np.sum(Y_KE[x0_i:ep_i + 1])
                    phases = np.cumsum(Y_KE[x0_i:ep_i + 1] * ratio)
                    phases_list.append(phases)
                phases = np.hstack(phases_list).reshape([-1, 1])

        geometry = self.parametric_curve(phases, params)
        return geometry, Y, phases


    def parametric_curve(self, phases, params):
        raise NotImplementedError

    def link_params(self):
        self.parameters = None

    def get_initial_params(self):
        return [None]

    def get_params(self):
        return [None]

    def update_gradients(self, dL_dX, phases):
        pass



class sinusoidal_base(geometry_base):
    def __init__(self,BC, R, r, n, name='sinusoid'):
        super(sinusoidal_base, self).__init__(BC, name='sinusoid')
        self.set_params(R,r,n)

    def set_params(self, R=1.618, r=1, n=.8):
        self.R_init, self.R = R, Param('R', R)
        self.r_init, self.r = r, Param('r', r)
        self.n_init, self.n = n, Param('n', n)

    def get_params(self):
        return [self.R, self.r, self.n]

    def get_initial_params(self):
        return [self.R_init, self.r_init, self.n_init]

    def link_params(self):
        self.link_parameter(self.R)
        self.link_parameter(self.r)
        self.link_parameter(self.n)

    def update_gradients(self, dL_dX, phases):
        self.R.gradient = np.sum(dL_dX * self.dX_dR(phases))
        self.r.gradient = np.sum(dL_dX * self.dX_dr(phases))
        self.n.gradient = np.sum(dL_dX * self.dX_dn(phases))

class ellipse(sinusoidal_base):
    def __init__(self,BC,R=1, r=1, n=1):
        super(ellipse, self).__init__(BC, R, r, n, name='ellipse')

    def parametric_curve(self, phases, params):
        R = params[0]
        r = params[1]
        n = params[2]
        d1 = (R * np.cos(n * phases)).reshape(-1, 1)
        d2 = (r * np.sin(n * phases)).reshape(-1, 1)
        return np.hstack([d1, d2])

    def dX_dR(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dR(phases)
        tmp[:, 1] = self.d2_dR(phases)
        return tmp

    def dX_dr(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dr(phases)
        tmp[:, 1] = self.d2_dr(phases)
        return tmp

    def dX_dn(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dn(phases)
        tmp[:, 1] = self.d2_dn(phases)
        return tmp

    def d1_dR(self, phases):
        return np.cos(self.n*phases).flatten()

    def d2_dR(self, phases):
        return np.zeros(phases.shape).flatten()

    def d1_dr(self, phases):
        return np.zeros(phases.shape).flatten()

    def d2_dr(self, phases):
        return np.sin(self.n*phases).flatten()

    def d1_dn(self, phases):
        return (-self.n*self.R*np.sin(self.n*phases)).flatten()

    def d2_dn(self, phases):
        return (self.n*self.r*np.cos(self.n*phases)).flatten()

class fourier(sinusoidal_base):

    def __init__(self, BC, name='fourier'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        """
            Generate a Fourier basis set.

            Parameters:
            - num_waves (int): Total number of waves (including the constant vector if odd).
            - num_points (int): Number of points to generate for each wave.

            Returns:
            - fourier_basis (numpy array): An array where each column is a basis function.
            """
        num_points = phases.shape[0]
        num_waves = params
        # Time array from 0 to 2pi
        t = np.linspace(0, 2 * np.pi, num_points)

        # Initialize the basis array
        if num_waves % 2 == 1:
            # Include the constant vector if num_waves is odd
            fourier_basis = np.ones((num_points, num_waves))
            num_harmonics = (num_waves - 1) // 2
        else:
            # Exclude the constant vector if num_waves is even
            fourier_basis = np.zeros((num_points, num_waves))
            num_harmonics = num_waves // 2

        # Fill the basis with sine and cosine functions
        for i in range(1, num_harmonics + 1):
            sine_index = 2 * i - 2 if num_waves % 2 == 0 else 2 * i - 1
            cosine_index = 2 * i - 1 if num_waves % 2 == 0 else 2 * i
            fourier_basis[:, sine_index] = np.sin(i * t)  # Sine functions
            fourier_basis[:, cosine_index] = np.cos(i * t)  # Cosine functions

        return fourier_basis

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']