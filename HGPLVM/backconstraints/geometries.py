import numpy as np
from GPy.core.parameterization import Parameterized
from GPy.core import Param
from GPy.core.mapping import Mapping
from scipy.interpolate import interp1d
from scipy import special
from scipy.signal import wavelets
import time


import math
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

class none(geometry_base):

    def __init__(self, BC, name='linear'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        return np.empty((phases.shape[0], 0))

class linear(geometry_base):

    def __init__(self, BC, name='linear'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        d1 = np.linspace(-1, 1, phases.shape[0]).reshape(-1,1)
        d2 = np.linspace(-1, 1, phases.shape[0]).reshape(-1,1)
        d3 = np.linspace(-1, 1, phases.shape[0]).reshape(-1,1)
        d4 = np.linspace(-1, 1, phases.shape[0]).reshape(-1, 1)
        return np.hstack([d1,d2,d3])

class fourier(geometry_base):
    def __init__(self, BC, num_waves=3, name='fourier'):
        super().__init__(BC, name=name)
        self.num_waves = num_waves
        self.set_params()

    def set_params(self):
        # Amplitude parameters
        self.amplitudes = [Param(f'alpha_{i}', 1.0) for i in range(self.num_waves)]
        # Frequency parameters (start with standard frequencies)
        self.frequencies = [Param(f'f_{i}', i//2 + 1) for i in range(self.num_waves)]
        # Phase shift parameters (start with 0)
        self.phase_shifts = [Param(f'phi_{i}', 0.0) for i in range(self.num_waves)]

    def get_params(self):
        return self.amplitudes + self.frequencies + self.phase_shifts

    def get_initial_params(self):
        initial_amplitudes = [1.0 for _ in range(self.num_waves)]
        initial_frequencies = [i//2 + 1 for i in range(self.num_waves)]
        initial_phase_shifts = [0.0 for _ in range(self.num_waves)]
        return initial_amplitudes + initial_frequencies + initial_phase_shifts

    def link_params(self):
        #for param in self.get_params():
        #    self.link_parameter(param)
        pass
        #self.link_parameter(self.amplitudes)
        #self.link_parameter(self.frequencies)
        #self.link_parameter(self.phase_shifts)   

    def parametric_curve(self, phases, params):
        num_points = phases.shape[0]
        t = np.linspace(0, 2 * np.pi, num_points)

        amplitudes = params[:self.num_waves]
        frequencies =  params[self.num_waves:2 * self.num_waves]
        phase_shifts = params[2 * self.num_waves:]
        basis = np.ones((num_points, self.num_waves))
        basis[:, 0] *= amplitudes[0]

        for i in range(0, self.num_waves):
            if i % 2 == 1:
                basis[:, i] = amplitudes[i] * np.sin(frequencies[i] * phases.flatten() + phase_shifts[i])
            else:
                basis[:, i] = amplitudes[i] * np.cos(frequencies[i] * phases.flatten() + phase_shifts[i])


        return basis

    def update_gradients(self, dL_dX, phases):
        t = phases.flatten()

        # Gradients for amplitudes
        for i, param in enumerate(self.amplitudes):
            '''if i == 0:
                grad = np.ones_like(t)
            el'''
            if i % 2 == 1:
                grad = np.sin(self.frequencies[i].values * t + self.phase_shifts[i].values)
            else:
                grad = np.cos(self.frequencies[i].values * t + self.phase_shifts[i].values)
            param.gradient = np.sum(dL_dX * grad.reshape(-1, 1))

        # Gradients for frequencies
        for i, param in enumerate(self.frequencies):
            if (i+1) % 2 == 1:
                grad = self.amplitudes[i].values * t * np.cos(param.values * t + self.phase_shifts[i].values)
            else:
                grad = -self.amplitudes[i].values * t * np.sin(param.values * t + self.phase_shifts[i].values)
            param.gradient = np.sum(dL_dX * grad.reshape(-1, 1))

        # Gradients for phase shifts
        for i, param in enumerate(self.phase_shifts):
            if (i+1) % 2 == 1:
                grad = self.amplitudes[i].values * np.cos(self.frequencies[i].values * t + param.values)
            else:
                grad = -self.amplitudes[i].values * np.sin(self.frequencies[i].values * t + param.values)
            param.gradient = np.sum(dL_dX * grad.reshape(-1, 1))

class torus(geometry_base):
    def __init__(self,BC, name='torus'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        # Major and minor radii of the torus
        R = 3
        r = 1
        num_points = phases.shape[0]
        # Generate a single, dense, continuous line on the torus
        t = np.linspace(0, 100 * np.pi, num_points)
        theta = t * 7  # Non-integer ratio for theta
        phi = phases.flatten()  # Non-integer ratio for phi

        # Parametric equations of the torus
        d1 = ((R + r * np.cos(phi)) * np.cos(theta)).reshape(-1, 1)
        d2 = ((R + r * np.cos(phi)) * np.sin(theta)).reshape(-1, 1)
        d3 = (r * np.sin(phi)).reshape(-1, 1)
        return np.hstack([d1,d2,d3])

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

class toroid(sinusoidal_base):
    def __init__(self,BC,R=1.618, r=1, n=.8):#'R=1.618, r=1, n=.8'
        super(toroid, self).__init__(BC, R, r, n, name='toroid')

    def parametric_curve(self, phases, params):
        '''
        params = [R, r, n]
        R - outer torus radius
        r - inner torus radius
        n - number of winds around torus
        :param BC: backconstraint
        :return: 3D parametric curve of a toroid
        '''
        R = params[0]
        r = params[1]
        n = params[2]

        d1 = ((R + r * np.cos(n * phases)) * np.cos(phases)).reshape(-1, 1)
        d2 = ((R + r * np.cos(n * phases)) * np.sin(phases)).reshape(-1, 1)
        d3 = (r * np.sin(n * phases)).reshape(-1, 1)
        return np.hstack([d1,d2,d3])


    def dX_dR(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dR(phases)
        tmp[:, 1] = self.d2_dR(phases)
        # tmp[:, 2] = d3_dR(phases) #zero
        return tmp

    def dX_dr(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dr(phases)
        tmp[:, 1] = self.d2_dr(phases)
        tmp[:, 2] = self.d3_dr(phases)
        return tmp

    def dX_dn(self, phases):
        tmp = np.zeros(self.BC.X.shape)
        tmp[:, 0] = self.d1_dn(phases)
        tmp[:, 1] = self.d2_dn(phases)
        tmp[:, 2] = self.d3_dn(phases)
        return tmp

    def d1_dR(self, phases):
        return np.cos(phases).flatten()

    def d2_dR(self, phases):
        return np.sin(phases).flatten()

    def d3_dR(self, phases):
        return 0

    def d1_dr(self, phases):
        return (np.cos(self.n * phases) * np.cos(phases)).flatten()

    def d2_dr(self, phases):
        return (np.cos(self.n * phases) * np.sin(phases)).flatten()

    def d3_dr(self, phases):
        return np.sin(self.n * phases).flatten()

    def d1_dn(self, phases):
        return (-self.r * self.n * np.sin(self.n * phases) * np.cos(phases)).flatten()

    def d2_dn(self, phases):
        return (-self.r * self.n * np.sin(self.n * phases) * np.sin(phases)).flatten()

    def d3_dn(self, phases):
        return np.cos(self.n * phases).flatten()
def interpolate_sequence(sequence, time_steps, kind='linear'):
    # Convert sequence to a numpy array if not already
    sequence = np.array(sequence)

    # Check if the sequence is at least 2D (if not, make it 2D)
    if sequence.ndim == 1:
        sequence = sequence.reshape(-1, 1)

    # Generate x values corresponding to the indices of the sequence
    x_original = np.arange(sequence.shape[0])

    # Initialize an empty array for the interpolated values
    interpolated_sequence = np.zeros_like(sequence)

    # Iterate over each dimension of the sequence
    for i in range(sequence.shape[1]):
        interpolation_function = interp1d(x_original, sequence[:, i], kind=kind, fill_value="extrapolate")
        interpolated_sequence[:, i] = interpolation_function(time_steps)

    return interpolated_sequence


class sine(geometry_base):

    def __init__(self, BC, name='sine'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        shift = np.pi / 2
        #d2 = np.cos(phases).reshape(-1, 1)
        #d1 = np.sin(phases).reshape(-1, 1)
        '''d2 = 2*np.sin(phases).reshape(-1, 1)
        d3 = .5*np.sin(phases).reshape(-1, 1)
        d4 = np.sin(phases).reshape(-1, 1)'''
        d0 = np.ones(phases.shape[0]).reshape(-1, 1)
        d1 = np.sin(phases).reshape(-1, 1)
        d2 = np.cos(phases).reshape(-1, 1)
        d3 = np.sin(2 * phases).reshape(-1, 1)
        d4 = np.cos(2 * phases).reshape(-1, 1)
        d5 = np.sin(3 * phases).reshape(-1, 1)
        d6 = np.cos(3 * phases).reshape(-1, 1)



        #d4 = np.sin(4*phases).reshape(-1, 1)
        return np.hstack([d0,d1,d2,d3,d4])

class random_sine_waves(geometry_base):
    """
    Similar to your `fourier` class, but uses randomized initialization
    for the amplitude, frequency, and phase shift of each sine wave.
    """
    def __init__(self, BC, num_waves=3, name='random_sine_waves'):
        super().__init__(BC, name=name)
        self.num_waves = num_waves
        self.set_params()

    def set_params(self):
        """
        Randomly initialize amplitude, frequency, and phase shift
        for each wave, similar to random_sine_waves logic.
        """
        seed = int(time.time() * 1000) % (2**32)
        rng = np.random.RandomState(seed)

        self.amplitudes = []
        self.frequencies = []
        self.phase_shifts = []

        for i in range(self.num_waves):
            # Random draws for each wave
            amplitude = rng.uniform(0.1, 2.0)
            frequency = rng.uniform(0.1, 2.0)
            phase_shift = rng.uniform(0, 2 * np.pi)

            self.amplitudes.append(Param(f'A_{i}', amplitude))
            self.frequencies.append(Param(f'f_{i}', frequency))
            self.phase_shifts.append(Param(f'phi_{i}', phase_shift))

    def get_params(self):
        # Return a list of all Param objects (amplitudes, freqs, phases)
        return self.amplitudes + self.frequencies + self.phase_shifts

    def get_initial_params(self):
        # Return just the numerical values of the parameters
        init_amps = [p.values for p in self.amplitudes]
        init_freqs = [p.values for p in self.frequencies]
        init_phases = [p.values for p in self.phase_shifts]
        return init_amps + init_freqs + init_phases


    def link_params(self):
        """
        If you want to link parameters to a particular optimizer or
        constraint system (as in your original `fourier` class).
        """
        print('WARNING: RWS params not linked.')
        #for param in self.get_params():
        #    self.link_parameter(param)

    def parametric_curve(self, phases, params):
        """
        Create a matrix of shape (len(phases), num_waves) whose columns
        are sine waves derived from the parameters.
        
        :param phases: an array of time or phase values (shape: [n_points,])
        :param params: the flattened array of all parameters 
                       [amplitudes, frequencies, phase_shifts].
        :return: a 2D array [n_points, num_waves] of sine values.
        """
        n_points = phases.shape[0]

        # Split the params array
        amplitudes = params[:self.num_waves]
        frequencies = params[self.num_waves:2*self.num_waves]
        phase_shifts = params[2*self.num_waves:3*self.num_waves]

        # Construct the sine waves
        basis = np.zeros((n_points, self.num_waves))
        for i in range(self.num_waves):
            # Sine wave with amplitude_i, frequency_i, phase_shift_i
            # 2*pi factor is optional (depending on your chosen definition).
            basis[:, i] = amplitudes[i] * np.sin(
                2 * np.pi * frequencies[i] * phases.flatten() + phase_shifts[i]
            )

        return basis


class wavelet_basis(geometry_base):

    def __init__(self, BC, name='wavelet_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        x = np.linspace(0, 1, N)
        basis = np.zeros((D, N))
        for i in range(D):
            wavelet = wavelets.ricker(N, i + 1)
            basis[i] = wavelet / np.linalg.norm(wavelet)
        return basis.T
class legendre_basis(geometry_base):

    def __init__(self, BC, name='legendre_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        x = np.linspace(-1, 1, N)
        basis = np.zeros((D, N))
        for i in range(D):
            basis[i] = special.legendre(i)(x)
        return basis.T
class hermite_basis(geometry_base):

    def __init__(self, BC, name='hermite_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        x = np.linspace(-5, 5, N)  # Adjust range as needed
        basis = np.zeros((D, N))
        for i in range(D):
            basis[i] = special.hermite(i)(x)
        return basis.T
class laguerre_basis(geometry_base):

    def __init__(self, BC, name='laguerre_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        x = np.linspace(0, 5, N)  # Adjust range as needed
        basis = np.zeros((D, N))
        for i in range(D):
            basis[i] = special.laguerre(i)(x)
        return basis.T
class chebyshev_basis(geometry_base):

    def __init__(self, BC, name='chebyshev_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        x = np.linspace(-1, 1, N)
        basis = np.zeros((D, N))
        for i in range(D):
            basis[i] = special.chebyt(i)(x)
        return basis.T
class zernike_basis(geometry_base):

    def __init__(self, BC, name='zernike_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def zernike_radial(self, n, m, r):
        """
        Calculate the radial component of Zernike polynomial.
        """
        if (n - m) % 2:
            return r * 0

        R = r * 0
        for k in range((n - m) // 2 + 1):
            coef = (-1) ** k * np.math.factorial(n - k)
            coef /= np.math.factorial(k) * np.math.factorial((n + m) // 2 - k) * np.math.factorial((n - m) // 2 - k)
            R += coef * r ** (n - 2 * k)
        return R

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        grid_size = int(np.ceil(np.sqrt(N)))

        r = np.linspace(0, 1, grid_size)
        theta = np.linspace(0, 2 * np.pi, grid_size)
        r, theta = np.meshgrid(r, theta)

        basis = np.zeros((D, N))
        idx = 0
        for n in range(D):
            for m in range(-n, n + 1, 2):
                if idx < D:
                    Z = self.zernike_radial(n, abs(m), r)
                    if m < 0:
                        Z_full = (Z * np.sin(m * theta)).ravel()
                    else:
                        Z_full = (Z * np.cos(m * theta)).ravel()

                    # Truncate or pad to match N
                    if len(Z_full) > N:
                        basis[idx] = Z_full[:N]
                    else:
                        basis[idx, :len(Z_full)] = Z_full
                        basis[idx, len(Z_full):] = Z_full[-1]  # Pad with the last value

                    idx += 1
        return basis.T
class spherical_harmonics_basis(geometry_base):

    def __init__(self, BC, name='spherical_harmonics_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params
        grid_size = int(np.ceil(np.sqrt(N)))

        theta = np.linspace(0, np.pi, grid_size)
        phi = np.linspace(0, 2 * np.pi, grid_size)
        theta, phi = np.meshgrid(theta, phi)

        basis = np.zeros((D, N))
        idx = 0
        for l in range(int(np.sqrt(D))):
            for m in range(-l, l + 1):
                if idx < D:
                    Y = special.sph_harm(m, l, phi, theta)
                    Y_full = np.real(Y).ravel()

                    # Truncate or pad to match N
                    if len(Y_full) > N:
                        basis[idx] = Y_full[:N]
                    else:
                        basis[idx, :len(Y_full)] = Y_full
                        basis[idx, len(Y_full):] = Y_full[-1]  # Pad with the last value

                    idx += 1
        return basis.T
class haar_basis(geometry_base):

    def __init__(self, BC, name='haar_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params

        def haar_wavelet(x, k, j):
            scale = 2 ** j
            translated_x = scale * x - k
            return np.where((1 <= translated_x) & (translated_x < 1.5), 1,
                            np.where((1.5 <= translated_x) & (translated_x < 2), -1, 0))

        x = np.linspace(0, 1, N)
        basis = np.zeros((D, N))
        basis[0] = 1  # First Haar wavelet is constant
        idx = 1
        j = 0
        while idx < D:
            for k in range(2 ** j):
                if idx < D:
                    basis[idx] = haar_wavelet(x, k, j)
                    idx += 1
            j += 1
        return basis.T
class walsh_basis(geometry_base):

    def __init__(self, BC, name='walsh_basis'):
        super().__init__(BC, name=name)

    def get_params(self):
        return self.BC.param_dict['geo params']

    def get_initial_params(self):
        return self.BC.param_dict['geo params']

    def parametric_curve(self, phases, params):
        N = phases.shape[0]
        D = params

        def walsh(x, i):
            return np.prod([np.sign(np.sin(2 ** k * np.pi * x)) for k in range(i)], axis=0)

        x = np.linspace(0, 1, N)
        basis = np.zeros((D, N))
        for i in range(D):
            basis[i] = walsh(x, i)
        return basis.T



class helix(geometry_base):
    def __init__(self,BC, name='helix'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        num_turns = 5
        theta = np.linspace(0, 2 * np.pi * num_turns, phases.shape[0])
        r = 2  # Radius of the helix
        c = 5  # Pitch of the helix

        d1 = r * np.cos(theta)
        d2 = r * np.sin(theta)
        d3 = c * theta
        return np.hstack([d1,d2,d3])

class sphere(geometry_base):
    def __init__(self,BC, name='sphere'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        R=1
        r=1
        n=1
        phi=np.linspace(0,100*np.pi,phases.shape[0]).reshape(-1,1)
        theta=np.linspace(0,np.pi,phases.shape[0]).reshape(-1,1)
        d1 = ((R + r * np.sin(n * theta)) * np.sin(phi)).reshape(-1, 1)
        d2 = ((R + r * np.sin(n * theta)) * np.cos(phi)).reshape(-1, 1)
        d3 = (r * np.cos(n * theta)).reshape(-1, 1)
        return np.hstack([d1,d2,d3])
class cocentric_circles(geometry_base):
    def __init__(self,BC, name='cocentric_circles'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        # Generate an array of angles

        num_turns = 1
        height = 1
        theta = phases[:,0]#np.linspace(0, 2 * np.pi * num_turns, phases.shape[0])
        # Radius increases with theta
        r = np.linspace(1, 5, phases.shape[0])
        # Calculating the x, y, and z coordinates
        d1 = (r * np.cos(theta)).reshape(-1, 1)
        d2 = (r * np.sin(theta)).reshape(-1, 1)
        # Adding a linear height component to make it 3D
        d3 = np.linspace(0, height, phases.shape[0]).reshape(-1, 1)
        return np.hstack([d1,d2])
    
class mobius_strip(geometry_base):
    def __init__(self,BC, name='mobius_strip'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        # Generate an array of angles
        R = 1
        r = 1
        n = 1

        d1 = ((R*np.cos(phases) + r * np.cos(n * phases/2)) * np.cos(phases)).reshape(-1, 1)
        d2 = ((R*np.sin(phases) + r * np.cos(n * phases/2)) * np.sin(phases)).reshape(-1, 1)
        d3 = (r * np.sin(n * phases/2)).reshape(-1, 1)
        return np.hstack([d1,d2,d3])

class klein_bottle(geometry_base):
    def __init__(self,BC, name='klein_bottle'):
        super().__init__(BC, name=name)

    def parametric_curve(self, phases, params):
        num_points = phases.shape[0]#math.ceil(np.sqrt(phases.shape[0]))
        phi = phases
        #phi = 100*phases#np.linspace(0, 2 * np.pi, num_points)
        #theta, phi = np.meshgrid(theta, phi)
        #theta = np.linspace(0, 2 * np.pi, phases.shape[0])
        theta = phases#np.linspace(0, 4 * np.pi, phases.shape[0])

        r = 1  # Radius parameter, adjust as needed

        x = (r + np.cos(theta / 2) * np.sin(phi / 2) - np.sin(theta / 2) * np.sin(phi)) * np.cos(theta)
        y = (r + np.cos(theta / 2) * np.sin(phi / 2) - np.sin(theta / 2) * np.sin(phi)) * np.sin(theta)
        z = np.sin(theta / 2) * np.sin(phi / 2) + np.cos(theta / 2) * np.sin(phi)

        d1 = x.flatten()[:phases.shape[0]].reshape(-1, 1)
        d2 = y.flatten()[:phases.shape[0]].reshape(-1, 1)
        d3 = z.flatten()[:phases.shape[0]].reshape(-1, 1)
        return np.hstack([d1,d2,d3])


class EP_hands_geometry(geometry_base):
    def __init__(self, BC,  name='EP'):
        self.MH = None
        super().__init__(BC, name=name)

    def map(self, params, Y = None, phases = None):
        if self.MH is None:
            seq_len = self.BC.GPNode.seq_eps[0] - self.BC.GPNode.seq_x0s[0] + 1
            seq_x0s = np.arange(0, Y.shape[0], seq_len)
            seq_eps = np.arange(seq_len - 1, Y.shape[0], seq_len)
            MH_list = []
            for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):
                LH, RH, MH = self.hand_EPs(
                    self.BC.param_dict['data_set_class'].graphic.HGPLVM_angles_PV_to_stick_dicts_CCs(Y[x0_i:ep_i + 1]))

                MH -= np.mean(MH, axis=0)
                MH /= np.std(MH, axis=0)

                """MH_v = MH[1:, :] - MH[:-1, :]
                MH_v_inv = np.sum(1/(np.abs(MH_v)+.1), axis=1)
                MH_v_inv /= np.max(MH_v_inv)
                cumsum_MH = np.cumsum(MH_v_inv)
                cumsum_MH /= np.max(cumsum_MH)
                cumsum_MH *= seq_len

                CS_list = list(cumsum_MH)
                CS_list.insert(0,0)
                CS_arr = np.array(CS_list)
                #CS_arr = np.arange(0,seq_len,1)
                MH_interp = interpolate_sequence(MH, CS_arr, kind='linear')"""
                MH_list.append(MH)
            self.MH = np.vstack(MH_list)


        return self.MH, Y, phases

    def set_params(self, R=1.618, r=1, n=.8):
        self.R_init, self.R = R, Param('R', R)
        self.r_init, self.r = r, Param('r', r)
        self.n_init, self.n = n, Param('n', n)

    def get_params(self):
        return None

    def get_initial_params(self):
        return None

    def link_params(self):
        self.parameters = None

    def update_gradients(self, dL_dX, phases):
        pass

    def hand_EPs(self, Y_SD):
        LH = Y_SD['LeftHand']['CC']
        RH = Y_SD['RightHand']['CC']
        MH = (LH + RH) / 2
        return LH, RH, MH

class EP_hands_toroid_geometry(toroid):
    def __init__(self, BC):
        self.MH = None
        super().__init__(BC)

    def map(self, params, Y=None, phases=None):
        seq_len = self.BC.GPNode.seq_eps[0] - self.BC.GPNode.seq_x0s[0] + 1
        if Y is not None:
            seq_x0s = np.arange(0, Y.shape[0], seq_len)
            seq_eps = np.arange(seq_len - 1, Y.shape[0], seq_len)
        if self.MH is None:
            MH_list = []
            for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):
                LH, RH, MH = self.hand_EPs(
                    self.BC.param_dict['data_set_class'].graphic.HGPLVM_angles_PV_to_stick_dicts_CCs(
                        Y[x0_i:ep_i + 1]))

                MH -= np.mean(MH, axis=0)
                MH /= np.std(MH, axis=0)
                MH_list.append(MH)
            self.MH = np.vstack(MH_list)
            """
            from sklearn.decomposition import KernelPCA
            transformer = KernelPCA(n_components=self.MH.shape[1], kernel='rbf')
            self.MH = transformer.fit_transform(self.MH)"""

        if phases is None:
            Y_KE = 1 / np.sum((.5 * Y ** 2), 1)
            if Y.shape[0] < seq_len:
                ratio = 2 * np.pi / np.sum(Y_KE)
                phases = np.cumsum(Y_KE * ratio)
            else:
                phases_list = []
                for i, (x0_i, ep_i) in enumerate(zip(seq_x0s, seq_eps)):
                    ratio = 2 * np.pi / np.sum(Y_KE[x0_i:ep_i + 1])
                    phases = np.cumsum(Y_KE[x0_i:ep_i + 1] * ratio)
                    phases_list.append(phases)
                phases = np.hstack(phases_list).reshape([-1, 1])

        geometry = np.hstack([self.parametric_curve(phases, params),self.MH])
        # geometry -= geometry.mean(0)
        # geometry /= geometry.std(0)
        return geometry, Y, phases



    def hand_EPs(self, Y_SD):
        LH = Y_SD['LeftHand']['CC']
        RH = Y_SD['RightHand']['CC']
        MH = (LH + RH) / 2
        return LH, RH, MH