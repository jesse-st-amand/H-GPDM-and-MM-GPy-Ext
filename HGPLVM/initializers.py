import numpy as np
from GPy.core.parameterization import Parameterized
from GPy.core import Param
from GPy.core.mapping import Mapping
from scipy.interpolate import interp1d
from scipy import special
from scipy.signal import wavelets
from functools import wraps
import time


def basis_per_seq(func,N,D,num_seqs):
    """
        Generate a Fourier basis set.

        Parameters:
        - D (int): Total number of waves (including the constant vector if odd).
        - N (int): Number of points to generate for each wave.

        Returns:
        - fourier_basis (numpy array): An array where each column is a basis function.
        """
    seq_len = int(N / num_seqs)
    basis = func(seq_len, D)
    return np.tile(basis, (num_seqs, 1))

def fourier_basis(N, D, per_seq = False):
    """
        Generate a Fourier basis set.

        Parameters:
        - D (int): Total number of waves (including the constant vector if odd).
        - N (int): Number of points to generate for each wave.

        Returns:
        - fourier_basis (numpy array): An array where each column is a basis function.
        """
    # Time array from 0 to 2pi
    t = np.linspace(0, 2 * np.pi, N)

    # Initialize the basis array
    if D % 2 == 1:
        # Include the constant vector if D is odd
        fourier_basis = np.ones((N, D))
        num_harmonics = (D - 1) // 2
    else:
        # Exclude the constant vector if D is even
        fourier_basis = np.zeros((N, D))
        num_harmonics = D // 2

    # Fill the basis with sine and cosine functions
    for i in range(1, num_harmonics + 1):
        print(i)
        sine_index = 2 * i - 2 if D % 2 == 0 else 2 * i - 1
        cosine_index = 2 * i - 1 if D % 2 == 0 else 2 * i
        fourier_basis[:, sine_index] = np.sin(i * t)  # Sine functions
        fourier_basis[:, cosine_index] = np.cos(i * t)  # Cosine functions

    return fourier_basis

def wavelet_basis(N, D):
    basis = np.zeros((D, N))
    for i in range(D):
        wavelet = wavelets.ricker(N, i + 1)
        basis[i] = wavelet / np.linalg.norm(wavelet)
    return basis.T
def legendre_basis(N, D):
    x = np.linspace(-1, 1, N)
    basis = np.zeros((D, N))
    for i in range(D):
        basis[i] = special.legendre(i)(x)
    return basis.T
def hermite_basis(N, D):
    x = np.linspace(-5, 5, N)  # Adjust range as needed
    basis = np.zeros((D, N))
    for i in range(D):
        basis[i] = special.hermite(i)(x)
    return basis.T
def laguerre_basis(N, D):
    x = np.linspace(0, 5, N)  # Adjust range as needed
    basis = np.zeros((D, N))
    for i in range(D):
        basis[i] = special.laguerre(i)(x)
    return basis.T
def chebyshev_basis(N, D):
    x = np.linspace(-1, 1, N)
    basis = np.zeros((D, N))
    for i in range(D):
        basis[i] = special.chebyt(i)(x)
    return basis.T

def zernike_radial(n, m, r):
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

def zernike_basis(N, D):
    grid_size = int(np.ceil(np.sqrt(N)))
    r = np.linspace(0, 1, grid_size)
    theta = np.linspace(0, 2 * np.pi, grid_size)
    r, theta = np.meshgrid(r, theta)
    basis = np.zeros((D, N))
    idx = 0
    for n in range(D):
        for m in range(-n, n + 1, 2):
            if idx < D:
                Z = zernike_radial(n, abs(m), r)
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

def spherical_harmonics_basis(N, D):
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

def haar_basis(N, D):
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

def walsh_basis(N, D):
    def walsh(x, i):
        return np.prod([np.sign(np.sin(2 ** k * np.pi * x)) for k in range(i)], axis=0)
    x = np.linspace(0, 1, N)
    basis = np.zeros((D, N))
    for i in range(D):
        basis[i] = walsh(x, i)
    return basis.T


def random_sine_waves(N, D):
    # Generate a seed based on current time
    seed = int(time.time() * 1000) % (2**32)
    print(f"Seed used: {seed}")

    # Create a RandomState object with this seed
    rng = np.random.RandomState(seed)

    # Generate time points
    t = np.linspace(0, 2*np.pi, N)

    # Initialize the output array
    output = np.zeros((N, D))

    for d in range(D):
        # Randomize parameters for each sine wave using our RandomState object
        amplitude = rng.uniform(0.1, 2.0)
        frequency = rng.uniform(0.1, 2.0)
        phase_shift = rng.uniform(0, 2 * np.pi)

        # Generate the sine wave
        output[:, d] = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)

    return output

def sine_waves(N, D):
    # Generate a seed based on current time
    seed = int(time.time() * 1000) % (2**32)
    print(f"Seed used: {seed}")

    # Create a RandomState object with this seed
    rng = np.random.RandomState(seed)

    # Generate time points
    t = np.linspace(0, 2*np.pi, N)

    # Initialize the output array
    output = np.zeros((N, D))

    for d in range(D):
        # Randomize parameters for each sine wave using our RandomState object
        amplitude = rng.uniform(0.5, 2.0)
        frequency = 1#rng.uniform(0.1, 2.0)
        phase_shift = 1#rng.uniform(0, 2 * np.pi)

        # Generate the sine wave
        output[:, d] = amplitude * np.sin(2 * np.pi * frequency * t + phase_shift)

    return output


def lines(N, D):
    seed = int(time.time() * 1000) % (2 ** 32)
    print(f"Seed used: {seed}")

    # Create a RandomState object with this seed
    rng = np.random.RandomState(seed)
    # Generate time points
    t = np.linspace(0, 1, N)

    # Initialize the output array
    output = np.zeros((N, D))

    for d in range(D):
        # Randomize start and end points for each line
        start_point = rng.uniform(-1, 1)
        end_point = rng.uniform(-1, 1)

        # Generate the line
        output[:, d] = start_point + (end_point - start_point) * t

    return output

def per_seq_decorator(func):
    @wraps(func)
    def wrapper(N, D, per_seq='', num_seqs=1):
        if per_seq.lower() == 'per_seq':
            per_seq = True
        if per_seq:
            seq_len = int(N / num_seqs)
            basis = func(seq_len, D)
            return np.tile(basis, (num_seqs, 1))
        else:
            return func(N, D)
    return wrapper


# Apply the decorator to all basis functions
fourier_basis = per_seq_decorator(fourier_basis)
wavelet_basis = per_seq_decorator(wavelet_basis)
legendre_basis = per_seq_decorator(legendre_basis)
hermite_basis = per_seq_decorator(hermite_basis)
laguerre_basis = per_seq_decorator(laguerre_basis)
chebyshev_basis = per_seq_decorator(chebyshev_basis)
zernike_basis = per_seq_decorator(zernike_basis)
spherical_harmonics_basis = per_seq_decorator(spherical_harmonics_basis)
haar_basis = per_seq_decorator(haar_basis)
walsh_basis = per_seq_decorator(walsh_basis)

def FFT_2D(Y, D, num_seqs):
    # Perform FFT along the sequence dimension
    Y_fft = np.fft.fft(Y, axis=1)

    # Compute the magnitude spectrum (real-valued)
    Y_magnitude = np.abs(Y_fft)

    # Compute the energy of each frequency across all sequences
    energy = np.sum(Y_magnitude ** 2, axis=0)

    # Find the indices of the top input_dim energetic dimensions
    top_dim_indices = np.argsort(energy)[-D:]

    # Select only the top input_dim dimensions
    X_magnitude_reduced = Y_magnitude[:, top_dim_indices]

    # Reshape the result back to (N, input_dim)
    return X_magnitude_reduced

def FFT_3D(Y, D, num_seqs):
    N = Y.shape[0]
    n = int(N / num_seqs)

    # Reshape Y into (num_sequences, n, D)
    Y_reshaped = Y.reshape(num_seqs, n, Y.shape[1])

    # Perform FFT along the sequence dimension
    Y_fft = np.fft.fft(Y_reshaped, axis=1)

    # Compute the magnitude spectrum (real-valued)
    Y_magnitude = np.abs(Y_fft)

    # Compute the energy of each frequency across all sequences
    energy = np.sum(Y_magnitude ** 2, axis=(0, 1))

    # Find the indices of the top input_dim energetic dimensions
    top_dim_indices = np.argsort(energy)[-D:]

    # Select only the top input_dim dimensions
    X_magnitude_reduced = Y_magnitude[:, :, top_dim_indices]

    # Reshape the result back to (N, input_dim)
    return X_magnitude_reduced.reshape(N, D)

def random_projections(Y,D):
    seed = int(time.time() * 1000) % (2**32)
    print(f"Seed used: {seed}")

    # Create a RandomState object with this seed
    rng = np.random.RandomState(seed)
    _, n_features = Y.shape
    R = rng.randn(n_features, D)
    return np.dot(Y, R)
##


##

