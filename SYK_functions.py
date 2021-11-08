# Create Majorana Fermions and SYK Hamiltonian

import numpy as np
import numexpr as ne
from functools import reduce
from scipy.fft import fftfreq
from pyfftw.interfaces.numpy_fft import fft, ifft

def create_majorana_fermions(N):
    """Create Majorana Fermions.

    Creates Majorana Fermions - a set of N Hermitian matrices psi_i, i=1,..N
    obeying anti-commutation relations {psi_i,psi_j} = δ_{ij}

    Args:
        N: An integer denoting number of Majorana fermions.

    Returns:
        A dictionary containing matrix representations of Majorana fermions.
    """
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.array([[1, 0], [0, 1]])

    psi = dict()

    for i in range(1, N+1):
        if (i % 2) == 1:
            matlist = [Z] * int((i-1)/2)
            matlist.append(X)
            matlist = matlist + [I] * int((N/2 - (i+1)/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)
        else:
            matlist = [Z] * int((i - 2) / 2)
            matlist.append(Y)
            matlist = matlist + [I] * int((N/2 - i/2))
            psi[i] = 1/np.sqrt(2)*reduce(np.kron, matlist)
    return psi


def create_Hamiltonian(psi, N, realisations, J_squared=1):
    """Create Majorana Fermions.

    Creates multiple realisations of the SYK Hamiltonian

    Args:
        psi: A dictionary containing matrix representations of Majorana fermions.
        N: An integer denoting number of Majorana fermions.
        realisations: An integer denoting the number of realisations of the Hamiltonian to be created
        J_squared: Variance of couplings is given by 'J_squared * 3!/N^3'. Set to 1 by default.

    Returns:
        An array containing an SYK Hamiltonian for each realisation of the model.
    """

    H = 0
    J = dict()
    for i in range(1, N+1):
        for j in range(i+1, N+1):
            for k in range(j+1, N+1):
                for l in range(k+1, N+1):
                    # print([i, j, k, l])
                    J[i, j, k, l] = np.random.normal(loc=0, scale=np.sqrt(J_squared*np.math.factorial(3))/(N**(3/2)),
                                                     size=realisations)
                    M = psi[i] @ psi[j] @ psi[k] @ psi[l]
                    H = H + np.array([element * M for element in J[i, j, k, l]])
    return H


def G_SD(t0, dt, G_input, q, q_t, s_squared, iteration_length):
    """Compute large N two point function from Schwinger Dyson equations.

    Solves the Schwinger Dyson equations numerically with an iterative algorithm and using the FFT
    to switch between coordinate and Fourier space.

    Args:
        t0: A float. -t0 is the inverse temperature β of the model and sets the limits of the time interval
        on which G(t) is evaluated.
        dt: A float that sets the time step.
        G_input: A function used as the initial guess for G.
        q: integer ≥1. Defines number of fermions involved in random interactions in 1st Hamiltonion of modified SYK model.
        q_t: integer ≥1. Defines number of fermions involved in random interactions in 2nd (with s) Hamiltonian of modified SYK model.
        s_squared: Coupling between two SYK models.
        iteration_length: Integer denoting many iterations to carry out.

    Returns:
        t: Array containing time points on which functions are evaluated on.
        G: Array containing large N two point function.
        w: Array containing points in frequency space.
        S: Array containing sigma field.
        Sf: Array containing fourier transform of sigma field.
    """

    t = np.arange(t0, -t0, dt)  # define time points on which G(t) is evaluated

    # initialize G and sigma fields
    G = G_input(t)
    S = ne.evaluate("(s_squared * 2/q_t * 1/(2**(3-2*q_t)) * (G ** (2*q_t - 1))) + (2/q * (1/(2**(3-2*q))) * (G ** (2*q - 1)))")

    # frequency normalization factor is 2*np.pi/dt
    # convention of sign in exponential of definition of Fourier
    # transform In order to get a discretisation of the continuous Fourier
    # transform we need to multiply g by a phase factor
    w = ne.evaluate("-w * 2 * pi / dt", {
        "w": fftfreq(G.size),
        "pi": np.pi,
        "dt": dt,
        })

    phase = ne.evaluate("0.5 * dt * exp((-1j * t0) * w)")

    Gf = ne.evaluate("G_fft * phase", {
        "G_fft": fft(G),
        "phase": phase,
        })
    Gf[::2] = 0

    Sf = ne.evaluate("S_fft * phase", {
        "S_fft": fft(S),
        "phase": phase,
        })

    a = 0.5

    for k in range(1, iteration_length):
        Gf_adjustment = ne.evaluate("a * ((1 / (-1j * w - Sf)) - Gf)", {
            "a": a,
            "w": w[1::2],
            "Sf": Sf[1::2],
            "Gf": Gf[1::2],
            })
        Gf[1::2] += Gf_adjustment
        diff_new = ne.evaluate("sum(abs(Gf_adjustment))")
        if k > 1:
            if diff_new > diff:
                a = 0.5 * a
        diff = diff_new

        G = ifft(ne.evaluate("Gf / phase"), Gf.size)

        S = ne.evaluate("(s_squared * 2/q_t * 1/(2**(3-2*q_t)) * (G ** (2*q_t - 1))) + (2/q * (1/(2**(3-2*q))) * (G ** (2*q - 1)))")
        Sf = ne.evaluate("S_fft * phase", {
            "S_fft": fft(S),
            "phase": phase,
            })

    return t, G, w, S, Sf
