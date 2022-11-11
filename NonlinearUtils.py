# This module contains utilities and the AM-AM transfer functions for nonlinear power amplifier
# These functions provides both PyTorch and Numpy implementations

import numpy as np
import torch
import scipy.io
from scipy.stats import unitary_group


def dft_column(N, x):
    """
    A column of DFT matrix
    :param N: size of DFT matrix
    :param x:  value of the column
    :return:    a column corresponding to x
    """
    k = np.arange(N)
    f = np.exp(1j * 2 * np.pi * x * k)

    return f


def DFTMatrix(N):
    """
    Return the DFT matrix F in Dai's paper
    :param N:   Size of DFT matrix -- N x N
    :return:    F -- the DFT matrix F in Dai's paper
    """
    # construct the DFT matrix
    F = np.zeros((N, N), dtype=complex)
    for n in range(N):
        x = -1 / 2 + n / N
        F[:, n] = dft_column(N, x) / np.sqrt(N)

    return F


def SSA_numpy(A, A0, p):
    """
    solid-state amplifiers (SSA) and exhibits only AM-AM distortion, numpy version, for generating data
    :param A: Amplitude of the input signal
    :param A0:  clipping level
    :param p:   smoothness parameter
    :return:    nonlinear distorted signal
    """
    return np.divide(A, (1 + (np.divide(A, A0)) ** (2 * p)) ** (1 / (2 * p)))
    # return A  # for perfect linear power amplifier


def SSA_torch(A, A0, p):
    """
    solid-state amplifiers (SSA) and exhibits only AM-AM distortion, torch version, for computing loss
    :param A: Amplitude of the input signal
    :param A0:  clipping level
    :param p:   smoothness parameter
    :return:    nonlinear distorted signal
    """
    return torch.div(A, (1 + (torch.div(A, A0)) ** (2 * p)) ** (1 / (2 * p)))
    # return A  # for perfect linear power amplifier


def POLY_SSA_torch(A, A0, p):
    """
    "polynomial for SSA model. For the special case of only AM-AM distortion these parameters are real.
    The AM-AM distortion model is then given by g(A = sum_{n=0~N-1} beta(n+1) * A ** (n+1)"
    "it is useful to define the following special case of (6.10),
    which takes into account the odd orders up to the 5th"
    :param A: Amplitude of the input signal
    :param A0:  clipping level
    :param p:   smoothness parameter
    :return:    nonlinear distorted signal
    """
    beta3 = - 1.0/(2*p*A0**(2*p))
    beta5 = (2*p + 1.0)/(2 * (2*p)**2 * A0**(4*p))
    # beta7 = -5/(16 * A0**6)   # for p = 1 only
    # beta9 = 35/(128 * A0**6)   # for p = 1 only

    beta7 = 0
    beta9 = 0
    return A * (1.0 + beta3 * A**(2*p) + beta5 * A**(4*p) + beta7 * A**(6*p) + beta9 * A**(8*p))


def POLY(A, para):
    """
    :param A: Input Signal Strength
    :param para:  odd order polynomial parameters
    :return: nonlinear distorted signal
    """
    p = 1   # for p = 1 SSA model
    beta3 = para[0]    # 3rd order parameter
    beta5 = para[1]    # 5th order parameter
    return A * (1.0 + beta3 * A ** (2 * p) + beta5 * A ** (4 * p))


def generate_ch(Nr, K, num):
    """
    :param Nr:
    :param K:
    :param num: number of samples
    :return: Gaussian channel matrix H
    """
    H = (np.random.randn(num, Nr, K) + 1j * np.random.randn(num, Nr, K)) / np.sqrt(2)
    return H


def steering_vector(phi, N, lamda, d):
    """
    Generate steering vectors for one AOA
    :param N: BS antenna number
    :param phi: AOA/AOD
    :param lamda: wavelength
    :param d: antenna spacing
    :return: a(phi) -- the steering vector
    """
    k = np.arange(N)
    a = 1/np.sqrt(N) * np.exp(-1j*2*np.pi*d/lamda * np.sin(phi) * k)
    return a


def generate_sparse_ch(sys, phis, lamda, d, num):
    """
    :param sys: system parameters
    :param phis: AoAs
    :param lamda: wavelength
    :param d: antenna spacing
    :param num: number of samples
    :return: low-rank channel matrix H
    """
    P = sys['P']
    N = sys['N']
    h = np.zeros((N, num), dtype=complex)
    if len(phis.shape) > 1:     # random AoA
        for n in range(num):
            # generate one realization of channel
            xi = 1 / np.sqrt(2) * (np.random.randn(P, 1) + 1j * np.random.randn(P, 1))
            h_n = np.zeros(N, dtype=complex)
            phi_n = phis[n]
            for p in range(P):
                a_p = steering_vector(phi_n[p], N, lamda, d)
                h_n = h_n + 1 / np.sqrt(P) * xi[p] * a_p
            h[:, n] = h_n
    else:   # fixed AoA
        for n in range(num):
            # generate one realization of channel
            xi = 1/np.sqrt(2) * (np.random.randn(P, 1) + 1j * np.random.randn(P, 1))
            h_n = np.zeros(N, dtype=complex)
            for p in range(P):
                a_p = steering_vector(phis[p], N, lamda, d)
                h_n = h_n + 1/np.sqrt(P) * xi[p] * a_p
            h[:, n] = h_n

    return h


def generate_pilot(N, M, rho=1, p_type='UnitModule'):
    """
    :param p_type: type of pilot, = 'UnitModule' or 'RealGaussian'
    :param N: tx antenna num
    :param M:  pilot length
    :param rho: tx power
    :return:    pilot matrix S satisfying trace(S*S^T)= rho * M
    """
    if p_type == 'UnitModule':
        F = DFTMatrix(N)
        Theta = np.random.rand(N, M) * 2 * np.pi
        PS = np.sqrt(rho) * np.exp(1j * Theta)
        S = np.matmul(F, PS)
        return S, PS
    elif p_type == 'RealGaussian':
        S = np.sqrt(rho) * np.random.randn(N, M)
        PS = S
        return S, PS


def CE_err(H, H_hat):
    """

    :param H: true h of shape num x 2N
    :param H_hat: estimated shape of num x 2N
    :return: MSE in dB
    """
    diff = H - H_hat
    # err_normal = 10 * np.log10(np.power(np.linalg.norm(diff, ord='fro', axis=(1, 2)) /
    #                                     np.linalg.norm(H, ord='fro', axis=(1, 2)), 2))
    err_normal = 10*torch.log10(torch.div(torch.norm(diff, dim=1), torch.norm(H, dim=1)) ** 2)
    err_dB = torch.mean(err_normal)
    return err_dB    # torch version


def CE_clx_err(H, H_hat):
    """
    :param H: true complex h of shape N x num
    :param H_hat: estimated complex channel shape of N x num
    :return: MSE in dB
    """
    diff = H - H_hat
    # err_normal = 10 * np.log10(np.power(np.linalg.norm(diff, ord='fro', axis=(1, 2)) /
    #                                     np.linalg.norm(H, ord='fro', axis=(1, 2)), 2))
    err_normal = 10*np.log10(np.divide(np.linalg.norm(diff, axis=0), np.linalg.norm(H, axis=0)) ** 2)
    err_dB = np.mean(err_normal)
    return err_dB    # torch version


def extract_CE_model(model_CE, model):
    """
    :param model_CE: CE model
    :param model: complete model
    :return:
    """
    pretrained_dict = model.state_dict()
    model_CE_dict = model_CE.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_CE_dict}
    # 2. overwrite entries in the existing state dict
    model_CE_dict.update(pretrained_dict)
    # 3. load the new state dict
    model_CE.load_state_dict(pretrained_dict)
    Txbeta3 = model.Txbeta3.item()
    Txbeta5 = model.Txbeta5.item()
    Rxbeta3 = model.Rxbeta3.item()
    Rxbeta5 = model.Rxbeta5.item()
    # beta7 = model.beta7.item()

    Tx_poly_params = [Txbeta3, Txbeta5]
    Rx_poly_params = [Rxbeta3, Rxbeta5]
    return model_CE, Tx_poly_params, Rx_poly_params


def copy_weights(model_pre, model):
    """
    Copy weights from model_pre to model, these have the same DNN structure
    :param model_pre:
    :param model:
    :return: model
    """
    model_pre_state = model_pre.state_dict()
    own_state = model.state_dict()
    for name, param in model_pre_state.items():
        own_state[name].copy_(param)


def complex_decompose(h):
    """
    h_tilde = [Re(h);Im(h)]
    :param h: complex numpy matrix N x num
    :return: h_tilde 2N x num
    """
    h_real = np.real(h)
    h_imag = np.imag(h)
    h_tilde = np.concatenate((h_real, h_imag), axis=0)
    return h_tilde


def recover_complex_ch(h_tilde):
    """
    h_tilde is real and dim = 2N x num
    :param N: N
    :param h_tilde: real vectorized numpy vector
    :return: complex h N x num
    """
    N = int(h_tilde.shape[0]/2)
    h_real = h_tilde[0:N, :]
    h_imag = h_tilde[N:, :]

    h = h_real + 1j * h_imag
    return h


def real_imag_vec(H, Nr, K):
    """
    :param H: n x Nr x K complex matrix
    :param Nr: Nr
    :param K: K
    :return: h_tilde = vec([Re(H) Im(H)])
    """
    H_tilde = complex_decompose(H)
    h_tilde = np.reshape(H_tilde, (-1, 2 * Nr * K), order='F')  # vectorize real & imag
    return h_tilde


def load_dataset_SNR(dirct):
    """
    Load Y, S, H in of specific SNR, or random SNR
    :return:    S, Y, H
    """
    data_S = scipy.io.loadmat(dirct + '/S_8.mat')
    data_F = scipy.io.loadmat(dirct + '/F.mat')
    S = data_S['S']
    F = data_F['F']

    return S, F


# Set the learning rate and the optimizer
# A0 = 2
# p = 1
# # Y = torch.tensor(1.0, requires_grad=True, dtype=torch.float)
# H = torch.ones(2, 3, requires_grad=True, dtype=torch.float)
# S = torch.randn(3, 3)
# N = torch.randn(2, 3)
# Y = H @ S + N
# R = SSA_torch(Y, A0, p)
# loss = torch.sum(R)
# loss.backward()
# print(H.grad)
