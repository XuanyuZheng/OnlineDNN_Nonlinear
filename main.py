# main function for experiments
##################################################
# Author: {Xuanyu ZHENG}
# Copyright: Copyright {2020}, {online DL-based
# nonlinear channel estimator}
# Credits: [{Xuanyu ZHENG, Vincent LAU}]
# Version: {1}.{0}.{0}
##################################################
from torch.utils.data import Dataset, DataLoader
from NonlinearUtils import *
import torch.nn as nn
import matplotlib.pyplot as plt
import time


class NonlinearData(Dataset):
    # Constructor
    def __init__(self, sys, f_para, S, F, num):
        """
        Construct R = f(S' * h) + n using numpy, then convert to vectorized tensor
        :param sys: a dictionary containing sys = {'N': N, 'M': M, 'P': P, 'rho': rho,
         'SNR': SNR, 'fc': fc, 'lamda': lamda, 'D': D}
        :param f_para: the function parameters for SSA, f_para = {'A0':A0, 'p':p}
        :param S； pilot matrix, fixed for training & validation & testing set, complex N x M
        :param F； dft matrix, fixed for training & validation & testing set, complex N x N
        :param num: number of samples
        """
        N = sys['N']  # BS tx antenna number
        M = sys['M']  # pilot length
        P = sys['P']  # number of scatterers
        rho = sys['rho']  # tx power
        SNR = sys['SNR']  # per channel SNR
        fc = sys['fc']  # carrier frequency
        D = sys['D']  # antenna spacing
        noise_var = rho / (10.0 ** (SNR / 10.0))  # compute noise variance

        S_T = S.T    # transpose of pilot matrix
        F_real = np.real(F)
        F_imag = np.imag(F)

        A0_tx = f_para['A0_tx']  # clipping-level
        p_tx = f_para['p_tx']  # smoothness parameter
        A0_rx = f_para['A0_rx']  # clipping-level
        p_rx = f_para['p_rx']  # smoothness parameter

        # # get the fading channels from some distribution
        # get the fading low-rank channels from some distribution
        # note that the channel can come from other distributions, such as a CSM / COST2100 channel models
        phis_fix = np.arcsin(lamda / D * (-1 / 2 + np.arange(N) / N))
        # phis = phis_fix[[1, 7, 14]]
        phis = phis_fix[[1, 20, 40, 50]]    # choosing the first P AoDs
        # or random choice
        # phis = phis_fix[np.random.choice(N, P)]
        # phis = np.random.choice(N, (num, P))

        # we just use multipath channels to generate real-time channel data in this code
        h = generate_sparse_ch(sys, phis, lamda, D, num)  # shape = N x num
        h_tilde = complex_decompose(h)  # shape = 2N x num

        # computute the transmotted pilot
        fS_T = SSA_numpy(S_T, A0_tx, p_tx)
        # compute the transmitted basis A=f(S_H) * F
        A = fS_T @ F

        # get the real counterpart of measurement matrix
        A_real = np.real(A)
        A_imag = np.imag(A)
        A_row1 = np.concatenate((A_real, -A_imag), axis=1)
        A_row2 = np.concatenate((A_imag, A_real), axis=1)
        A_tilde = np.concatenate((A_row1, A_row2), axis=0)

        # get the anngular sparse channel
        x = F.conj().T @ h  # shape = N x num
        x_tilde = complex_decompose(x)  # shape = 2N x num

        Noise = (np.random.randn(M, num) + 1j*np.random.randn(M, num)) \
            * np.sqrt(noise_var/2)  # shape = num x Nr x Ls

        # get the linear output
        y = np.matmul(fS_T, h)  # shape = M x num

        # get the nonlinear output r
        y_amp = np.abs(y)
        y_agl = np.angle(y)
        r_amp = SSA_numpy(y_amp, A0_rx, p_rx)
        r_agl = y_agl
        r = r_amp * np.exp(1j*r_agl) + Noise
        # r = y + Noise
        r_tilde = complex_decompose(r)

        # transform to pytorch and swap dimension
        self.r_tilde = torch.from_numpy(r_tilde.T).float()    # real-vectorized nonlinear signal num x 2M
        self.h_tilde = torch.from_numpy(h_tilde.T).float()    # real-vectorized spatial channel
        self.x_tilde = torch.from_numpy(x_tilde.T).float()    # real-vectorized angular sparse channel
        self.A_tilde = torch.from_numpy(A_tilde.T).float()    # real-counterpart of RIP matrix
        self.F_real = torch.from_numpy(F_real).float()    # real-counterpart of real of DFT matrix
        self.F_imag = torch.from_numpy(F_imag).float()    # real-counterpart of imag of DFT matrix
        self.S = torch.from_numpy(S).float()

        self.F = F
        self.len = num

    def __getitem__(self, index):
        return self.r_tilde[index], self.h_tilde[index], self.x_tilde[index]

    # Get Length
    def __len__(self):
        return self.len

    # get the real counterpart RIP matrix
    def get_A_tilde(self):
        return self.A_tilde

    # get the real counterpart RIP matrix
    def get_F_RealImag(self):
        return self.F_real, self.F_imag

    # get the tensor version of pilot matrix
    def get_S(self):
        return self.S


class Net(nn.Module):
    def __init__(self, widths, train_poly=False, p=0.5):
        """
        :param widths: a list contains the widths of each layer
        :param p: dropout parameter, p=0 means no dropout, default is p=0
        """
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p)
        # initialize poly parameters
        self.Txbeta3 = torch.nn.Parameter(0.0 * torch.randn(1), requires_grad=train_poly)
        self.Txbeta5 = torch.nn.Parameter(0.0 * torch.randn(1), requires_grad=train_poly)
        self.Rxbeta3 = torch.nn.Parameter(0.0 * torch.randn(1), requires_grad=train_poly)
        self.Rxbeta5 = torch.nn.Parameter(0.0 * torch.randn(1), requires_grad=train_poly)

        for input_size, output_size in zip(widths, widths[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

        # By default, the weights are initialized by uniform distribution in [-1/sqrt(in_size), +1/sqrt(out_size)]
        # We can also use Xavier method for uniform distribution in [+-sqrt(6/(in_size+out_size))]
        # Explicitly call torch.nn.init.xavier_uniform_(linear1.weight)
        # for relu, use He initialization, calling torch.nn.init.kaiming_uniform_(linear1.weight, nonlinearity='relu')

    def forward(self, x):
        """
        :param x:   x = r_tilde: input of dimension num x 2M
        :return:
        """
        L = len(self.hidden)  # number of (hidden + output) layers
        p = 1
        S_tensor_T = torch.transpose(S_tensor, 0, 1)    # M x N
        for (l, linear) in zip(range(L), self.hidden):
            if l < L - 1:  # for hidden layers
                x = torch.relu(linear(x))
                x = self.drop(x)
            else:  # for output layer
                x_tilde_hat = linear(x)  # output channel, dim(x_tilde_hat) = num x 2N
                fS_T_hat = torch.mul(S_tensor_T, (1 + self.Txbeta3 * S_tensor_T ** (2 * p) + self.Txbeta5 * S_tensor_T ** (4 * p)))
                A_real = torch.matmul(fS_T_hat, F_real)
                A_imag = torch.matmul(fS_T_hat, F_imag)
                A_tilde_row1 = torch.cat((A_real, -A_imag), dim=1)
                A_tilde_row2 = torch.cat((A_imag, A_real), dim=1)
                A_tilde = torch.transpose(torch.cat((A_tilde_row1, A_tilde_row2), dim=0), 0, 1)

                y_tilde = torch.matmul(x_tilde_hat, A_tilde)  # linear output y, dim(y) = num x 2M
                y_real = y_tilde[:, 0:M]    # dim = num x M
                y_imag = y_tilde[:, M:]     # dim = num x M
                y_amp = torch.sqrt(y_real ** 2 + y_imag ** 2)   # dim = num x M
                r_amp = (1 + self.Rxbeta3 * y_amp ** (2 * p) + self.Rxbeta5 * y_amp ** (4 * p))  # DNN output r
                r_amp_dup = torch.cat((r_amp, r_amp), dim=1)    # dim = num x 2M
                r_tilde_hat = r_amp_dup * y_tilde   # dim = num x 2M

        return r_tilde_hat, x_tilde_hat


# A clever way of building DNN, where you do not need to add hidden layers manually
class NetCE(nn.Module):
    def __init__(self, widths, p=0):
        """
        :param widths: a list contains the widths of each layer
        :param p: dropout parameter, p=0 means no dropout, default is p=0
        """
        super(NetCE, self).__init__()
        self.hidden = nn.ModuleList()
        self.drop = nn.Dropout(p=p)

        for input_size, output_size in zip(widths, widths[1:]):
            linear = nn.Linear(input_size, output_size)
            self.hidden.append(linear)

    def forward(self, x):
        L = len(self.hidden)  # number of (hidden + output) layers
        for (l, linear) in zip(range(L), self.hidden):
            if l < L - 1:  # for hidden layers
                x = torch.relu(linear(x))
                x = self.drop(x)
            else:  # for output layer
                x_tilde_hat = linear(x)  # output channel, dim(x_tilde_hat) = 2N x num
        return x_tilde_hat


def train(model, criterion, train_loader, val_dataset, optimizer, epochs=2000):
    LOSS = []       # store the loss in training
    LOSS_val = []   # store the loss in validation
    ERR = []        # store the err of real-sparse vector in training
    ERR_val = []    # store the err of real-sparse vector in

    batch_size = train_loader.batch_size
    noise_var = rho / (10.0 ** (SNR / 10.0))  # compute noise variance
    gamma = np.sqrt(noise_var / rho)   # compute regularizer constant gamma
    print('gamma = ', gamma)
    # gamma = 0.01 / np.sqrt(rho)
    itera = 0
    for epoch in range(epochs):
        for r_tilde, h_tilde, x_tilde in train_loader:
            # Now r_tilde has dim = num x 2M
            itera = itera + 1
            model.train()
            optimizer.zero_grad()
            r_tilde_hat, x_tilde_hat = model(r_tilde)

            # received signal training
            x_real = x_tilde_hat[:, 0:N]
            x_imag = x_tilde_hat[:, N:]
            # complex lasso loss, normalized by signal size 2 * M and sample number
            loss = criterion(r_tilde_hat, r_tilde) + \
                   gamma / (2*N*batch_size) * torch.sum(torch.sqrt(x_real ** 2 + x_imag ** 2))
            loss.backward()

            optimizer.step()

            # tracking the loss & accuracy
            if itera % 10 == 0:
                # loss&err for every epoch
                model.eval()    # this is to turn to evaluatio mode (turn off drop out)
                r_tilde_hat, x_tilde_hat = model(r_tilde)
                # received signal training
                x_tilde_hat_real = x_tilde_hat[:, 0:N]
                x_tilde_hat_imag = x_tilde_hat[:, N:]
                loss = criterion(r_tilde_hat, r_tilde) + \
                       gamma / (2*N*batch_size) * torch.sum(torch.sqrt(x_tilde_hat_real ** 2 + x_tilde_hat_imag ** 2))  # least sqaure loss, normalized by signal size 2 * M
                LOSS.append(loss.data.item())
                err = CE_err(x_tilde, x_tilde_hat)
                ERR.append(err)

                # loss performance on validation set
                r_tilde_val = val_dataset.r_tilde.detach()  # received signal
                # h_tilde_val = val_dataset.h_tilde.detach()  # true spatial channel
                x_tilde_val = val_dataset.x_tilde.detach()  # true sparse channel
                val_num = x_tilde_val.shape[0]

                r_val_hat, x_val_hat = model(r_tilde_val)
                x_val_hat_real = x_val_hat[:, 0:N]
                x_val_hat_imag = x_val_hat[:, N:]
                # loss_val = criterion(h_hat_val, h_val.float())  # labeled data training
                loss_val = criterion(r_val_hat, r_tilde_val) + \
                    gamma / (2*N*val_num) * torch.sum(torch.sqrt(x_val_hat_real ** 2 + x_val_hat_imag ** 2))
                # loss_val = criterion(h_hat_val, S_bar, r_val, para)
                LOSS_val.append(loss_val.data.item())

                # channel estimation performance
                err_val = CE_err(x_tilde_val, x_val_hat)    # error in angular domain

                ERR_val.append(err_val)

                print('Epoch:', epoch, 'Itera:', itera,
                      'loss =', '{:.6f}'.format(loss.data.item()), 'loss_val =', '{:.6f}'.format(loss_val.data.item()),
                      'err =', '{:.3f}'.format(err.data.item()), 'err_val =', '{:.3f}'.format(err_val.data.item()),
                      'Txbeta3 =', '{:.3f}'.format(model.Txbeta3.item()), 'Txbeta5 =', '{:.3f}'.format((model.Txbeta5.item())),
                      'Rxbeta3 =', '{:.3f}'.format(model.Rxbeta3.item()), 'Rxbeta5 =', '{:.3f}'.format((model.Rxbeta5.item())))

    return LOSS, ERR, LOSS_val, ERR_val


# Start Experiment
# set system parameters
t0 = time.time()
N = 64      # BS tx antenna number
M = 20      # pilot length
P = 3       # number of scatterers
rho = 1     # tx power
SNR = 30    # per channel SNR
fc = 28e9    # carrier frequency
c = 3e8     # speed of light
lamda = c/fc        # carrier wavelength
D = 0.5 * lamda     # antenna spacing
sys = {'N': N, 'M': M, 'P': P,
       'rho': rho, 'SNR': SNR, 'fc': fc, 'lamda': lamda, 'D': D}
f_para = {'A0_tx': 1.5, 'p_tx': 1, 'A0_rx': 1.5, 'p_rx': 1}    #
# compute the true poly parameters to the 5-th order
a3_tx = - 1.0 / (2 * f_para['p_tx'] * f_para['A0_tx'] ** (2 * f_para['p_tx']))
a5_tx = (2 * f_para['p_tx'] + 1.0) / (2 * (2 * f_para['p_tx']) ** 2 * f_para['A0_tx'] ** (4 * f_para['p_tx']))
print('a3 =', a3_tx, 'a5 =', a5_tx)
# compute the DFT matrix
F = DFTMatrix(N)

# 2. load pilots S from numpy
S = np.load('S64.npy')

num = 500000          # real-time data
val_num = 1000      # validation data available
test_num = 1000     # test data available


# create training data
train_dataset = NonlinearData(sys, f_para, S, F, num)
A_tilde = train_dataset.get_A_tilde()
F_real, F_imag = train_dataset.get_F_RealImag()
S_tensor = train_dataset.get_S()
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

# validation data
val_dataset = NonlinearData(sys, f_para, S, F, val_num)

# create the model
in_size = 2 * M  # vectorized r as input
out_size = 2 * N  # vectorized h as input
print('in_size =', in_size, 'out_size =', out_size)
widths = [in_size, int(M*N/2), int(M*N/2), out_size]

model = Net(widths, train_poly=True, p=0.2)
# Set the learning rate and the optimizer
learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

# define MSE loss for training with labeled data
criterion = nn.MSELoss()

# train the model
# Set the model using dropout to training mode; this is the default mode, but it's good practice to write in code :
model.train()
t1 = time.time()
LOSS, ERR, LOSS_val, ERR_val = train(model, criterion, train_loader, val_dataset, optimizer, epochs=1)
# in online training, we will have a infinite data set containing many real-time measurement and we won't loop over the
# dataset (i.e, there is only one epoch)
# compute training time
t3 = time.time()
elapsed = t3 - t1
print("Time used for training is", elapsed, 'seconds')

# extract CE model
model_CE = NetCE(widths)
model_CE, _, _ = extract_CE_model(model_CE, model)

# save the model
torch.save(model.state_dict(), 'model.pt')
# save the metrics during training as numpy array during training
np.save('LOSS_iter_batch100.npy', np.array(LOSS))
np.save('ERR_iter_batch100.npy', np.array(ERR))
np.save('LOSS_val_iter_batch100.npy', np.array(LOSS_val))
np.save('ERR_val_iter_batch100.npy', np.array(ERR_val))


# load the model
modeled = Net(widths)
# modeled.cuda()
modeled.load_state_dict(torch.load('model.pt'))
# extract CE model
modeled_CE = NetCE(widths)
modeled_CE, Tx_poly_params, Rx_poly_params = extract_CE_model(modeled_CE, modeled)
# Set the model to evaluation mode, i.e., turn off dropout
modeled.eval()

# evaluate for SNR = 0:30 dB
SNRs = list(range(0, 31, 5))
SNR_num = len(SNRs)
ERR_test_SNR = []
ERR_clx_test_SNR = []
for s in range(SNR_num):
    sys['SNR'] = SNRs[s]
    test_data = NonlinearData(sys, f_para, S, F, test_num)
    r_test = test_data.r_tilde
    h_test = test_data.h_tilde
    x_test = test_data.x_tilde

    t11 = time.time()
    x_hat_test = modeled_CE(r_test).detach()
    t12 = time.time()
    err_test = CE_err(x_test, x_hat_test)

    h_test_np = h_test.numpy().T
    x_hat_test_np = x_hat_test.numpy().T
    h_test_clx = recover_complex_ch(h_test_np)      # N x num
    x_test_clx = recover_complex_ch(x_hat_test_np)     # N x num
    h_test_clx_hat = np.matmul(F, x_test_clx)
    err_clx_test = CE_clx_err(h_test_clx, h_test_clx_hat)

    ERR_test_SNR.append(err_test)   # real sparse channel error
    ERR_clx_test_SNR.append(err_clx_test)   # complex spatial channel error


np.save('err_test_SNR_poly.npy', np.array(ERR_test_SNR))
np.save('err_clx_test_SNR_poly_P3.npy', np.array(ERR_clx_test_SNR))
print(ERR_clx_test_SNR, t12-t11)


# plot LOSS in training and validation
plt.figure()
plt.plot(LOSS, label='train loss')
plt.plot(LOSS_val, label='val loss')
plt.legend()
plt.xlabel('iteration')
plt.ylabel('mean of loss function')
plt.grid()

# plot ERR in training and validation
plt.figure()
plt.plot(ERR, label='train err')
plt.plot(ERR_val, label='val err')
plt.xlabel('iteration')
plt.ylabel('MSE in dB')
plt.legend()
plt.grid()

# plot NMSE of channel estimation v.s. SNR
ERR_Linear_LS = np.load('err_linear_LS_SNRs.npy')
ERR_NonLinear_LS = np.load('err_nonlinear_LS_SNRs.npy')
ERR_Offline = np.load('err_clx_test_SNR_offline.npy')
plt.figure()
plt.plot(SNRs, ERR_test_SNR, '-*', label='Online DNN, pilot length = 20')
plt.plot(SNRs, ERR_Offline, '-d', label='Offline DNN, pilot length = 20')
# plt.plot(SNRs, ERR_clx_test_SNR, 'x',  label='complex test error M=10')
# plt.plot(SNRs, ERR_Linear_LS, '-o', label='LS with linear PA, pilot length = 20')
plt.plot(SNRs, ERR_NonLinear_LS, '-x', label='LS ignoring nonlinearity, pilot length = 40')
plt.xlabel('SNR in dB')
plt.ylabel('NMSE (dB) of channel estimation')
plt.legend()
plt.grid()

# plot PA estimation performance
A = np.arange(0, 2, 0.1)    # voltage in [0, 2]
V = f_para['A0_tx']
pp = f_para['p_tx']

A_ssa = SSA_numpy(A, V, pp)
A_Txpoly = POLY(A, Tx_poly_params)
A_Rxpoly = POLY(A, Rx_poly_params)
plt.figure()
plt.plot(A, A, label='ideal linear PA')
plt.plot(A, A_ssa, label='actual nonlinear PA')
plt.plot(A, A_Txpoly, 'o', label='estimated Tx nonlinear PA')
plt.plot(A, A_Rxpoly, 'x', label='estimated Rx nonlinear PA')
plt.xlabel('Input amplitude')
plt.ylabel('Output amplitude')
plt.title('Tx and Rx nonlinearity')
plt.legend()
plt.grid()

plt.show()
