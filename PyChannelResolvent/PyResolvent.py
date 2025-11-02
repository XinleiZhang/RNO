# This class is used for construct resolvent operator

import numpy as np
import time
from scipy import linalg


def c2p(Ny):
    theta = np.linspace(0, np.pi, Ny, dtype=np.float64)
    T = np.cos(np.outer(theta, np.arange(Ny, dtype=np.float64)))
    return T


def p2c(Ny):
    T = np.linalg.solve(c2p(Ny), np.eye(Ny))
    return T


def chebyshev_deriv(Ny, order=1):

    Op = np.zeros((Ny, Ny), dtype=np.float64)
    N = Ny - 1
    for n in range(N + 1):
        c = 1
        if (n == 0): c = 2
        for p in range(N + 1):
            if (p > n and (p + n - 1) % 2 == 0): Op[n, p] = 2 * p / c

    M = np.linalg.matrix_power(Op, order)
    return M


def deriv(Ny, order=1):

    Op = chebyshev_deriv(Ny, order=order)
    T = c2p(Ny)
    # M = T @ Op @ np.linalg.inv(T)
    M = T @ (np.linalg.solve(T.T, Op.T)).T
    if (order == 4):
        M[0, :] = M[2, :]
        M[1, :] = M[2, :]
        # M[2, :] = M[3, :]
        # M[3, :] = M[4, :]

        M[-1, :] = M[-3, :]
        M[-2, :] = M[-3, :]
    # M[-3, :] = M[-4, :]
    # M[-4, :] = M[-5, :]
    return M


def csd_component(csd_, component='xx'):
    # check dimension of csd_
    import numpy as np
    if csd_.ndim != 3:
        raise ValueError(
            'CSD matrix must have 3 dimensions: [frequency,y1,y2]')
    if component not in ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']:
        raise ValueError('\'component\' {component} not valid')

    if np.shape(csd_)[1] != np.shape(csd_)[2]:
        raise ValueError('CSD matrix must be square')

    Ny = np.shape(csd_)[1] // 3
    Ny0 = Ny * 0
    Ny1 = Ny * 1
    Ny2 = Ny * 2
    Ny3 = Ny * 3

    if component == 'xx':
        var = csd_[:, Ny0:Ny1, Ny0:Ny1]
    if component == 'xy':
        var = csd_[:, Ny0:Ny1, Ny1:Ny2]
    if component == 'xz':
        var = csd_[:, Ny0:Ny1, Ny2:Ny3]
    if component == 'yx':
        var = csd_[:, Ny1:Ny2, Ny0:Ny1]
    if component == 'yy':
        var = csd_[:, Ny1:Ny2, Ny1:Ny2]
    if component == 'yz':
        var = csd_[:, Ny1:Ny2, Ny2:Ny3]
    if component == 'zx':
        var = csd_[:, Ny2:Ny3, Ny0:Ny1]
    if component == 'zy':
        var = csd_[:, Ny2:Ny3, Ny1:Ny2]
    if component == 'zz':
        var = csd_[:, Ny2:Ny3, Ny2:Ny3]
    return 0.5 * (np.flip(np.flip(np.swapaxes(var, 1, 2), axis=2), axis=1) +
                  var)


def s2p_y_op(Ny):
    Op = np.zeros((Ny, Ny))
    for i in range(Ny):
        for j in range(Ny):
            Op[i, j] = np.cos(np.pi * i * j / (Ny - 1))
    return Op


def p2s_y_op(Ny):
    Op = np.linalg.solve(s2p_y_op(Ny), np.eye(Ny))
    return Op


def integral_weights(Ny):
    n = np.arange(Ny)
    n[1] = 3
    coeff = (1 + np.power(-1, n)) / (1 - n * n)
    return coeff @ p2s_y_op(Ny)


def weight_sum(psd_, Ny_, w_ax=1, s_ax=[]):
    weight = integral_weights(Ny_)
    if psd_.shape[w_ax] == Ny_:
        pass
    elif psd_.shape[w_ax] == Ny_ * 3:
        weight = np.concatenate([weight, weight, weight])
    else:
        print(f'wrong dimension in {w_ax}')
        return None  # Explicitly return None for error case

    # Reshape weight for proper broadcasting
    weight = np.expand_dims(weight,
                            axis=tuple(i for i in range(psd_.ndim)
                                       if i != w_ax))

    # Calculate weighted sum (this reduces dimensionality by 1)
    psd_weighted_sum = np.sum(psd_ * weight, axis=w_ax)

    # Adjust s_ax indices for the reduced array
    if s_ax:
        # Need to handle cases where s_ax might be after w_ax
        adjusted_s_ax = [ax if ax < w_ax else ax - 1 for ax in s_ax]
        psd_weighted_sum = np.sum(psd_weighted_sum, axis=tuple(adjusted_s_ax))

    return psd_weighted_sum


class Resolvent:

    def __init__(self, Re, Retau, Ny):
        self.Re = Re
        self.Retau = Retau
        self.nu = 1.0 / Re
        self.utau = self.Retau / self.Re
        self.Ny = Ny
        self.NyH = Ny // 2 + 1
        self.y = np.cos(np.linspace(0, np.pi, Ny))
        self.yplus = Retau * (1 - np.abs(self.y))
        self.Dy1 = deriv(self.Ny)
        self.Dy2 = deriv(self.Ny, 2)
        self.Dy4 = deriv(self.Ny, 4)
        self.integral_weight = integral_weights(Ny)
        self.integral_weights = np.concatenate(
            (self.integral_weight, self.integral_weight, self.integral_weight))
        self.C1 = np.hstack([np.eye(Ny), np.zeros((Ny, 2 * Ny))])
        self.C2 = np.hstack(
            [np.zeros((Ny, Ny)),
             np.eye(Ny), np.zeros((Ny, Ny))])
        self.C3 = np.hstack([np.zeros((Ny, 2 * Ny)), np.eye(Ny)])

    ################################################################################
    # private
    ################################################################################

    def __Laplacian(self, kx_, kz_, order=1):
        eye = np.eye(self.Ny, dtype=np.float64)
        k2 = kx_ * kx_ + kz_ * kz_
        L = self.Dy2 - k2 * eye

        if order == 2:
            k4 = k2 * k2
            L = -2 * k2 * self.Dy2 + k4 * eye
            L += self.Dy4
        return L

    def operator_OS(self, kx_, kz_, U_, nut_=None):
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        L = self.__Laplacian(kx_, kz_)  # Laplacian operator
        L2 = self.__Laplacian(kx_, kz_, 2)

        k2 = kx_ * kx_ + kz_ * kz_  # wave-number square
        dnutdy = self.Dy1 @ nut_
        ddnutdyy = self.Dy2 @ nut_
        ddUdyy = self.Dy2 @ U_

        eye = np.eye(self.Ny, dtype=np.complex128)

        OP = -1j * kx_ * (np.diag(U_) @ L - np.diag(ddUdyy))
        OP += np.diag(self.nu + nut_) @ L2
        OP += 2 * np.diag(dnutdy) @ (L @ self.Dy1) + np.diag(ddnutdyy) @ (
            self.Dy2 + k2 * eye)

        # OP = -1j * kx_ * np.diag(
        #     np.einsum('i,ij->j', U_, L, optimize=True) - ddUdyy)
        # OP += np.einsum('i,ij->j', self.nu + nut_, L2, optimize=True)
        # OP += 2 * np.einsum(
        #     'i,ij->j', dnutdy, L @ self.Dy1, optimize=True) + np.einsum(
        #         'i,ij->j', ddnutdyy, self.Dy2 + k2 * eye, optimize=True)

        return OP

    def operator_SQ(self, kx_, kz_, U_, nut_=None):
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        L = self.__Laplacian(kx_, kz_)  # Laplacian operator
        U = U_
        dnutdy = self.Dy1 @ nut_
        OP = -1j * kx_ * np.diag(U) + np.diag(self.nu + nut_) @ L + np.diag(
            dnutdy) @ self.Dy1
        return OP

    def operator_sweep(self, kx_, kz_, sweep_u_, sweep_v_, sweep_w_,
                       lambda_y_):
        # k_term = -np.sqrt(
        #     np.power(kx_ * sweep_u_, 2) + np.power(kz_ * sweep_w_, 2))
        k_term = -np.hypot(kx_ * sweep_u_, kz_ * sweep_w_)

        result = np.diag(k_term)

        # dVdy = self.Dy1 @ sweep_v_
        # result += np.einsum('i,i,ij->ij', lambda_y_, dVdy, self.Dy1)
        # result += np.einsum('i,i,ij->ij', lambda_y_, sweep_v_, self.Dy2)
        result += np.einsum('i,ij,j,jk->ik',
                            lambda_y_,
                            self.Dy1,
                            sweep_v_,
                            self.Dy1,
                            optimize=True)
        return result

        # return np.diag(k_term) + np.diag(lambda_y_) @ self.Dy1 @ np.diag(
        # sweep_v_) @ self.Dy1

        # dVdy = self.Dy1 @ sweep_v_
        # return np.diag(k_term) + np.diag(lambda_y_) @ (
        #     np.diag(dVdy) @ self.Dy1 + np.diag(sweep_v_) @ self.Dy2)

    def operator_C(self, kx_, kz_):
        Ny = self.Ny

        C = np.zeros((3 * Ny, 2 * Ny), dtype=np.complex128)

        eye = np.eye(Ny, dtype=np.complex128)

        k2 = kx_ * kx_ + kz_ * kz_
        C[:Ny, :Ny] = self.Dy1 * (1j * kx_ / k2)
        C[:Ny, Ny:2 * Ny] = eye * (-1j * kz_ / k2)
        C[Ny:2 * Ny, :Ny] = eye
        C[2 * Ny:3 * Ny, :Ny] = self.Dy1 * (1j * kz_ / k2)
        C[2 * Ny:3 * Ny, Ny:2 * Ny] = eye * (1j * kx_ / k2)

        return C

    def operator_D(self, kx_, kz_):
        Ny = self.Ny
        D = np.zeros((2 * Ny, 3 * Ny), dtype=np.complex128)
        D[:Ny, Ny:2 * Ny] = np.eye(Ny)
        D[Ny:, :Ny] = np.eye(Ny) * 1j * kz_
        D[Ny:, 2 * Ny:] = np.eye(Ny) * (-1j * kx_)
        return D

    ################################################################################
    # public
    ################################################################################

    def Cess_eddy_viscosity(self, s=1.0):
        import numpy as np
        kappa = 0.426
        A = 25.4
        f = kappa * self.Retau / 3 * (1 - self.y**2) * (1 + 2 * self.y**2) * (
            1 - np.exp(-self.yplus / A))
        nut = 0.5 * np.sqrt(1 + s * np.power(f, 2)) - 0.5
        nut *= self.nu
        return nut

    def eddy_viscosity_nondim(self, s=1.0):
        return self.eddy_viscosity(s=s) / self.nu

    def mean_velocity(self, nut_):
        dUdy = -self.utau**2 / (self.nu + nut_) * self.y
        Dy = np.copy(self.Dy1)
        Dy[0, 0] += 1.0
        U = np.linalg.solve(Dy, dUdy)
        U[0] = 0
        U[-1] = 0
        return U

    def operator_Ah(self,
                    kx_,
                    kz_,
                    U_,
                    nut_=None,
                    sweep_u_=None,
                    sweep_v_=None,
                    sweep_w_=None,
                    lambda_y_=None):

        Ny = self.Ny
        A = np.zeros((2 * Ny, 2 * Ny), dtype=np.complex128)
        U = U_
        dUdy = self.Dy1 @ U

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2

        A[Ny0:Ny1, Ny0:Ny1] = self.operator_OS(kx_, kz_, U_, nut_)
        A[Ny1:Ny2, Ny0:Ny1] = -1j * kz_ * np.diag(dUdy)
        A[Ny1:Ny2, Ny1:Ny2] = self.operator_SQ(kx_, kz_, U_, nut_)

        if sweep_u_ is not None:
            swp = self.operator_sweep(kx_, kz_, sweep_u_, sweep_v_, sweep_w_,
                                      lambda_y_)
            A[Ny0:Ny1, Ny0:Ny1] += (self.Dy1 @ swp @ self.Dy1 -
                                    (kx_ * kx_ + kz_ * kz_) * np.eye(self.Ny))
            A[Ny1:Ny2, Ny1:Ny2] += swp

        return A

    def operator_Bh(self, kx_, kz_):

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3
        k2 = kx_ * kx_ + kz_ * kz_

        B = np.zeros((Ny2, Ny3), dtype=np.complex128)
        eye = np.eye(Ny1, dtype=np.complex128)
        B[Ny0:Ny1, Ny0:Ny1] = -1j * kx_ * self.Dy1
        B[Ny0:Ny1, Ny1:Ny2] = -k2 * eye
        B[Ny0:Ny1, Ny2:Ny3] = -1j * kz_ * self.Dy1
        B[Ny1:Ny2, Ny0:Ny1] = 1j * kz_ * eye
        B[Ny1:Ny2, Ny2:Ny3] = -1j * kx_ * eye
        return B

    def operator_Bsh(self, kx_, kz_):
        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        B = np.zeros((Ny2, Ny3), dtype=np.complex128)
        eye = np.eye(Ny1, dtype=np.complex128)
        B[Ny0:Ny1, Ny1:Ny2] = self.__Laplacian(kx_, kz_)
        B[Ny1:Ny2, Ny0:Ny1] = 1j * kz_ * eye
        B[Ny1:Ny2, Ny2:Ny3] = -1j * kx_ * eye
        return B

    def solve_solenoidal(self, kx_, kz_, f):
        import numpy as np
        # This function isolate the solenoidal part of vector f
        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        Matrix_B = np.zeros((Ny3, Ny3))
        Matrix_B[Ny0:Ny2, :] = self.operator_Bh(kx_, kz_)
        Matrix_B[0, :] = 0
        Matrix_B[Ny1 - 1, :] = 0

        Lap = self.__Laplacian(kx_, kz_)
        Matrix_a = np.zeros((Ny3, Ny3))
        Matrix_a[Ny0:Ny1, Ny1:Ny2] = Lap
        Matrix_a[Ny1:Ny2, Ny0:Ny1] = 1j * kz_ * np.eye(Ny1)
        Matrix_a[Ny1:Ny2, Ny2:Ny3] = -1j * kx_ * np.eye(Ny1)
        Matrix_a[Ny2:Ny3, Ny0:Ny1] = 1j * kx_ * np.eye(Ny1)
        Matrix_a[Ny2:Ny3, Ny1:Ny2] = self.Dy1
        Matrix_a[Ny2:Ny3, Ny2:Ny3] = 1j * (kz_ * np.eye(Ny1))

        # BC
        Matrix_a[0, :] = 0
        Matrix_a[0, Ny1] = 1
        Matrix_a[Ny1 - 1, :] = 0
        Matrix_a[Ny1 - 1, Ny2 - 1] = 1

        # return np.linalg.inv(Matrix_a) @ (Matrix_B @ f)
        return np.linalg.solve(Matrix_a, Matrix_B @ f)

    def operator_L_swp(self, kx_, kz_, U_, sweep_u_, sweep_v_, sweep_w_,
                       lambda_y_):
        """
        compute the linear operator L in prime resolvent
        """
        # Initialize nut_ safely and validate shapes
        U_ = np.asarray(U_)
        sweep_u_ = np.zeros(
            self.Ny) if sweep_u_ is None else np.asarray(sweep_u_)
        sweep_v_ = np.zeros(
            self.Ny) if sweep_v_ is None else np.asarray(sweep_v_)
        sweep_w_ = np.zeros(
            self.Ny) if sweep_w_ is None else np.asarray(sweep_w_)
        lambda_y_ = np.zeros(
            self.Ny) if lambda_y_ is None else np.asarray(lambda_y_)

        if sweep_u_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_u_ must be shape (Ny,). Expected {self.Ny}, got {sweep_u_.shape}'
            )
        if sweep_v_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_v_ must be shape (Ny,). Expected {self.Ny}, got {sweep_v_.shape}'
            )
        if sweep_w_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_w_ must be shape (Ny,). Expected {self.Ny}, got {sweep_w_.shape}'
            )
        if lambda_y_.shape != (self.Ny, ):
            raise ValueError(
                f'lambda_y_ must be shape (Ny,). Expected {self.Ny}, got {lambda_y_.shape}'
            )

        # Precompute reusable terms
        Lap = self.__Laplacian(kx_, kz_).copy()
        dUdy = self.Dy1 @ U_
        Ny = self.Ny
        L = np.zeros((3 * Ny, 3 * Ny), dtype=np.complex128)

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        U_diag = -1j * kx_ * U_
        L[Ny0:Ny1, Ny0:Ny1] = np.diag(U_diag) + self.nu * Lap.copy()
        L[Ny0:Ny1, Ny1:Ny2] = -np.diag(dUdy)
        L[Ny1:Ny2, Ny1:Ny2] = L[Ny0:Ny1, Ny0:Ny1]
        L[Ny2:Ny3, Ny2:Ny3] = L[Ny0:Ny1, Ny0:Ny1]

        swp = self.operator_sweep(kx_, kz_, sweep_u_, sweep_v_, sweep_w_,
                                  lambda_y_)
        # dvdy = (self.Dy1 @ sweep_v_).copy()
        # d_sub = np.einsum(
        #     'i,ij->ij', lambda_y_ * dvdy, self.Dy1.copy()) + np.einsum(
        #         'i,ij->ij', lambda_y_ * sweep_v_, self.Dy2.copy())
        # d_sub -= np.diag(np.hypot(kx_ * sweep_u_, kz_ * sweep_w_))
        L[Ny0:Ny1, Ny0:Ny1] += swp
        L[Ny1:Ny2, Ny1:Ny2] += swp
        L[Ny2:Ny3, Ny2:Ny3] += swp
        # L[:Ny3, :Ny3] += L_swp
        return L

    def operator_L(self, kx_, kz_, omega_, U_, nut_=None):
        """"
        computes the linear operator L

        Parameters:
            kx_, kz_ : float
                wavenumbers in x and z directions
            U_ : array-like, shape (Ny,)
                mean velocity profile
        
        Returns:
            L: array-like, shape (3*Ny, 3*Ny)
                The computaed operator (complex128)
        
        Raises:
            ValueError: If U_ or nut_ do not match size Ny
        """

        # Initialize nut_ safely and validate shapes
        U_ = np.asarray(U_)
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)

        if U_.shape != (self.Ny, ):
            raise ValueError(
                f'U_ must be shape (Ny,). Expected {self.Ny}, got {U_.shape}')
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        # This function returns the resolvent operator L
        # f_s=Lu
        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        Ah = self.operator_Ah(kx_, kz_, U_, nut_)
        D = self.operator_D(kx_, kz_)
        L = np.zeros((Ny2, Ny2))
        Lap = self.__Laplacian(kx_, kz_)
        L[Ny0:Ny1, Ny0:Ny1] = Lap
        L[Ny1:Ny2, Ny1:Ny2] = np.eye(self.Ny)

        Matrix_s = np.zeros((Ny3, Ny3), dtype=np.complex128)
        Matrix_s[:Ny2, :] = -(-1j * omega_ * L + Ah) @ D
        Matrix_s[0, :] = 0
        Matrix_s[Ny1 - 1, :] = 0

        Matrix_a = np.zeros((Ny3, Ny3), dtype=np.complex128)
        Matrix_a[Ny0:Ny1, Ny1:Ny2] = Lap
        Matrix_a[Ny1:Ny2, Ny0:Ny1] = 1j * kz_ * np.eye(Ny1)
        Matrix_a[Ny1:Ny2, Ny2:Ny3] = -1j * kx_ * np.eye(Ny1)
        Matrix_a[Ny2:Ny3, Ny0:Ny1] = 1j * kx_ * np.eye(Ny1)
        Matrix_a[Ny2:Ny3, Ny1:Ny2] = self.Dy1
        Matrix_a[Ny2:Ny3, Ny2:Ny3] = 1j * kz_ * np.eye(Ny1)

        # BC
        Matrix_a[0, :] = 0
        Matrix_a[0, Ny1] = 1
        Matrix_a[Ny1 - 1, :] = 0
        Matrix_a[Ny1 - 1, Ny2 - 1] = 1

        return L

    def operator_R(self,
                   kx_,
                   kz_,
                   omega_,
                   U_,
                   nut_=None,
                   sweep_u_=None,
                   sweep_v_=None,
                   sweep_w_=None,
                   lambda_y_=None):
        """
        # This function return the resolvent operator R
        # u=Rf
        """
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        Ny1 = self.Ny
        Ny2 = self.Ny * 2

        # use OS resolvent
        C = self.operator_C(kx_, kz_)
        Ah = self.operator_Ah(kx_, kz_, U_, nut_, sweep_u_, sweep_v_, sweep_w_,
                              lambda_y_)
        Bh = self.operator_Bh(kx_, kz_)

        # discrete fourth-order equation of state vector with homogeneous boundary condition
        L = np.zeros((Ny2, Ny2))
        L[:Ny1, :Ny1] = self.__Laplacian(kx_, kz_)
        L[Ny1:Ny2, Ny1:Ny2] = np.eye(Ny1)

        # -(i\omega L+A_h)
        A = -(-1j * omega_ * L + Ah)

        # boundary conditions
        # v(1)=0
        A[0, :] = 0
        A[0, 0] = 1
        Bh[0, :] = 0
        # v(-1)=0
        A[self.Ny - 1, :] = 0
        A[self.Ny - 1, self.Ny - 1] = 1
        Bh[self.Ny - 1, :] = 0
        # dvdy(1)=0
        A[1, :] = 0
        A[1, :self.Ny] = self.Dy1[0, :]
        Bh[1, :] = 0
        # dvdy(-1)=0
        A[self.Ny - 2, :] = 0
        A[self.Ny - 2, :self.Ny] = self.Dy1[-1, :]
        Bh[self.Ny - 2, :] = 0
        # omega_y(1)=0
        A[self.Ny, :] = 0
        A[self.Ny, self.Ny] = 1
        Bh[self.Ny, :] = 0
        # omega_y(-1)=0
        A[-1, :] = 0
        A[-1, -1] = 1
        Bh[-1, :] = 0

        # return C @ np.linalg.inv(A) @ Bh
        R = C @ np.linalg.solve(A, Bh)

        return R

    def operator_Hh(self, kx_, kz_, omega_, U_, nut_=None):
        # This function return the resolvent operator Hh
        # Hh u=Bh f
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3
        Ah = self.operator_Ah(kx_, kz_, U_, nut_)
        D = self.operator_D(kx_, kz_)
        L = np.zeros((Ny2, Ny2))
        Lap = self.__Laplacian(kx_, kz_)
        L[Ny0:Ny1, Ny0:Ny1] = Lap
        L[Ny1:Ny2, Ny1:Ny2] = np.eye(Ny1)

        Matrix = np.zeros((Ny2, Ny3), dtype=np.complex128)
        Matrix = -(-1j * omega_ * L + Ah) @ D

        return Matrix

    def operator_div(self, kx_, kz_):
        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3
        D = np.zeros((self.Ny, self.Ny * 3), dtype=np.complex128)
        D[:, Ny0:Ny1] = 1j * kx_ * np.eye(Ny1)
        D[:, Ny1:Ny2] = self.Dy1.copy()
        D[:, Ny2:Ny3] = 1j * kz_ * np.eye(Ny1)
        return D

    def operator_H_rapid(self, kx_, U_):
        U_ = np.zeros(self.Ny) if U_ is None else np.asarray(U_)
        dUdy = (self.Dy1 @ U_).copy()
        H_r = -2j * kx_ * np.diag(dUdy)
        return H_r

    def div_nut(self, kx_, kz_, nut_):
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        dnutdy = np.einsum('ij,j->i', self.Dy1, nut_, optimize=True)
        dnutdy2 = np.einsum("ij,j->i", self.Dy2, nut_, optimize=True)
        H_s = 2 * np.diag(dnutdy) @ self.__Laplacian(kx_, kz_).copy()
        H_s += 2 * np.diag(dnutdy2) @ self.Dy1.copy()
        return H_s

    def div_sweep(self, kx_, kz_, sweep_u_, sweep_v_, sweep_w_, lambda_y_):
        sweep_u_ = np.zeros(
            self.Ny) if sweep_u_ is None else np.asarray(sweep_u_)
        sweep_v_ = np.zeros(
            self.Ny) if sweep_v_ is None else np.asarray(sweep_v_)
        sweep_w_ = np.zeros(
            self.Ny) if sweep_w_ is None else np.asarray(sweep_w_)
        lambda_y_ = np.zeros(
            self.Ny) if lambda_y_ is None else np.asarray(lambda_y_)

        if sweep_u_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_u_ must be shape (Ny,). Expected {self.Ny}, got {sweep_u_.shape}'
            )
        if sweep_v_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_v_ must be shape (Ny,). Expected {self.Ny}, got {sweep_v_.shape}'
            )
        if sweep_w_.shape != (self.Ny, ):
            raise ValueError(
                f'sweep_w_ must be shape (Ny,). Expected {self.Ny}, got {sweep_w_.shape}'
            )
        if lambda_y_.shape != (self.Ny, ):
            raise ValueError(
                f'lambda_y_ must be shape (Ny,). Expected {self.Ny}, got {sweep_w_.shape}'
            )

        swp = self.operator_sweep(kx_, kz_, sweep_u_, sweep_v_, sweep_w_,
                                  lambda_y_)
        return self.Dy1 @ swp - swp @ self.Dy1

    def Helmholtz_num_solver(self, kx_, kz_):
        """
        Solves the Helmholtz equation (∂_yy - k²)p = s with Neumann boundary conditions ∂_y p(±1) = 0.
        
        Uses spectral numerical approach to return a linear operator solution.
        
        Parameters:
            kx_ (float): x-component of wavevector
            kz_ (float): z-component of wavevector
            
        Returns:
            Linear operator that solves the Helmholtz equation for given wavevector components
            
        Raises:
            ValueError: if wavevector components are both zero
        """
        Lap = self.__Laplacian(kx_, kz_).copy()
        Lap[0, :] = self.Dy1[0, :].copy()
        Lap[-1, :] = self.Dy1[-1, :].copy()
        B = np.eye(self.Ny)
        B[0, 0] = 0
        B[-1, -1] = 0
        return np.linalg.solve(Lap, B)

    def solenoidal_forcing_csd_dns(self,
                                   U_,
                                   kx_,
                                   kz_,
                                   freq_,
                                   vel_csd_,
                                   nut_=None):

        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )

        if np.size(freq_) != np.shape(vel_csd_)[0]:
            print(
                f'shape of freq_{np.shape(freq_)}, shape of vel_csd_ {np.shape(vel_csd_)}'
            )
            raise ValueError('shape of frequency not consistent')
        f_csd = np.zeros_like(vel_csd_, dtype=np.complex128)

        for i in range(np.size(freq_)):
            omega = freq_[i]
            t1 = time.time()
            L = self.operator_L(kx_, kz_, omega, U_, nut_)
            t2 = time.time()
            f_csd[i, :, :] = np.einsum('ij,jk,kl->il',
                                       L,
                                       vel_csd_[i, :, :],
                                       np.conjugate(L.T),
                                       optimize=True)
            t3 = time.time()
            print(
                f'freq {i} = {freq_[i]:.4f}, operator L: {t2-t1:.4e}, csd: {t3-t2:.4e}'
            )
        return f_csd

    def solenoidal_forcing_psd_match_m1(self, U_, nut_, kx_, kz_, freq_,
                                        vel_csd_):
        """
        DONT USE THIS FUNCTION (lack of physical interpretation)
        """
        if np.size(freq_) != np.shape(vel_csd_)[0]:
            print(
                f'shape of freq_{np.shape(freq_)}, shape of vel_csd_ {np.shape(vel_csd_)}'
            )
            raise ValueError('shape of frequency not consistent')
        f_psd = np.zeros((np.size(freq_), self.Ny * 3))

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        for i in range(np.size(freq_)):
            t1 = time.time()
            omega = freq_[i]
            Hh = self.operator_Hh(kx_, kz_, omega, U_, nut_)
            Bh = self.operator_Bsh(kx_, kz_)
            R = self.operator_R(kx_, kz_, omega, U_, nut_)
            Bh2 = np.real(np.power(np.abs(Bh), 2))
            Hh2 = np.real(np.power(np.abs(Hh), 2))
            Bh2_inv = np.linalg.pinv(Bh2)
            stf_psd = np.einsum('ij,jk,k->i',
                                Bh2_inv,
                                Hh2,
                                np.diag(np.real(vel_csd_[i, :, :])),
                                optimize=True)

            # determine the stf_psd magnitude
            vel_psd_check = np.einsum('ij,j->i', np.power(np.abs(R), 2),
                                      stf_psd)
            gamma_1 = np.max(np.diag(vel_csd_[i, :, :])[Ny0:Ny1]) / np.max(
                vel_psd_check[Ny0:Ny1])
            gamma_2 = np.max(np.diag(vel_csd_[i, :, :])[Ny1:Ny2]) / np.max(
                vel_psd_check[Ny1:Ny2])
            gamma_3 = np.max(np.diag(vel_csd_[i, :, :])[Ny2:Ny3]) / np.max(
                vel_psd_check[Ny2:Ny3])

            stf_psd[Ny0:Ny1] *= np.real(gamma_1)
            stf_psd[Ny1:Ny2] *= np.real(gamma_2)
            stf_psd[Ny2:Ny3] *= np.real(gamma_3)

            f_psd[i, :] = np.real(stf_psd)

            t3 = time.time()
            print(f'freq {i} = {freq_[i]:.4f}, using time {t3-t1:.4e}')
        return f_psd

    def solenoidal_forcing_psd_match_m2(self,
                                        U_,
                                        kx_,
                                        kz_,
                                        freq_,
                                        vel_csd_,
                                        nut_=None):
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )
        """
        NOT SUGGESTION FOR THIS FUNCTION (trivial performance)
        """
        if np.size(freq_) != np.shape(vel_csd_)[0]:
            print(
                f'shape of freq_{np.shape(freq_)}, shape of vel_csd_ {np.shape(vel_csd_)}'
            )
            raise ValueError('shape of frequency not consistent')
        f_psd = np.zeros((np.size(freq_), self.Ny * 3))

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        for i in range(np.size(freq_)):
            t1 = time.time()
            omega = freq_[i]
            L = self.operator_L(kx_, kz_, omega, U_, nut_)
            stf_psd = np.einsum('ij,jk,ki->i',
                                L,
                                vel_csd_[i, :, :],
                                np.conj(L.T),
                                optimize=True)

            # determine the stf_psd magnitude
            R = self.operator_R(kx_, kz_, omega, U_, nut_)
            vel_psd_check = np.einsum('ij,j,ji->i', R, stf_psd, np.conj(R.T))

            gamma_1 = np.max(np.diag(vel_csd_[i, :, :])[Ny0:Ny1]) / np.max(
                vel_psd_check[Ny0:Ny1])
            gamma_2 = np.max(np.diag(vel_csd_[i, :, :])[Ny1:Ny2]) / np.max(
                vel_psd_check[Ny1:Ny2])
            gamma_3 = np.max(np.diag(vel_csd_[i, :, :])[Ny2:Ny3]) / np.max(
                vel_psd_check[Ny2:Ny3])

            stf_psd[Ny0:Ny1] *= np.real(gamma_1)
            stf_psd[Ny1:Ny2] *= np.real(gamma_2)
            stf_psd[Ny2:Ny3] *= np.real(gamma_3)

            f_psd[i, :] = np.real(stf_psd)

            t3 = time.time()
            print(f'freq {i} = {freq_[i]:.4f}, using time {t3-t1:.4e}')
        return f_psd

    def solenoidal_forcing_psd_match_m3(self, U_, nut_, kx_, kz_, freq_,
                                        vel_csd_):
        """
        The same as _m4, this one the the older version, computational costly
        (good performance, well physical interpretation)
        """
        if np.size(freq_) != np.shape(vel_csd_)[0]:
            print(
                f'shape of freq_{np.shape(freq_)}, shape of vel_csd_ {np.shape(vel_csd_)}'
            )
            raise ValueError('shape of frequency not consistent')
        f_psd = np.zeros((np.size(freq_), self.Ny * 3))

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        weight = integral_weights(Ny1)
        mat_w_left = np.zeros((3, Ny3))
        mat_w_right = np.zeros((Ny3, 3))
        mat_w_left[0, Ny0:Ny1] = weight
        mat_w_left[1, Ny1:Ny2] = weight
        mat_w_left[2, Ny2:Ny3] = weight
        mat_w_right[Ny0:Ny1, 0] = 1
        mat_w_right[Ny1:Ny2, 1] = 1
        mat_w_right[Ny2:Ny3, 2] = 1
        mat_A = np.zeros((3, 3))
        mat_b = np.zeros(3)

        for i in range(np.size(freq_)):
            t1 = time.time()
            omega = freq_[i]
            L = self.operator_L(kx_, kz_, omega, U_, nut_)
            stf_psd = np.einsum('ij,jk,ki->i',
                                L,
                                vel_csd_[i, :, :],
                                np.conj(L.T),
                                optimize=True)

            # determine the stf_psd magnitude
            R = self.operator_R(kx_, kz_, omega, U_, nut_)
            R2 = np.power(np.abs(R), 2)
            R2K = R2 @ np.diag(np.real(stf_psd))
            mat_tke_res = np.einsum('ij,jk,kl->il',
                                    mat_w_left,
                                    R2K,
                                    mat_w_right,
                                    optimize=True)
            mat_tke_dns = np.einsum('ij,j->i', mat_w_left,
                                    np.diag(np.real(vel_csd_[i, :, :])))
            # solve coefficient
            mat_A[0, :] = np.sum(mat_tke_res, axis=0)
            mat_b[0] = np.sum(mat_tke_dns)

            mat_A[1, 1] = 1
            mat_A[1, 0] = -1
            mat_b[1] = 0

            mat_A[2, 2] = 1
            mat_A[2, 1] = -1
            mat_b[2] = 0

            gamma = np.linalg.solve(mat_A, mat_b)

            stf_psd *= gamma[0]

            f_psd[i, :] = np.real(stf_psd)

            t3 = time.time()
            print(f'freq {i} = {freq_[i]:.4f}, using time {t3-t1:.4e}')
        return f_psd

    def solenoidal_forcing_psd_match_m4(self,
                                        U_,
                                        kx_,
                                        kz_,
                                        freq_,
                                        vel_csd_,
                                        nut_=None):
        nut_ = np.zeros(self.Ny) if nut_ is None else np.asarray(nut_)
        if nut_.shape != (self.Ny, ):
            raise ValueError(
                f'nut_ must be shape (Ny,). Expected {self.Ny}, got {nut_.shape}'
            )
        """
        USE THIS! (good performance, well physical interpretation, and computational efficiency)
        """
        if np.size(freq_) != np.shape(vel_csd_)[0]:
            print(
                f'shape of freq_{np.shape(freq_)}, shape of vel_csd_ {np.shape(vel_csd_)}'
            )
            raise ValueError('shape of frequency not consistent')
        f_psd = np.zeros((np.size(freq_), self.Ny * 3))

        Ny0 = 0
        Ny1 = self.Ny
        Ny2 = self.Ny * 2
        Ny3 = self.Ny * 3

        weight = integral_weights(Ny1)
        weights = np.concatenate((weight, weight, weight))

        for i in range(np.size(freq_)):
            t1 = time.time()
            omega = freq_[i]
            L = self.operator_L(kx_, kz_, omega, U_, nut_)

            t2 = time.time()
            stf_psd = np.einsum('ij,jk,ki->i',
                                L,
                                vel_csd_[i, :, :],
                                np.conj(L.T),
                                optimize=True)
            t3 = time.time()
            stf_psd_temp = np.copy(stf_psd)
            stf_psd[Ny0:Ny1] = 0.5 * (stf_psd_temp[Ny0:Ny1] +
                                      np.flip(stf_psd_temp[Ny0:Ny1]))
            stf_psd[Ny1:Ny2] = 0.5 * (stf_psd_temp[Ny1:Ny2] +
                                      np.flip(stf_psd_temp[Ny1:Ny2]))
            stf_psd[Ny2:Ny3] = 0.5 * (stf_psd_temp[Ny2:Ny3] +
                                      np.flip(stf_psd_temp[Ny2:Ny3]))

            # determine the stf_psd magnitude
            R = self.operator_R(kx_, kz_, omega, U_, nut_)

            t4 = time.time()
            vel_psd = np.einsum('ij,j,ji->i',
                                R,
                                stf_psd,
                                np.conj(R.T),
                                optimize=True)
            tke_res = np.dot(vel_psd, weights)
            tke_dns = np.dot(np.diag(vel_csd_[i, :, :]), weights)

            gamma = np.real(tke_dns) / np.real(tke_res)

            f_psd[i, :] = np.real(stf_psd * gamma)

            t5 = time.time()
            print(
                f'freq {i:4d} = {freq_[i]:8.2e}, using time: {t5-t1:.2e}, indetails L: {t2-t1:.2e}, stf_psd: {t3-t2:.2e}, R: {t4-t3:.3e}, f_psd: {t5-t4:.3e}'
            )
        return f_psd


import numpy as np
import os
import PyChannelResolvent.ChannelField as CF
import PyChannelResolvent.WelchFFT as WelchFFT
from scipy.fft import fftfreq, fftshift


class DNSVEL:

    def __init__(self,
                 Re,
                 Retau,
                 Ny,
                 Nx,
                 Nz,
                 Nt_window,
                 dt_sample,
                 kxid,
                 kzid,
                 dkx,
                 dkz,
                 caseDir=''):
        self.var_dict = {}
        self.Re = Re
        self.Retau = Retau
        self.nu = 1.0 / Re
        self.utau = Retau / Re
        self.dt = dt_sample
        self.Nt_window = Nt_window
        self.Ny = Ny
        self.NyH = Ny // 2 + 1
        self.Nx = Nx
        self.NxH = Nx // 2
        self.Nz = Nz
        self.NzH = Nz // 2
        self.kxid = kxid
        self.kzid = kzid
        self.y = np.cos(np.linspace(0, np.pi, Ny))
        self.dist = 1 - np.abs(self.y)
        self.yplus = Retau * (1 - np.abs(self.y))
        self.omega = fftshift(
            fftfreq(int(Nt_window), d=dt_sample / (2 * np.pi)))
        self.wavex = np.linspace(0, -self.NxH + 1, self.NxH) * dkx
        self.wavez = np.roll(np.linspace(self.NzH - 1, -self.NzH, Nz),
                             -(self.NzH - 1)) * dkz
        self.domega = 2 * np.pi / (Nt_window * dt_sample)
        self.kx = self.wavex[kxid]
        self.kz = self.wavez[kzid]
        self.dkx = dkx
        self.dkz = dkz
        self.dx = 2 * np.pi / (self.Nx * self.dkx)
        self.dz = 2 * np.pi / (self.Nz * self.dkz)
        self.caseDir = caseDir

    def add_var(self, name_, value_):
        self.var_dict[name_] = value_

    def getvar(self, name_):
        return self.var_dict.get(name_, f'key {name_} not found')

    def loadUmean(self):
        data = np.loadtxt(
            os.path.join(self.caseDir, 'RESULTS_COV',
                         'UMEAN-20000-70000-per-10.H5'))
        # print(data)
        # print(np.shape(data))
        self.add_var('Umean', 0.5 * (data + np.flip(data)))

    def load_velocity(self, ITB, ITE):
        # This function loads the velocity Fourier mode for specifed wavenumbers
        # And transform it from Chebyshev space to physical space
        data = np.load(
            os.path.join(self.caseDir, 'PROBE_VELOCITY',
                         f'VEL-{ITB}-{ITE}-kx-{self.kxid}-kz-{self.kzid}.npy'))
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        vel = np.einsum('ij,zjk->zik',
                        CF.s2p_y_op(self.Ny),
                        data,
                        optimize=True)
        self.add_var('vel', vel)

    def calculate_shear_stress(self, overlap=0.75):
        if not ('vel' in self.var_dict):
            raise ValueError('\'vel\' empty, please load velocity')
        vel = self.getvar('vel')
        _, vel_fft = WelchFFT.Welch_FFT(vel,
                                        dt=self.dt,
                                        seg_size=self.Nt_window,
                                        overlap=overlap)
        print(np.shape(vel_fft))
        uv_fft = np.real(
            np.mean(vel_fft[:, :, :, 0] * vel_fft[:, :, :, 1].conj(), axis=0))
        uw_fft = np.real(
            np.mean(vel_fft[:, :, :, 0] * vel_fft[:, :, :, 2].conj(), axis=0))
        vw_fft = np.real(
            np.mean(vel_fft[:, :, :, 1] * vel_fft[:, :, :, 2].conj(), axis=0))

        uv_fft = 0.5 * uv_fft - 0.5 * np.flip(uv_fft, axis=1)
        uw_fft = 0.5 * uw_fft + 0.5 * np.flip(uw_fft, axis=1)
        vw_fft = 0.5 * vw_fft - 0.5 * np.flip(vw_fft, axis=1)

        uv_fft /= (self.domega * self.dkx * self.dkz)
        uw_fft /= (self.domega * self.dkx * self.dkz)
        vw_fft /= (self.domega * self.dkx * self.dkz)

        self.add_var('uv_psd', uv_fft)
        self.add_var('uw_psd', uw_fft)
        self.add_var('vw_psd', vw_fft)

    def calculate_velocity_psd(self, overlap=0.75):
        if not ('vel' in self.var_dict):
            raise ValueError('\'vel\' empty, please load velocity')
        vel = self.getvar('vel')
        _, vel_fft = WelchFFT.Welch_FFT(vel,
                                        dt=self.dt,
                                        seg_size=self.Nt_window,
                                        overlap=overlap)
        vel_psd = np.real(np.mean(vel_fft * np.conjugate(vel_fft), axis=0))
        vel_psd = 0.5 * (vel_psd + np.flip(vel_psd, axis=1))
        vel_psd /= (self.domega * self.dkx * self.dkz)
        self.add_var('vel_psd', vel_psd)

    def calculate_velocity_csd(self, overlap=0.75):
        if not ('vel' in self.var_dict):
            raise ValueError('\'vel\' empty, please load velocity')
        vel = self.getvar('vel')
        t1 = time.time()
        _, vel_fft = WelchFFT.Welch_FFT(vel,
                                        dt=self.dt,
                                        seg_size=self.Nt_window,
                                        overlap=overlap)
        t2 = time.time()
        print(f'FFT done, using time: {t2-t1:.4e}')

        # reshape vel_fft[N_ens,N_freq,N_y,3] to vel_fft[N_ens,N_freq,N_y*3]
        vel_fft_reshaped = vel_fft.reshape(vel_fft.shape[0],
                                           vel_fft.shape[1],
                                           -1,
                                           order='F')
        t3 = time.time()
        print(f'reshape done, using time: {t3-t2:.4e}')
        N_ens = np.shape(vel_fft)[0]
        vel_csd = np.einsum('ijl,ijn->jln',
                            vel_fft_reshaped,
                            np.conj(vel_fft_reshaped),
                            optimize=True) / N_ens
        t4 = time.time()
        print(f'csd done, using time: {t4-t3:.4e}')
        vel_csd /= (self.domega * self.dkx * self.dkz)
        self.add_var('vel_csd', vel_csd)

    def velocity_csd_component(self, component='xx'):
        if not ('vel_csd' in self.var_dict):
            raise ValueError('\'vel_csd\' empty, please process velocity csd')
        if component not in [
                'xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz'
        ]:
            raise ValueError('\'component\' {component} not valid')

        return csd_component(self.getvar('vel_csd'), component)

    def load_pressure(self, ITB, ITE):
        # This function loads the velocity Fourier mode for specifed wavenumbers
        # And transform it from Chebyshev space to physical space
        # Validate inputs
        if not isinstance(ITB, int) or not isinstance(ITE, int):
            raise ValueError("ITB and ITE must be integers")
        if ITB > ITE:
            raise ValueError("ITB must be less than or equal to ITE")

        # Construct file path
        filename = f'P-{ITB}-{ITE}-kx-{self.kxid}-kz-{self.kzid}.npy'
        filepath = os.path.join(self.caseDir, 'PROBE_PRESSURE', filename)

        try:
            # Load data
            data = np.load(filepath)
            print('pressure probe data shape', np.shape(data))
            # Validate data shape
            if data.shape[1] != self.Ny:
                raise ValueError(
                    f'Y points mismatch: expected {self.Ny}, got {data.shape[1]}'
                )
            # Add to variables
            self.add_var('p', data)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Pressure data file not found at: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error loading pressure data: {str(e)}")

    def calculate_pressure_psd(self, overlap=0.75):
        if not ('p' in self.var_dict):
            raise ValueError('\'p\' empty, please load velocity')
        p = self.getvar('p')
        # print(np.shape(p))
        # print(np.shape(p))
        _, p_fft = WelchFFT.Welch_FFT(p,
                                      dt=self.dt,
                                      seg_size=self.Nt_window,
                                      overlap=overlap)
        # print(np.shape(p_fft))
        p_psd = np.real(np.mean(p_fft * np.conjugate(p_fft), axis=0))
        # print(np.shape(p_psd))
        p_psd = 0.5 * (p_psd + np.flip(p_psd, axis=1))
        p_psd /= (self.domega * self.dkx * self.dkz)
        self.add_var('p_psd', p_psd)

    def calculate_pressure_csd(self, overlap=0.75):
        if not ('p' in self.var_dict):
            raise ValueError('\'p\' empty, please load velocity')
        p = self.getvar('p')
        t1 = time.time()
        _, p_fft = WelchFFT.Welch_FFT(p,
                                      dt=self.dt,
                                      seg_size=self.Nt_window,
                                      overlap=overlap)
        t2 = time.time()
        print(f'FFT done, using time: {t2-t1:.4e}')

        # reshape vel_fft[N_ens,N_freq,N_y,3] to vel_fft[N_ens,N_freq,N_y*3]
        p_fft_reshaped = p_fft.reshape(p_fft.shape[0],
                                       p_fft.shape[1],
                                       -1,
                                       order='F')
        t3 = time.time()
        print(f'reshape done, using time: {t3-t2:.4e}')
        N_ens = np.shape(p_fft)[0]
        p_csd = np.einsum('ijl,ijn->jln',
                          p_fft_reshaped,
                          np.conj(p_fft_reshaped),
                          optimize=True) / N_ens
        t4 = time.time()
        print(f'csd done, using time: {t4-t3:.4e}')
        p_csd /= (self.domega * self.dkx * self.dkz)
        self.add_var('p_csd', p_csd)

    def load_sweep_velocity(self):
        data_path = os.path.join(self.caseDir, "./RESULTS_VEL_RMS/DNS-rms.dat")
        uu, vv, ww = np.loadtxt(data_path,
                                usecols=[1, 2, 3],
                                skiprows=1,
                                unpack=True)
        self.add_var('Vx', np.sqrt(uu))
        self.add_var('Vy', np.sqrt(vv))
        self.add_var('Vz', np.sqrt(ww))

    def load_lambda_y(self):
        self.add_var(
            'lambda_y',
            np.load(os.path.join(self.caseDir,
                                 './RESULTS_LAMBDA/LAMBDA_Y.npy')))

    def tke(self, vel_psd_):
        # check vel_psd_ shape is [self.Nt_window,self.Ny,3]
        # Check if the shape of vel_psd_ is correct
        assert vel_psd_.shape == (self.Nt_window, self.Ny, 3), (
            f"Expected shape ({self.Nt_window}, {self.Ny}, 3), but got {vel_psd_.shape}"
        )
        weight = integral_weights(self.Ny)
        vel_psd_freq_intg = np.sum(vel_psd_, axis=0) * self.domega
        tke = np.sum(np.sum(vel_psd_freq_intg, axis=1) * weight)
        return tke  #scalar

    def tke_flat(self, vel_psd_):
        assert vel_psd_.shape == (self.Nt_window, self.Ny * 3), (
            f"Expected shape ({self.Nt_window}, {self.Ny*3}), but got {vel_psd_.shape}"
        )
        weight = integral_weights(self.Ny)
        weights = np.concatenate([weight, weight, weight])
        # print(np.shape(weights), weights)
        vel_psd_freq_intg = np.sum(vel_psd_, axis=0) * self.domega
        tke = np.sum(vel_psd_freq_intg * weights)
        return np.real(tke)

    def energy_norm(self, psd_):
        # check vel_psd_ shape is [self.Nt_window,self.Ny]
        assert psd_.shape == (self.Nt_window, self.Ny), (
            f"Expected shape ({self.Nt_window}, {self.Ny}), but got {psd_.shape}"
        )
        weight = integral_weights(self.Ny)
        psd_freq_intg = np.sum(psd_, axis=0) * self.domega
        norm = np.sum(psd_freq_intg * weight)
        return norm  #scalar

    def energy_norm_on_freq(self, psd_):
        # check vel_psd_ shape is [self.Nt_window,self.Ny]
        assert psd_.shape == (self.Nt_window, self.Ny), (
            f"Expected shape ({self.Nt_window}, {self.Ny}), but got {psd_.shape}"
        )
        weight = integral_weights(self.Ny)
        psd_norm = np.einsum('ij,j->i', psd_, weight, optimize='True')
        return psd_norm  # on frequency

    def tke_on_freq(self, psd_):
        # check vel_psd_ shape is [self.Nt_window,self.Ny,3]
        assert psd_.shape == (self.Nt_window, self.Ny, 3), (
            f"Expected shape ({self.Nt_window}, {self.Ny}, 3), but got {psd_.shape}"
        )
        weight = integral_weights(self.Ny)
        psd_norm = np.einsum('ijk,j->ik', psd_, weight, optimize='True')
        return np.sum(psd_norm, axis=1)  # on frequency

    def weight_sum(self, psd_, w_ax=1, s_ax=[]):
        weight = integral_weights(self.Ny)
        if psd_.shape[w_ax] == self.Ny:
            pass
        elif psd_.shape[w_ax] == self.Ny * 3:
            weight = np.concatenate([weight, weight, weight])
        else:
            print(f'wrong dimension in {w_ax}')
            return None  # Explicitly return None for error case

        # Reshape weight for proper broadcasting
        weight = np.expand_dims(weight,
                                axis=tuple(i for i in range(psd_.ndim)
                                           if i != w_ax))

        # Calculate weighted sum (this reduces dimensionality by 1)
        psd_weighted_sum = np.sum(psd_ * weight, axis=w_ax)

        # Adjust s_ax indices for the reduced array
        if s_ax:
            # Need to handle cases where s_ax might be after w_ax
            adjusted_s_ax = [ax if ax < w_ax else ax - 1 for ax in s_ax]
            psd_weighted_sum = np.sum(psd_weighted_sum,
                                      axis=tuple(adjusted_s_ax))

        return psd_weighted_sum

        # def process_Velocity(self, overlap=0.5):

    #     if not ('vel' in self.var_dict):
    #         print('\'vel\' empty, please load velocity')
    #         return
    #     vel = self.getvar('vel')

    #     _, vel_fft = WelchFFT.Welch_FFT(vel,
    #                                     dt=self.dt,
    #                                     seg_size=self.Nt_window,
    #                                     overlap=overlap)
    #     N_ens = np.shape(vel_fft)[0]
    #     # calculate cross spectral density
    #     vel_csd = np.einsum('ijkl,ijnl->jknl', vel_fft,
    #                         np.conj(vel_fft)) / N_ens
    #     vel_psd = np.mean(vel_hat * np.conjugate(vel_hat), axis=0)
    #     vel_psd = 0.5 * (vel_psd + np.flip(vel_psd, axis=1))
    #     self.add_var('u_fft_ens', vel_hat[:, :, :, 0])
    #     self.add_var('v_fft_ens', vel_hat[:, :, :, 1])
    #     self.add_var('w_fft_ens', vel_hat[:, :, :, 2])
    #     self.add_var('u_psd', vel_psd[:, :, 0])
    #     self.add_var('v_psd', vel_psd[:, :, 1])
    #     self.add_var('w_psd', vel_psd[:, :, 2])

    def process_Force(self, ITB, ITE):
        data = np.load(
            os.path.join(self.caseDir, 'PROBE_FORCE',
                         f'F-{ITB}-{ITE}-kx-{self.kxid}-kz-{self.kzid}.npy'))
        assert data.shape[1] == self.Ny, 'Y points mismatch'
        f = np.einsum('ij,zjk->zik', CF.s2p_y_op(self.Ny), data, optimize=True)
        _, f_hat = WelchFFT.Welch_FFT(f, dt=self.dt, seg_size=self.Nt_window)
        stf_hat = f_hat[:, :, :, 0:3] - f_hat[:, :, :, 3:6]
        f_psd = np.mean(f_hat * np.conjugate(f_hat), axis=0)
        f_psd = 0.5 * (f_psd + np.flip(f_psd, axis=1))
        stf_psd = np.mean(stf_hat * np.conjugate(stf_hat), axis=0)
        stf_psd = 0.5 * (stf_psd + np.flip(stf_psd, axis=1))

        self.add_var('nlfx_fft_ens', f_hat[:, :, :, 0])
        self.add_var('nlfy_fft_ens', f_hat[:, :, :, 1])
        self.add_var('nlfz_fft_ens', f_hat[:, :, :, 2])

        self.add_var('edfx_fft_ens', f_hat[:, :, :, 3])
        self.add_var('edfy_fft_ens', f_hat[:, :, :, 4])
        self.add_var('edfz_fft_ens', f_hat[:, :, :, 5])

        self.add_var('stfx_fft_ens', stf_hat[:, :, :, 0])
        self.add_var('stfy_fft_ens', stf_hat[:, :, :, 1])
        self.add_var('stfz_fft_ens', stf_hat[:, :, :, 2])

        self.add_var('nlfx_psd', f_psd[:, :, 0])
        self.add_var('nlfy_psd', f_psd[:, :, 1])
        self.add_var('nlfz_psd', f_psd[:, :, 2])

        self.add_var('edfx_psd', f_psd[:, :, 3])
        self.add_var('edfy_psd', f_psd[:, :, 4])
        self.add_var('edfz_psd', f_psd[:, :, 5])

        self.add_var('stfx_psd', stf_psd[:, :, 0])
        self.add_var('stfy_psd', stf_psd[:, :, 1])
        self.add_var('stfz_psd', stf_psd[:, :, 2])
