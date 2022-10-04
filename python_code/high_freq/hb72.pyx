cimport cython
import time
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound, nonecheck, cdivision
import scipy.special as sps
from scipy import interpolate
from cython.parallel import prange
import multiprocessing
from cython.parallel import prange
from joblib import Parallel, delayed

#DBL = np.double ctypedef
#np.double_t DBL_C

cdef extern from "complex.h":
    pass

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef buoy_anamoly_1d(np.ndarray[np.float64_t,ndim=1] Y):

    cdef:
        np.ndarray[np.float64_t,ndim=1] b0
        np.ndarray[np.float64_t,ndim=1] db0

    cdef double ampl = -0.06
    cdef double delta = 5.e-6
    b0  = ampl*sps.erf(delta * Y) 

    cdef double coeff = ampl*2.*delta/( np.sqrt(np.pi) )
    db0 = coeff*np.exp(-delta*delta*Y**2) 
    
    return b0, db0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef buoy_anamoly_2d(np.ndarray[np.float64_t,ndim=2] Y):

    cdef:
        np.ndarray[np.float64_t,ndim=2] b0

    cdef double ampl = -0.06
    cdef double delta = 5.e-6
    b0  = ampl*sps.erf(delta * Y) 
    
    return b0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sol_in_physical_space(np.ndarray[np.complex128_t, ndim=2] A_fourier, int n_max, 
    int parity, np.ndarray[np.float64_t,ndim=1] y,  np.ndarray[np.float64_t,ndim=1] z, double H):
    
    cdef:
        np.ndarray[np.float64_t,ndim=2] A_physical_space
        np.ndarray[np.int_t,ndim=1] vmode
        np.ndarray[np.float64_t,ndim=1] store_
        np.ndarray[np.float64_t,ndim=1] A_invfft

    vmode = np.arange(1, n_max, dtype=np.int)
    cdef int imax = len(y)
    cdef int jmax = len(z)
    A_physical_space = np.zeros( (imax, jmax) )
    
    if parity == 0:  # belongs to sine mode
        for j in range(n_max-1): 
            A_invfft = np.real( np.fft.ifft( A_fourier[:,j] ) )
            store_ = np.sin( vmode[j]*np.pi*z/H )
            for i in range( imax ):
                A_physical_space[i,:] += A_invfft[i] * store_
    
    if parity == 1:  # belongs to cosine mode 
        for j in range(n_max-1): 
            A_invfft = np.real( np.fft.ifft( A_fourier[:,j] ) )
            store_ = np.cos( vmode[j]*np.pi*z/H )
            for i in range( imax ):
                A_physical_space[i,:] += A_invfft[i] * store_
        
    return A_physical_space   

# fourier y-derivative
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef fourier_yderiv(double[:,::1] var2d, double[::1] y):

    cdef:
        np.ndarray[np.float64_t,ndim=1] l
        np.ndarray[np.complex128_t, ndim=2] var_ifft
        np.ndarray[np.float64_t, ndim=2] var2d_yderiv
    
    cdef np.complex128_t varc128 = 1j

    cdef int ny = np.shape(var2d)[0]
    cdef int nz = np.shape(var2d)[1]
    l = 2*np.pi*np.fft.fftfreq(ny, d=y[2]-y[1])

    # inverse FFT along y-direction
    var_ifft = np.fft.fft(var2d, axis=0, norm=None)

    # FFT derivtaive
    for kt in range(nz):
        var_ifft[:,kt] = varc128 * l *var_ifft[:,kt]
    
    var2d_yderiv = np.real( np.fft.ifft(var_ifft, axis=0, norm=None) )
    return var2d_yderiv


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sincos_zderiv(np.ndarray[np.float64_t,ndim=2] var2d, int n_max, double H, 
    np.ndarray[np.float64_t,ndim=2] z, int parity):

    cdef:
        np.ndarray[np.int_t, ndim=1] vmode
        np.ndarray[np.float64_t, ndim=2] Az
        np.ndarray[np.float64_t, ndim=2] tmp_

    cdef int ny = np.shape(var2d)[0]
    cdef int nz = np.shape(var2d)[1]

    vmode = np.arange(1, n_max, 1)
    Az = np.zeros(np.shape(var2d))
    
    if parity == 0:
        for it in vmode:
            tmp_ = var2d * np.sin(it*np.pi*z/H)
            An = np.trapz(tmp_, z[0,:], axis=1) * 2./H * (it*np.pi/H)
            Az += An[...,None]*np.cos(it*np.pi*z/H) * it*np.pi/H
                    
    elif parity == 1:
        for it in vmode:
            tmp_ = var2d * np.cos(it*np.pi*z/H)
            An = np.trapz(tmp_, z[0,:], axis=1)* 2./H* (it*np.pi/H)
            Az += An[...,None]*np.sin(it*np.pi*z/H) * it*np.pi/H

    return Az


# calculating z-derivative using fdm
@cython.boundscheck(False)
@cython.wraparound(False)
cdef fdm_ddz_(np.ndarray[np.float64_t,ndim=2] A, double[::1] z):

    cdef int ny = np.shape(A)[0]
    cdef int nz = np.shape(A)[1]
    cdef int it

    cdef:
        np.ndarray[np.float64_t, ndim=2] Az
        np.ndarray[np.float64_t, ndim=1] dz
        np.ndarray[np.float64_t, ndim=1] dA_it_mius_half
        np.ndarray[np.float64_t, ndim=1] dA_it_plus_half

    Az = np.zeros( np.shape(A) )
    dz = np.zeros(nz)
    
    for it in range(nz-1): dz[it] = abs( z[it+1] - z[it] )
    dz[nz-1] = dz[1]
    
    for it in range( 1, nz-1 ):
        dA_it_mius_half = ( A[:,it] - A[:,it-1] )/dz[it-1]
        dA_it_plus_half = ( A[:,it+1] - A[:,it] )/dz[it]
        
        Az[:,it] = ( dz[it-1]*dA_it_plus_half + dz[it]*dA_it_mius_half )/( dz[it]+dz[it-1] )
    
    Az[:,0] = ( A[:,1] - A[:,0] )/abs( z[1] - z[0] )
    Az[:,nz-1] = ( A[:,nz-1] - A[:,nz-2] )/abs( z[nz-1] - z[nz-2] )
    
    return Az                    


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef hb72_sol(int n_max, double time_, np.ndarray[np.float64_t,ndim=1] y_numeric, 
    np.ndarray[np.float64_t,ndim=1] z_numeric, np.float64_t f, np.float64_t alpha, np.float64_t N, np.float64_t H):

    cdef:
        np.ndarray[np.int_t,ndim=1] n
        np.ndarray[np.float64_t,ndim=1] m_n

    cdef np.complex128_t varc128 = 1j
    cdef np.float64_t N2 = N*N

    # total number of vertical mode
    n = np.arange(1, n_max, 1)
    m_n = n*np.pi/H 

    cdef:
        np.ndarray[np.float64_t,ndim=1] y_gmc
        np.ndarray[np.float64_t,ndim=1] z_gmc
        np.ndarray[np.float64_t,ndim=1] l 

    cdef np.float64_t Ly = 6000.e3
    cdef np.int_t ny = 6000

    # defining generalised momentum coordinates
    y_gmc = np.linspace( -Ly/2, Ly/2, ny )
    cdef np.float64_t dy = y_gmc[2]-y_gmc[1]
    #print('dy: %f' %(dy))
    z_gmc = z_numeric 

    # y-wavenumber   
    l = 2.*np.pi*np.fft.fftfreq(ny, d=dy)

    cdef:
        np.ndarray[np.float64_t,ndim=2] L_
        np.ndarray[np.float64_t,ndim=2] M_n
        np.ndarray[np.float64_t,ndim=1] An
        
    L_, M_n = np.meshgrid(l, m_n, indexing='ij')
    An = -2.*H*(-1 + (-1)**n)/(n*np.pi)**2

    cdef:
        np.ndarray[np.float64_t,ndim=1] db0
        np.ndarray[np.complex128_t,ndim=1] db0_hat
        np.ndarray[np.complex128_t,ndim=2] _DBO_hat 
        np.ndarray[np.float64_t,ndim=2] AN_ 

    _, db0 = buoy_anamoly_1d(y_gmc)
    db0_hat = np.fft.fft( db0 )

    _DBO_hat, AN_ = np.meshgrid(db0_hat, An, indexing='ij')

    cdef:
        np.ndarray[np.float64_t,ndim=2] denom_ 
        np.ndarray[np.complex128_t,ndim=2] numer_ 
        np.ndarray[np.complex128_t,ndim=2] u_hat
        np.ndarray[np.complex128_t,ndim=2] ut_hat
        np.ndarray[np.float64_t,ndim=2] tmp
        np.ndarray[np.float64_t,ndim=2] u_phys

    # In HB72 soluton, α is small, hence α/f << 1 (thus neglected)
    denom_ = f**2.*np.ones(np.shape(L_)) + N2*( L_*np.exp(alpha*time_)/M_n )**2.
    numer_ = -AN_*f*_DBO_hat*np.exp(alpha*time_)
    u_hat = numer_/denom_

    cdef int cos_ = 1
    cdef int sin_ = 0  

    # 1 → cosine mode, 0 → sine mode
    u_phys = sol_in_physical_space(u_hat, n_max, cos_, y_gmc, z_gmc, H)
    print('max/min value of u_phys: %f %f ' %(u_phys.max(), u_phys.min()))

    tmp = N2 * (L_/M_n)**2. * np.exp( 2.*alpha*time_ )
    ut_hat = ( f**2. - tmp )/( f**2. + tmp ) * alpha * u_hat

    cdef:
        np.ndarray[np.complex128_t,ndim=2] v_hat 
        np.ndarray[np.complex128_t,ndim=2] w_hat
        np.ndarray[np.complex128_t,ndim=2] b_hat
        np.ndarray[np.float64_t,ndim=2] v_phys
        np.ndarray[np.float64_t,ndim=2] w_phys
        np.ndarray[np.float64_t,ndim=2] b_phys


    ####### here everything on generalized momentum coordinate system
    # calculating v-velocity 
    v_hat = ( ut_hat + alpha*u_hat )/f
    v_phys = sol_in_physical_space(v_hat, n_max, cos_, y_gmc, z_gmc, H)

    # calculating w-velocity
    w_hat = -varc128 * L_ * v_hat/M_n 
    w_phys = sol_in_physical_space(w_hat, n_max, sin_, y_gmc, z_gmc, H)

    # calculating Δb
    b_hat = u_hat * N2/f * varc128 * np.exp(alpha*time_) * L_/M_n
    b_phys = sol_in_physical_space(b_hat, n_max, sin_, y_gmc, z_gmc, H)

    cdef:
        np.ndarray[np.float64_t,ndim=2] Y
        np.ndarray[np.float64_t,ndim=2] Z

    Y, Z = np.meshgrid(y_gmc, z_gmc, indexing='ij')
    b_phys = b_phys + buoy_anamoly_2d(Y)

#### now converting everything to Eulerian coordinates
    Y = Y*np.exp(-alpha*time_) + u_phys/f

    cdef:
        np.ndarray[np.float64_t,ndim=2] uY_phys
        np.ndarray[np.float64_t,ndim=2] vY_phys
        np.ndarray[np.float64_t,ndim=2] bY_phys

        np.ndarray[np.float64_t,ndim=2] uZ_phys
        np.ndarray[np.float64_t,ndim=2] vZ_phys

        np.ndarray[np.float64_t,ndim=2] uy_phys
        np.ndarray[np.float64_t,ndim=2] vy_phys
        np.ndarray[np.float64_t,ndim=2] by_phys

        np.ndarray[np.float64_t,ndim=2] Jacobian

    uY_phys = fourier_yderiv(u_phys, y_gmc)
    uZ_phys = fdm_ddz_(u_phys, z_gmc)
    #uZ_phys = sincos_zderiv(u_phys, n_max, H, Z, cos_) 

    Jacobian = np.exp(alpha*time_)/( 1. + np.exp(alpha*time_)/f*uY_phys ) 
    print( 'max absolute value of the jacobian: ', np.max(abs(Jacobian)) )
    w_phys = w_phys*Jacobian
    v_phys = v_phys + w_phys/f*uZ_phys

    #bY_phys = fourier_yderiv( b_phys, y_gmc )
    #vY_phys = fourier_yderiv( v_phys, y_gmc )
    #wY_phys = fourier_yderiv( w_phys, y_gmc )
    #wZ_phys = sincos_zderiv( w_phys, n_max, H, Z, cos_)

    #uy_phys = Jacobian*uY_phys
    #vy_phys = Jacobian*vY_phys
    #by_phys = Jacobian*bY_phys

    return Y, Z, u_phys, v_phys, w_phys, b_phys #, uy_phys, vy_phys, by_phys

