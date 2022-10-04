"""
Dedalus script: 2D - frontogenesis (governed by HB72 model) + NIWs -> interaction
"""

import chunk
import numpy as np
from mpi4py import MPI
from scipy.special import erf
from scipy import interpolate
import time
import h5py
import pathlib
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.tools import post
from dedalus.core.operators import GeneralFunction
import logging
logger = logging.getLogger(__name__)
from joblib import Parallel, delayed

import hb72 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def interp_to_numericGrid_par(var, y, ynG):

    interp_ = interpolate.interp1d(y, var, kind='cubic', fill_value='extrapolate')
    unG_t = interp_(ynG).T

    return unG_t.T


def get_wvel(V, y, z):

    l = 2*np.pi*np.fft.fftfreq(len(y), d=y[2]-y[1])
    Vy = hb72.fourier_yderiv(V, y)

    print('max/min of V: ', V.max(), V.min())
    print('max/min of Vy: ', Vy.max(), Vy.min())
    Wz = - Vy # by using ∂yV+∂zW = 0

    W = np.zeros(np.shape(V))
    for jt in range(len(z)):
        if jt ==0: 
            W[:,jt] = 0.
        elif jt ==len(z)-1: 
            W[:,jt] = 0.
        else:
            W[:,jt] = np.trapz(Wz[:,0:jt+1], z[0:jt+1]) 

    Y, Z =  np.meshgrid(y, z, indexing='ij')
    # check divergence
    div = Vy + hb72.sincos_zderiv( W, 101, 1.e3, Z, 0)
    print('min/max value of ∇⋅u: ', div.min(), div.max())

    return Wz

# calculate geostrophic velocity U from `Thermal Wind' balance
# ∂U/∂z = -1/f*∂B/∂y
def cal_U_frm_TWB(By_, f, y, z, H):

    ny, nz = len(y[:,0]), len(z[0,:])
    n = np.arange(1, 101, 1)
    An = -2.*H*( -1 + (-1)**n )/( n*np.pi )**2.

    print('shape of By: ', np.shape(By_))

    an = np.zeros( (ny, len(n)) )
    for it in range(ny):
        for jt in range(len(n)):
            an[it,jt] = -1/f * By_[it,0] * An[jt]

    U_ = np.zeros((ny,nz))
    for it in range(ny):
        for jt in range(len(n)):
            U_[it,:] += an[it,jt] * np.cos(n[jt]*np.pi*z[0,:]/H)
    
    return U_

n_core = 8

# Basic Parameters
L, Lz = (500.e3, 1.e3)
H = Lz

# Richardson number
Ri = 15.
# Coriolis parameters
f = 1.e-4
# Vertical stratification (N^2 = ∂B/∂z)
N = f*100.; N2 = N*N
# Horizontal buoyancy gradient (S^2 = ∂B/∂y)
S = (f**2*N**2/Ri)**(1/4)
S2 = S*S

# Strain value
α = 0.1*f

# vertical mode number of IW
m = 1.*np.pi/H

# frequency of IW
ω = 1.5*f #f*np.sqrt(1. - 1./Ri)*1.001

# horizontal wavenumber of IW
k = m*(ω*ω - f*f)/np.sqrt( S2**2. + N**2.*(ω*ω-f*f) )
Ly = 2*np.pi/abs(k)

# Non-dimensional variable used in IW initial condition (see the paper)
Bs = S2/(f*f-ω*ω)

# parameters of initially imposed buoyancy anamoly
ampl = -0.06
δ = 5e-6
coeff = ampl*2.*δ/( np.sqrt(np.pi) )

# Create bases and domain
y_basis = de.Fourier('y',  2500, interval=(-L, L)) # , dealias=3/2)
z_basis = de.SinCos('z', 180, interval=(-Lz, 0)) #,  dealias=3/2)
domain = de.Domain([y_basis, z_basis], grid_dtype=np.float64)

problem=de.IVP(domain, variables=['p','u','v','w','b'], time='t')

problem.meta['b', 'w']['z']['parity'] = -1 
problem.meta['p', 'u', 'v']['z']['parity'] = 1

y = domain.grid(0)
z = domain.grid(1)

problem.parameters['ν'] = 2.e-4
problem.parameters['α'] = α
problem.parameters['f'] = f
problem.parameters['νhw'] = 5.e7


problem.parameters['ampl'] = ampl = -0.06
problem.parameters['N2'] = N*N

B = domain.new_field()
B.meta['z']['parity'] = -1
B['g'] = ampl*erf(δ * y)   
problem.parameters['B'] = B

By = domain.new_field()
By.meta['z']['parity'] = -1
By['g'] = coeff*np.exp(-δ**2*y**2) 
problem.parameters['By'] = By

U = domain.new_field()
U.meta['z']['parity'] = 1
U['g'] = cal_U_frm_TWB(By['g'], f, y, z, H) 
problem.parameters['U'] = U

V = domain.new_field()
V.meta['z']['parity'] = 1
V['g'] = 0. 
problem.parameters['V'] = V  

W = domain.new_field()
W.meta['z']['parity'] = -1
W['g'] = 0. 
problem.parameters['W'] = W

problem.substitutions['Δ(β,ν)']    = "ν*d(β, y=2)"
problem.substitutions['Δ2(β,νh)']  = "νh*d(β, y=4)" 
problem.substitutions['D(β,ν,νh)'] = "Δ(β,ν) - Δ2(β,νh)"
problem.substitutions['RS(ψy,ψz,F)'] = "ψy*dy(F) + ψz*dz(F)"
problem.substitutions['Σ(α,F)'] = "- α*y*dy(F)"

# equations for the wave part
problem.add_equation("dy(v) + dz(w) = 0", condition = "(ny != 0) or (nz != 0)")
problem.add_equation("dt(u) - ν*dz(dz(u)) - D(u,ν,νhw) - f*v + α*u = - RS(V,W,u) - Σ(α,u)")
problem.add_equation("dt(v) - ν*dz(dz(v)) - D(v,ν,νhw) + dy(p) + f*u - α*v = - RS(V,W,v) - RS(v,w,V) - Σ(α,v)")
problem.add_equation("dz(p) - b = 0")
problem.add_equation("dt(b) - ν*dz(dz(b)) - D(b,ν,νhw) + N2*w = - RS(V,W,b) - RS(v,w,B) - Σ(α,b)")
problem.add_equation("p  = 0", condition="(nz == 0) and (ny == 0)")
#####

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF2)
logger.info('Solver built')

# Initial conditions for the wave-part
u = solver.state['u']
v = solver.state['v']
w = solver.state['w']
b = solver.state['b']

w_ampl = 0.1

# initial condition for the wave-part
u['g'] = np.real( w_ampl*1j/(ω*f)*( f**2*( (m*np.cos(m*z) - 1j*k*Bs*np.sin(m*z))*np.exp(1j*k*(y-Bs*z)) ) 
     + S2*1j*np.sin(m*z)*k*np.exp(1j*k*(y-Bs*z)) ) )

v['g'] = np.real( w_ampl*(m*np.cos(m*z) - 1j*k*Bs*np.sin(m*z))*np.exp(1j*k*(y-Bs*z)) )

w['g'] = np.real( w_ampl*-1j*np.sin(m*z)*k*np.exp(1j*k*(y-Bs*z)) )

b['g'] = np.real( 1j/ω*( S2*w_ampl*(m*np.cos(m*z) - 1j*k*Bs*np.sin(m*z))*np.exp(1j*k*(y-Bs*z)) 
  - N2*w_ampl*-1j*np.sin(m*z)*k*np.exp(1j*k*(y-Bs*z)) ) )


u['g'] *= np.exp( -( (y/(1.5*Ly))**2 ) )
v['g'] *= np.exp( -( (y/(1.5*Ly))**2 ) )
w['g'] *= np.exp( -( (y/(1.5*Ly))**2 ) )
b['g'] *= np.exp( -( (y/(1.5*Ly))**2 ) )

# Timestepping and output
dt = 20. 
T = 2*np.pi/f

# Integration parameters
solver.stop_sim_time = 3.*T
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
flows = solver.evaluator.add_file_handler('flow', sim_dt=0.01*T, max_writes=100)
flows.add_task("U", name='U')
flows.add_task("V", name='V')
flows.add_task("W", name='W')
flows.add_task("B", name='B')
flows.add_task("dy(U)", name='Uy')
flows.add_task("dy(V)", name='Vy')
flows.add_task("dy(B)", name='By')

waves = solver.evaluator.add_file_handler('wave', sim_dt=0.01*T, max_writes=100)
waves.add_task("u", name='u')
waves.add_task("v", name='v')
waves.add_task("w", name='w')
waves.add_task("b", name='b')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=20)
flow.add_property("0.5*sqrt(U*U + V*V + W*W)", name='KE')
flow.add_property("dy(V)/f", name='div')
flow.add_property("sqrt(dy(B)*dy(B))/N2", name='slope')

wave = flow_tools.GlobalFlowProperty(solver, cadence=10)
wave.add_property("0.5*sqrt(u*u + v*v + w*w)", name='KE')

# Main loop
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        solver.step(dt)
        if (solver.iteration-1) % 20 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max flow KE = %f' %flow.max('KE'))
            logger.info('Max wave KE = %f' %wave.max('KE'))
            logger.info('Max flow divergence = %f' %flow.min('div'))
            logger.info('Max slope = %f' %flow.max('slope'))

        Y_euler, Z_euler, u_phys, v_phys, w_phys, b_phys = \
        hb72.hb72_sol(101, solver.sim_time, y[:,0], z[0,:], f, α, N, 1.e3)
        print('max/min values of Y_euler: ', Y_euler.max(), Y_euler.min())

        if solver.iteration-1 == 0: 
            Yg, Zg = np.meshgrid(y[:,0], z[0,:], indexing='ij')
            nz_ = len(z[0,:])
            print('shape of Y_euler: ', np.shape(Y_euler))

        tmp_ = Parallel(n_jobs=n_core)(delayed(interp_to_numericGrid_par)(u_phys[:,it], Y_euler[:,it], y[:,0]) \
        for it in range(nz_))
        U['g'] = np.array(tmp_).T

        tmp_ = Parallel(n_jobs=n_core)(delayed(interp_to_numericGrid_par)(v_phys[:,it], Y_euler[:,it], y[:,0]) \
        for it in range(nz_))
        V['g'] = np.array(tmp_).T

        tmp_ = Parallel(n_jobs=n_core)(delayed(interp_to_numericGrid_par)(w_phys[:,it], Y_euler[:,it], y[:,0]) \
        for it in range(nz_))
        W['g'] =  np.array(tmp_).T 

        tmp_ = Parallel(n_jobs=n_core)(delayed(interp_to_numericGrid_par)(b_phys[:,it], Y_euler[:,it], y[:,0]) \
        for it in range(nz_))
        B['g'] =  np.array(tmp_).T 


except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))
