#!/usr/bin/env python
import numpy as np
import scipy.linalg

def state_parser(Ns,Ds,Nv,Nx,Dx):
  out = {}

  out['Ns'] = Ns
  out['Ds'] = Ds
  out['Nv'] = Nv
  out['Nx'] = Nx
  out['Dx'] = Dx
  out['total'] = 3*Ns*Ds + Nv + 2*Nx*Dx

  slip_integral = range(0,Ns*Ds)
  out['slip_integral'] = np.reshape(slip_integral,(Ns,Ds))

  slip = range(Ns*Ds,2*Ns*Ds)
  out['slip'] = np.reshape(slip,(Ns,Ds))

  slip_derivative = range(2*Ns*Ds,3*Ns*Ds)
  out['slip_derivative'] = np.reshape(slip_derivative,(Ns,Ds))

  out['fluidity'] = range(3*Ns*Ds,3*Ns*Ds + Nv)

  secular_velocity = range(3*Ns*Ds + Nv,3*Ns*Ds + Nv + Nx*Dx)
  out['secular_velocity'] = np.reshape(secular_velocity,(Nx,Dx))

  baseline = range(3*Ns*Ds + Nv + Nx*Dx,3*Ns*Ds + Nv + 2*Nx*Dx)
  out['baseline_displacement'] = np.reshape(baseline,(Nx,Dx))

  return out


def flat_data(u):
  '''
  takes a Nx by Dx array of data and flattens it to one dimension
  '''
  return np.reshape(u,(np.prod(np.shape(u)),))


def flat_covariance(C):
  '''
  takes a Nx by Dx by Dx covariance array and flattens it to two
  dimension
  '''
  return scipy.linalg.block_diag(*C)


def observation(X,t,F,G,p,flatten=True):
  tect = X[p['secular_velocity']]*t + X[p['baseline_displacement']]
  slip = np.einsum('ijkl,ij',F,X[p['slip']])
  visc = np.einsum('ijklm,ij,k',G,X[p['slip_integral']],
                                  X[p['fluidity']])
  out = tect + slip + visc

  if flatten:
    out = flat_data(out)

  return out


def transition(X,t2,t1,p):
  Xout = np.zeros(p['total'])
  dt = t2 - t1
  Xout[p['slip_integral']] += (X[p['slip_integral']] +
                               X[p['slip']]*dt +
                               X[p['slip_derivative']]*0.5*dt**2)

  Xout[p['slip']] += (X[p['slip']] +
                      X[p['slip_derivative']]*dt)

  Xout[p['slip_derivative']] += X[p['slip_derivative']]

  Xout[p['fluidity']] += X[p['fluidity']]

  Xout[p['secular_velocity']] += X[p['secular_velocity']]

  Xout[p['baseline_displacement']] += X[p['baseline_displacement']]

  return Xout


def process_covariance(t2,t1,alpha,p):
  dt = t2 - t1
  Q = np.zeros((p['total'],p['total']))

  Q[p['slip_derivative'],p['slip_derivative']] = alpha**2*dt
  Q[p['slip'],p['slip']] = alpha**2*(dt**3)/3.0
  Q[p['slip_integral'],p['slip_integral']] = alpha**2*(dt**5)/20.0

  Q[p['slip_integral'],p['slip']] = alpha**2*(dt**4)/8.0
  Q[p['slip'],p['slip_integral']] = alpha**2*(dt**4)/8.0

  Q[p['slip_derivative'],p['slip']] = alpha**2*(dt**2)/2.0
  Q[p['slip'],p['slip_derivative']] = alpha**2*(dt**2)/2.0

  Q[p['slip_derivative'],p['slip_integral']] = alpha**2*(dt**3)/6.0
  Q[p['slip_integral'],p['slip_derivative']] = alpha**2*(dt**3)/6.0

  return Q
