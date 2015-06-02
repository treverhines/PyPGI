#!/usr/bin/env python
import sys
import json
import h5py
import numpy as np
import filterfcn
import regularization
import inverse

def main():
  try:
    f = open('input/usrparam.json')
  except IOError:
    raise ImportError(
      'must have a json file named "usrparam.json" within the input ')
  param = json.load(f)
  f.close()

  try:
    from usrfcn import slip_basis
  except ImportError:
    raise ImportError(
      'must have a module named "usrfcn.py" within the input '
      'directory and it must contain the function slip_basis')

  try:
    from usrfcn import fluidity_basis
  except ImportError:
    raise ImportError(
      'must have a module named usrfcn.py within the input '
      'directory and it must contain the function fluidity_basis')

  try:
    f = h5py.File('input/greens_functions.h5','r')
  except IOError:
    raise IOError(
      'must have a h5py file named greens_functions.h5 within the '
      'input directory')

  try:
    F = f['elastic'][...]
    G = f['viscoelastic'][...]
    x = f['metadata/position'][...]
  except KeyError:
    raise KeyError

  f.close()
  try:
    f = h5py.File('input/data.h5','r')
  except IOError:
    raise IOError(
      'must have a h5py file named data.h5 within the input directory')

  alpha = param['slip_acceleration_variance']
  u = f['data'][...]
  Cd = f['covariance'][...]
  x2 = f['metadata/position'][...]
  t = f['metadata/time'][...]

  # check for consistency between input
  Ns,Ds,Nv,Nx,Dx = np.shape(G)
  Nt = len(t)
  assert np.all(np.isclose(x,x2))
  assert np.shape(u) == (Nt,Nx,Dx)
  assert np.shape(Cd) == (Nt,Nx,Dx,Dx)
  assert np.shape(F) == (Ns,Ds,Nx,Dx)
  assert np.shape(G) == (Ns,Ds,Nv,Nx,Dx)

  p = filterfcn.state_parser(Ns,Ds,Nv,Nx,Dx)

  Xprior,Cprior = regularization.priors(p)

  reg = regularization.regularization_matrix(param,p)

  kalman = inverse.KalmanFilter(Xprior,Cprior,
                                filterfcn.transition,
                                filterfcn.observation,
                                filterfcn.process_covariance)
  for i in range(Nt):
    flat_ui = filterfcn.flat_data(u[i,...])
    flat_Cdi = filterfcn.flat_covariance(Cd[i,...])
    kalman.update(flat_ui,flat_Cdi,
                  observation_args=(t[i],F,G,p),
                  solver_kwargs={'regularization':reg})
    if i != (Nt - 1):
      kalman.predict(transition_args=(t[i+1],t[i],p),
                     process_covariance_args=(t[i+1],t[i],alpha,p))   

  kalman.smooth()
  
  print(kalman.state[0]['smooth'][p['fluidity']])
  print(kalman.state[0]['smooth'][p['slip']])
  return


