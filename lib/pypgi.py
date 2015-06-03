#!/usr/bin/env python
import numpy as np
import inverse
import logging
import h5py
import scipy

logger = logging.getLogger(__name__)

def hdf5_to_dict(f):
  if type(f) is h5py.File:
    return hdf5_to_dict(f['/'])

  if type(f) is h5py.Group:
    return {i:hdf5_to_dict(v) for i,v in f.iteritems()}

  if type(f) is h5py.Dataset:
    return np.array(f)


def dict_to_hdf5(d,f):
  for i,v in d.iteritems():
    if type(v) is dict:
      f.create_group(i) 
      dict_to_hdf5(v,f[i])

    else:   
      f.create_dataset(i,data=v)


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


def kalmanfilter(data,prior,gf,reg,param):
  '''
  Nt: number of time steps
  Nx: number of positions
  Dx: spatial dimensions of coordinates and displacements 
  Ns: number of slip basis functions per slip direction
  Ds: number of slip directions
  Nv: number of fluidity basis functions
  total: number of state parameters (Ns*Ds + Nv + 2*Nx*Dx)

  Parameters
  ----------
    data: \ mean : (Nt,Nx,Dx) array
          \ covariance : (Nt,Nx,Dx,Dx) array
          \ metadata \ position : (Nx,Dx) array
                       time : (Nt,) array

    prior: \ mean : (total,) array
           \ covariance : (total,total) array

    gf: \ elastic : (Ns,Ds,Nx,Dx) array
        \ viscoelastic : (Ns,Ds,Dv,Nx,Dx) array
        \ metadata \ position : (Nx,Dx) array

    reg: \ regularization : (*,total) array

    params: user parameter dictionary

    

  Returns
  -------
    out: \ slip_integral \ mean : (Nt,Ns,Ds) array
                         \ uncertainty :(Nt,Ns,Ds) array
         \ slip \ mean : (Nt,Ns,Ds) array
                \ uncertainty : (Nt,Ns,Ds) array
         \ slip_derivative \ mean : (Nt,Ns,Ds) array
                           \ uncertainty : (Nt,Ns,Ds) array
         \ fluidity \ mean : (Nv,) array
                    \ uncertainty : (Nv,) array
         \ secular_velocity \ mean : (Nx,Dx) array
                            \ uncertainty : (Nx,Dx) array
         \ baseline_displacement \ mean : (Nx,Dx) array
                                 \ uncertainty : (Nx,Dx) array

  '''
  F = gf['elastic']
  G = gf['viscoelastic']
  x = gf['metadata']['position']
  alpha = param['slip_acceleration_variance']
  u = data['mean']
  Cd = data['covariance']
  x2 = data['metadata']['position']
  t = data['metadata']['time']

  # check for consistency between input
  Ns,Ds,Nv,Nx,Dx = np.shape(G)
  Nt = len(t)
  assert np.all(np.isclose(x,x2))
  assert np.shape(u) == (Nt,Nx,Dx)
  assert np.shape(Cd) == (Nt,Nx,Dx,Dx)
  assert np.shape(F) == (Ns,Ds,Nx,Dx)
  assert np.shape(G) == (Ns,Ds,Nv,Nx,Dx)
  p = state_parser(Ns,Ds,Nv,Nx,Dx)
  
  if reg is None:
    reg1 = inverse.tikhonov_matrix(p['slip'],0,column_no=p['total'])
    reg2 = inverse.tikhonov_matrix(p['fluidity'],0,column_no=p['total'])
    regmat = np.vstack((reg1,reg2))

  else:
    regmat = reg['regularization']

  if prior is None:
    Xprior = np.zeros(p['total'])  
    Cprior = 1e6*np.eye(p['total'])  

  else:
    Xprior = prior['mean']
    Cprior = prior['covariance']

  kalman = inverse.KalmanFilter(Xprior,Cprior,
                                transition,
                                observation,
                                process_covariance)

  logger.info('starting Kalman filter iterations')
  for i in range(Nt):
    flat_ui = flat_data(u[i,...])
    flat_Cdi = flat_covariance(Cd[i,...])
    kalman.update(flat_ui,flat_Cdi,
                  observation_args=(t[i],F,G,p),
                  solver_kwargs={'regularization':regmat,
                                 'LM_damping':False})
    if i != (Nt - 1):
      kalman.predict(transition_args=(t[i+1],t[i],p),
                     process_covariance_args=(t[i+1],t[i],alpha,p))   

    logger.info('finished Kalman filter iteration %s of %s' % (i+1,Nt))

  logger.info('smoothing with RTS method')
  kalman.smooth()

  state_mean = kalman.get('smooth')
  state_covariance = kalman.get('smooth_covariance')
  state_std = [np.sqrt(np.diag(i)) for i in state_covariance]

  logger.info('computing predicted data')
  # compute predicted data
  pred = np.zeros((Nt,p['Nx'],p['Dx']))
  for i in range(Nt):
    pred[i,:,:] = observation(state_mean[i],t[i],F,G,p,flatten=False)


  slip_int = {'mean':np.array([i[p['slip_integral']] for i in state_mean]),   
              'uncertainty':np.array([i[p['slip_integral']] for i in state_std])}

  slip = {'mean':np.array([i[p['slip']] for i in state_mean]),   
          'uncertainty':np.array([i[p['slip']] for i in state_std])}

  slip_dif = {'mean':np.array([i[p['slip_derivative']] for i in state_mean]),   
              'uncertainty':np.array([i[p['slip_derivative']] for i in state_std])}

  visc = {'mean':state_mean[-1][p['slip']],   
          'uncertainty':state_std[-1][p['slip']]}

  vel = {'mean':state_mean[-1][p['secular_velocity']],   
         'uncertainty':state_std[-1][p['secular_velocity']]}

  base = {'mean':state_mean[-1][p['baseline_displacement']],   
          'uncertainty':state_std[-1][p['baseline_displacement']]}

  state_soln = {'slip_integral':slip_int,
                'slip':slip,
                'slip_derivative':slip_dif,
                'visc':visc,
                'secular_velocity':vel,
                'baseline_displacement':base}

  out = {'state':state_soln,
         'predicted':pred}

  return out


