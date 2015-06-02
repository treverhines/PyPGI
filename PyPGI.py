#!/usr/bin/env python
#
# This function is intended to check to validity of the input and 
# handle output 
#
import sys
sys.path.append('./lib')
import argparse
import pypgi
import h5py
import json
import logging
import misc

# Setup command line argument parser
p = argparse.ArgumentParser(
      description='Python-based Postseismic Geodetic Inverstion')

args = vars(p.parse_args())

# Setup logger
logger = logging.getLogger()
formatter = logging.Formatter(
              '%(asctime)s %(module)s: [%(levelname)s] %(message)s',
              '%m/%d/%Y %H:%M:%S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
#file_handler = logging.FileHandler(
#                 'output/%s/%s.log' % (name,name),'w')
#file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

# Load input files
data_file = h5py.File('input/data.h5')
data = pypgi.hdf5_to_dict(data_file)
data_file.close()

gf_file = h5py.File('input/greens_functions.h5')
gf = pypgi.hdf5_to_dict(gf_file)
gf_file.close()

param_file = open('input/usrparam.json')
param = json.load(param_file)
param_file.close()

try:
  prior_file = h5py.File('input/prior.h5','r')
  prior = pypgi.hdf5_to_dict(prior_file)
  prior_file.close()

except IOError:
  prior = None


try:
  reg_file = h5py.File('input/regularization.h5','r')
  reg = pypgi.hdf5_to_dict(reg_file)
  reg_file.close() 

except IOError:
  reg = None

out = pypgi.kalmanfilter(data,prior,gf,reg,param)

name = param['name']
itr = 1
outname = name
while True:
  try:
    out_file = h5py.File('output/%s.h5' % outname,'w-')
    break

  except IOError:
    logger.warning(
      'output name %s exists, trying %s-%s' % (outname,name,itr))
    outname = '%s-%s' % (name,itr)
    itr += 1

pypgi.dict_to_hdf5(out,out_file)
logger.info('output saved to output/%s.h5' % outname)
