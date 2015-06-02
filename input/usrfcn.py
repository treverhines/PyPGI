#!/usr/bin/env python
import spectral.bspline

slip_collocation = [0.0,0.5,1.0]
fluidity_collocation = [1.0,5.0,10.0,20.0,50.0,100.0]

def slip_basis(z,n,diff=None):
  knots = [0.0,0.0,0.0,0.5,1.0,1.0,1.0]
  return spectral.bspline.bspline_1d(z,knots,n,2,diff=diff)

def fluidity_basis(z,n,diff=None):
  knots = [1.0,1.0,1.0,5.0,10.0,20.0,50.0,100.0]
  return spectral.bspline.bspline_1d(z,knots,n,2,diff=diff)



