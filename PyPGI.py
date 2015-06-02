#!/usr/bin/env python
import sys
sys.path.append('./lib')
sys.path.append('./input')
import argparse
import pypgi

p = argparse.ArgumentParser(
      description='Python-based Postseismic Geodetic Inverstion')
args = p.parse_args()
pypgi.main()
