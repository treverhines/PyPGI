#!/usr/bin/env python
import sys
import argparse
sys.path.append('lib')
import PyPGI

p = argparse.ArgumentParser(
      description='Python-based Postseismic Geodetic Inverstion')
kwargs = dict(p.parse_args())
PyPGI.main(**kwargs)
