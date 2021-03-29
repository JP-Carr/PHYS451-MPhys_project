#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:17:44 2021

@author: james
"""
path="/home/james/Documents/GitHub/PHYS451-MPhys_project/DAGout/test1000/pulsars/J1727-0755.par"
import numpy as np
from cwinpy import HeterodynedData
times = np.linspace(1000000000.0, 1000086340.0, 1440)
fake = HeterodynedData(times=times, fakeasd="H1", injpar=path, par=path, inject=True)
print(fake.injection_snr)