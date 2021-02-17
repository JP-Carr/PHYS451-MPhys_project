def h0_to_q22(h0, dist, f0):
    q22=1.83e32*(h0/1e-25)*(dist/1)*(100/f0)**2
    return q22

def q22_to_h0(q22, dist, f0):
    h0=q22/(1.83e32*(1/1e-25)*(dist/1)*(100/f0)**2)
    return h0