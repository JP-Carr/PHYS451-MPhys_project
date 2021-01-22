from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import time 
from HeterodynedData import generate_het


def gen_phi0(psi):
    
   # print(psi[0])
    out=[]
    for j in phi0[0]:
        print(psi[0],j)
        het=generate_het(PHI0=j, PSI=psi[0]).data
        out.append(np.mean(np.absolute(origin-np.concatenate((het.real,het.imag)))))
    #    x=1/0
    return [psi[0],np.array(out)]


def make_het(psi, phi0):
    out=np.empty([len(phi0),len(psi)])
    m,n=0,0
    
    
    
    if __name__ == "__main__":
        max_processes = mp.cpu_count()  # number of simultaneous processes cannot excede the number of logical processors
        pool = mp.Pool(max_processes)
        processes = pool.map_async(gen_phi0, psi) # Assign processes to the processing pool
        pool.close() #finish assigning
        pool.join() #begins multiprocessing
        
        print("1-----------------------")
        output=processes.get()
        sorted(output, key=lambda x: x[0])
        print(output)
        print("2-----------------------")
        for i in output:
            i.pop(0)
       # output=[i[0] for i in output]
        output=np.vstack(output)
        print(output)
        return output
  #  for i in psi[:,0]:
        
  #  return out

#phi0, psi[:,0]


if __name__ == "__main__":
    start=time.time() #starts timing
    # Make data.
    origin=generate_het(PHI0=2.4, PSI=1.1).data
    origin=np.concatenate((origin.real,origin.imag))
    #y=generate_het(PSI=0.82, PHI0=0.74).data
    
    step=0.1
    phi0= np.arange(0., np.pi, step)
    psi= np.arange(0., np.pi/2, step/2)
    phi0, psi = np.meshgrid(phi0,psi) 
    
    print(len(phi0),len(psi))
   # x=1/0
    het=make_het(psi, phi0)
    print("Time elapsed = {}s".format(round(time.time()-start,2)))  #finishes timing
  #  print(het)
    
    """
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(phi0, psi, het, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    ax.set_xlabel(r'$\phi_0$')
    ax.set_ylabel('$\psi$')
    ax.set_zlabel('$\Delta$')
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()