import numpy as np
import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def de(x, y, z, s=29., r=7., b=3.):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the system
    Returns:
       x_dot, y_dot, z_dot: values of the attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = 10.*(y-x)
    y_dot = 15.*x - x*z - y
    z_dot = x*y - (8./3.)*z
    
    return x_dot, y_dot, z_dot



def synthetic(dt=0.01,num_steps=10000,x0=10.,y0=10.,z0=10.,xW=1.,yW=1.,zW=1.,xO=1.,yO=1.,zO=1.):
    # Need one more for the initial values
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    
    # Set initial values
    xs[0], ys[0], zs[0] = (x0,y0,z0)
    
    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(num_steps):
        x_dot, y_dot, z_dot = de(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt) + xW*np.random.normal(0,np.sqrt(dt)) #+ np.random.normal(0,xO)
        ys[i + 1] = ys[i] + (y_dot * dt) + yW*np.random.normal(0,np.sqrt(dt)) #+ np.random.normal(0,yO)
        zs[i + 1] = zs[i] + (z_dot * dt) + zW*np.random.normal(0,np.sqrt(dt)) #+ np.random.normal(0,zO)
    return (xs,ys,zs)

if __name__ == '__main__':    
    dt = 0.01
    num_steps = 100000
    (xs,ys,zs) = synthetic(dt,num_steps,xW=0.,yW=0.,zW=0.)
    print("xs={},ys={},zs={}".format(xs,ys,zs))
    
    # Plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("System")
    
    plt.show()
    
