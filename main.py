#%%
import numpy as np
from matplotlib import pyplot as plt
import struct
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.lib.function_base import quantile
matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")

matplotlib.rc("text.latex", preamble=r"""
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{bm}
\DeclareMathOperator{\newdiff}{d} % use \dif instead
\newcommand{\dif}{\newdiff\!} %the correct way to do derivatives
\newcommand{\bigoh}{\mathcal{O}}
\makeatletter
\let\oldabs\abs
\def\abs{\@ifstar{\oldabs}{\oldabs*}}
\newcommand\norm[1]{\left\lVert#1\right\rVert}
""")

#%%

def readfile(nx):
    filename = "sol-nx=%d-ny=%d.bin" % (nx, 2*nx)
    file = open(filename, mode='rb')

    bloc = file.read(36)
    nx = struct.unpack('i', bloc[0:4])[0]
    ny = struct.unpack('i', bloc[4:8])[0]
    L = struct.unpack('d', bloc[8:16])[0]
    H = struct.unpack('d', bloc[16:24])[0]
    dt = struct.unpack('d', bloc[24:32])[0]
    niter = struct.unpack('i', bloc[32:36])[0]

    raw_data = np.fromfile(file, dtype=np.float64, count=-1, sep='', offset=0)
    data = raw_data.reshape((ny+1, nx+1))
    return data, nx, ny, L, H, dt, niter

#%%

data, nx, ny, L, H, dt, niter = readfile(128)
x, dx = np.linspace(0, L, nx+1, retstep=True)
y, dy = np.linspace(0, H, ny+1, retstep=True)
fig = plt.figure(figsize=(6.4, 6.4))
ax = fig.add_subplot(1,1,1)
ax.set_title("Numerical sol, $n_x=%d$, $n_y=%d$, $t=%.3g$" % (nx, ny, dt*niter), fontsize=15)
c = ax.imshow(data, extent=[x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2], cmap="Spectral_r")
plt.colorbar(c)
plt.savefig("Figures/sol.pdf")
plt.show()


#%% Reference simulation

data2 = np.zeros_like(data)
data2[0,:] = 1.0; data2[-1,:] = 1.0; data2[:,0] = 1.0; data2[:,-1] = 1.0

for i in range(niter):
    data2[1:-1, 1:-1] += dt* ((data2[2:,1:-1] - 2* data2[1:-1,1:-1] + data2[:-2,1:-1])/dy**2 + (data2[1:-1,2:] - 2* data2[1:-1,1:-1] + data2[1:-1,:-2])/dx**2)


fig = plt.figure(figsize=(6.4, 6.4))
ax = fig.add_subplot(1,1,1)
ax.set_title(r"Difference compared to\\ a reference implementation", fontsize=15)
c = ax.imshow(data - data2, extent=[x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2], cmap="Spectral_r")
plt.colorbar(c)
plt.savefig("Figures/solcomp.pdf")
plt.show()


# %%

def refsol(nx, ny, L, H, t, num=25):
    x, y = np.meshgrid(np.linspace(0, L, nx+1), np.linspace(0, H, ny+1))
    sol = x*0.0
    for i in range(num):
        for j in range(num):
            sol += 16/((2*i+1)*(2*j+1)*np.pi**2) * np.sin((2*i+1)*np.pi*x) * np.sin((2*j+1)*np.pi*y) * np.exp(-(2*i+1)**2*np.pi**2*t )* np.exp(-(2*j+1)**2*np.pi**2*t )

    sol = 1-sol
    print(np.exp(-(2*num+1)**2*t ))
    return sol

def L2norm(data, dx, dy):
    return (np.einsum("ij,ij->", data, data) * dx*dy)**0.5

# %%

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(8,7))
axes = axes.ravel()

err = []
dxx = []
for nx in [4, 8, 16, 32, 64, 128, 256]:
    data, nx, ny, L, H, dt, niter = readfile(nx)
    ref = refsol(nx, ny, L, H, dt*niter) 
    dx = L/nx
    dy = H/ny

    i = len(err)
    if i < 4:
        ax = axes[i]
        c = ax.imshow(np.abs(ref - data), extent=[x[0]-dx/2,x[-1]+dx/2,y[0]-dy/2,y[-1]+dy/2], cmap="Spectral_r")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(c, cax=cax, orientation='vertical')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_title("Error with $n_x=%d$" % nx)

    err += [L2norm(ref - data, dx, dy)]
    dxx += [dx]
plt.tight_layout()

# %%

err = np.array(err)
dxx = np.array(dxx)

poww = 2
plt.loglog(dxx, err, "o", label="Eperimental $L_2$ error")
plt.loglog(dxx, dxx**poww*err[-1]/((dxx**poww)[-1]), '--k', linewidth=0.7, label=r"$\bigoh(\Delta t^{%g} )$" % poww)
plt.grid()
plt.xlabel("$\Delta x$")
plt.ylabel("$L_2$ error")
plt.tight_layout()
plt.savefig("Convergence.pdf")
plt.legend()


# %%

for rank in range(256):
    tsample = np.fromfile("Tsamples/Tsamples-nproc=256-rank=%d-nx=4096-ny=4096.bin" % rank, dtype=np.float64, count=-1, sep='', offset=0)

    tsample = tsample[:-2].reshape((-1, 6))
    tsample = tsample[:,[0, 1]].ravel()

    tsample -= tsample[0]

    segments = tsample.reshape((-1, 1))

    time = tsample[6:-1:6]
    ddt = time[1:] - time[:-1]

    #plt.plot(tsample, tsample*0, ".")

    #%%
    s2 = tsample[np.logical_and(tsample < 100/1000, tsample > -1/1000)]
    s3 = np.zeros(int(len(s2)*1.5))
    s3[0::3] = s2[0::2]
    s3[1::3] = s2[1::2]
    s3[2::3] = np.NaN

    dh = 0.5
    null = s3*0 + rank
    plt.fill_between(s3*1e3, null-dh,null+dh, lw=0., color="k")

plt.xlabel("Time (ms)")
plt.xlim([0, 100])
plt.ylim([-0.5, 255.5])
plt.tight_layout()
plt.savefig("idle_propagation.pdf")


plt.xlim([5, 20])
plt.ylim([-0.5, 255.5])
plt.tight_layout()
plt.savefig("idle_propagation_small.pdf")
#plt.show()
plt.close()


# %%

fig, axes = plt.subplots(2, 2, figsize=(10,7))
axes = axes.ravel()

for i, nx in enumerate([64, 256, 1024, 4096]):
    ax = axes[i]
    meantimes = []
    quantiles = []
    Nproc = np.arange(1, 17)**2
    for nproc in Nproc:
        tsample = np.fromfile("Tsamples/Tsamples-nproc=%d-rank=0-nx=%d-ny=%d.bin" % (nproc, nx, nx), dtype=np.float64, count=-1, sep='', offset=0)

        tsample = tsample[:-2].reshape((-1, 6))
        tsample = tsample[:,[0, 1]].ravel()
        tsample -= tsample[0]

        segments = tsample.reshape((-1, 1))

        time = tsample[0:-1:2]
        ddt = time[1:] - time[:-1]
        mt = np.mean(ddt)
        quant = np.quantile(ddt, [0.1, 0.9])
        meantimes += [mt]
        quantiles += [quant]

    meantimes = np.array(meantimes)
    quantiles = np.array(quantiles).T
    
    ax.set_title("$n_x=%d$"%nx)
    if i == 0:
        ax.plot(Nproc, meantimes, "o-")
        ax.plot(Nproc, quantiles[0], "-k", linewidth=0.5, label=r"10\% quantiles")
        ax.plot(Nproc, quantiles[1], "-k", linewidth=0.5)
        ax.fill_between(Nproc, quantiles[0], quantiles[1], color="k", alpha=0.1, zorder=-10)

    else:
        ax.loglog(Nproc, meantimes, "o-")
        ax.loglog(Nproc, quantiles[0], "-k", linewidth=0.5, label=r"10\% quantiles")
        ax.loglog(Nproc, quantiles[1], "-k", linewidth=0.5)
        ax.fill_between(Nproc, quantiles[0], quantiles[1], color="k", alpha=0.1, zorder=-10)

    ax.grid()

plt.tight_layout()
plt.savefig("strong_scaling.pdf")    
plt.show()

# %%


c = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

N = 8
M = 6
px, py = np.meshgrid(np.arange(N), np.arange(M))
rpx, rpy = np.meshgrid([N+1], np.arange(M))
lpx, lpy = np.meshgrid([-2], np.arange(M))
tpx, tpy = np.meshgrid(np.arange(N), [M+1])
bpx, bpy = np.meshgrid(np.arange(N), [-2])

plt.plot(px, py, "o", color="k", alpha=0.1)
plt.plot(rpx, rpy, "o", color=c[3])
plt.plot(lpx, lpy, "o", color=c[3])
plt.plot(tpx, tpy, "o", color=c[3])
plt.plot(bpx, bpy, "o", color=c[3])
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step0.pdf")


# %%
plt.plot(px, py, "o", color="k", alpha=0.1)
plt.plot(rpx, rpy, "o", color=c[0])
plt.plot(lpx, lpy, "o", color=c[0])
plt.plot(tpx, tpy, "o", color=c[0])
plt.plot(bpx, bpy, "o", color=c[0])
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step1.pdf")

# %%
v = 0*px
v[0] = 1; v[:,0] = 1
v[-1] = 1; v[:,-1] = 1

plt.plot(px, py, "o", color="k", alpha=0.1)
plt.plot(px[v > 0], py[v > 0], "o", color=c[0])
plt.plot(rpx, rpy, "o", color="k", alpha=0.1)
plt.plot(lpx, lpy, "o", color="k", alpha=0.1)
plt.plot(tpx, tpy, "o", color="k", alpha=0.1)
plt.plot(bpx, bpy, "o", color="k", alpha=0.1)
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step2.pdf")

# %%

v[1] = 2; v[:,1] = 2
v[-2] = 2; v[:,-2] = 2

plt.plot(px, py, "o", color="k", alpha=0.1)
plt.plot(px[v > 0], py[v > 0], "o", color=c[0])
plt.plot(px[v > 1], py[v > 1], "o", color=c[1])
plt.plot(rpx, rpy, "o", color="k", alpha=0.1)
plt.plot(lpx, lpy, "o", color="k", alpha=0.1)
plt.plot(tpx, tpy, "o", color="k", alpha=0.1)
plt.plot(bpx, bpy, "o", color="k", alpha=0.1)
plt.axis('off')
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step3.pdf")
# %%
plt.plot(px, py, "o", color=c[0])
plt.plot(px[v > 1], py[v > 1], "o", color=c[1])
plt.plot(rpx, rpy, "o", color="k", alpha=0.1)
plt.plot(lpx, lpy, "o", color="k", alpha=0.1)
plt.plot(tpx, tpy, "o", color="k", alpha=0.1)
plt.plot(bpx, bpy, "o", color="k", alpha=0.1)
plt.axis('off')
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step4.pdf")
# %%

plt.plot(px, py, "o", color=c[0])
plt.plot(rpx, rpy, "o", color="k", alpha=0.1)
plt.plot(lpx, lpy, "o", color="k", alpha=0.1)
plt.plot(tpx, tpy, "o", color="k", alpha=0.1)
plt.plot(bpx, bpy, "o", color="k", alpha=0.1)
plt.axis('off')
plt.axis('off')
plt.gca().set_aspect(1)
plt.savefig("step4.pdf")
# %%
