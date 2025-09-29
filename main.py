import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cm as cm
import scipy.stats as stats
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.linalg import eigh
from scipy.stats import poisson
from scipy.stats import norm


def randomwalk(p,q,n):
    dir=[]
    dirs=[-1,1]
    probs=[p,q]
    for i in range(n):
        dir.append(random.choices(dirs,probs,k=1)[0])
        #print(walk)
    rw=np.cumsum(dir)
    np.insert(rw,0,0)
    return range(n),rw
def ScaledWalk(time,steps):
    dir = []
    k=(1/np.sqrt(steps))
    dirs = [-1*k, k]
    for i in range(time*steps):
        dir.append(random.choices(dirs)[0])
    sw = np.cumsum(dir)
    np.insert(sw, 0, 0)
    return np.array(list(range(steps))), sw
def Wiener(T,N,seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt=T/N
    increments=np.sqrt(dt)*np.random.randn(N)
    W=np.cumsum(increments)
    W=np.insert(W,0,0)
    t=np.linspace(0,T,N+1)
    #print((t),len(W),W)
    return t,W
def Wiener2D(T, N, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    dX = np.sqrt(dt) * np.random.randn(N)
    dY = np.sqrt(dt) * np.random.randn(N)
    X = np.concatenate(([0.0], np.cumsum(dX)))
    Y = np.concatenate(([0.0], np.cumsum(dY)))
    t = np.linspace(0.0, T, N+1)
    return t, X, Y
def random_points_in_circle(center, radius, n_points):
    x0, y0 = center
    r = radius * np.sqrt(np.random.rand(n_points))      # sqrt ensures uniform area distribution
    theta = 2 * np.pi * np.random.rand(n_points)        # random angles
    x = x0 + r * np.cos(theta)
    y = y0 + r * np.sin(theta)
    return np.column_stack((x, y))
def hits(coordinate,list_coords,proximity):
    for coord in list_coords:
        if coord[0]+proximity[0] > coordinate[0] > coord[0]-proximity[0]:
            if coord[1]+proximity[1] > coordinate[1] > coord[1]-proximity[1]:
                return True
    return False
def BrownianMotion(num_part, mass_polen, mass_part,iters):
    polen_loc=[0.5,0.5]
    path=list()
    path.append(polen_loc)
    points=random_points_in_circle(polen_loc,1,num_part)
    for _ in range(iters):
        for i in range(num_part):
            polen_dx,polen_dy=0,0
            particle_dx,particle_dy=np.random.randn(),np.random.randn()
            if hits(points[i],[polen_loc],[0.5,0.5]):
                polen_dx+=mass_part / mass_polen*particle_dx
                polen_dy += mass_part / mass_polen * particle_dy
                particle_dx += mass_polen / mass_part * particle_dx
                particle_dy += mass_polen / mass_part * particle_dy
            polen_loc[0] += polen_dx
            polen_loc[1] += polen_dy
            points[i][0]+=particle_dx
            points[i][1]+=particle_dy
        path.append(tuple(polen_loc))

    return path
def BrownianModified(num_part,mass_polen, mass_part,iters):
    polen_loc = [0.5, 0.5]
    path = list()
    path.append(polen_loc)
    points = random_points_in_circle(polen_loc, 1, num_part)
    for _ in range(iters):
        for i in range(num_part):
            polen_dx, polen_dy = 0, 0
            particle_dx, particle_dy = np.random.randn(), np.random.randn()
            if hits(points[i], [polen_loc], [0.5, 0.5]):
                polen_dx += mass_part / mass_polen * particle_dx
                polen_dy += mass_part / mass_polen * particle_dy
                points[i] = random_points_in_circle(polen_loc, 1, 1)
            polen_loc[0] += polen_dx
            polen_loc[1] += polen_dy

        path.append(tuple(polen_loc))

    return path
def statistiks(vals):
    X=np.array(vals)
    mean_vec = X.mean(axis=0)  # (d,)
    cov_mat = np.cov(X, rowvar=False, ddof=1)  # (d,d)
    return mean_vec, cov_mat
def vectoradd(elem1,elem2):
    res=[0]*len(elem1)
    for i in range(len(elem1)):
        res[i]=elem1[i]+elem2[i]
    return res
def Lattice(dimensions,iters,probs=None,steps=None):
    if dimensions==1:
        if probs is None:
            probs = [0.5, 0.5]
        if steps is None:
            steps = [1, -1]
        Sn = [0]
        for i in range(1,iters):
           step=random.choices(steps, probs, k=1)[0]
           Sn.append(Sn[i-1]+step)
    if dimensions==2:
        if probs is None:
            probs = [0.25, 0.25,0.25,0.25]
        if steps is None:
            steps = [(1,0), (-1,0),(0,1), (0,-1)]
        else:
            steps=[(steps[0],0), (steps[1],0), (0,steps[2]), (0,steps[3])]
        Sn=[(0,0)]
        for i in range(1,iters):
            step=random.choices(steps, probs, k=1)[0]
            Sn.append(vectoradd(Sn[i-1],step))
    return Sn
def NonLattice(iters,distrib):
    Sn=[0]
    for i in range(1,iters):
        Sn.append(Sn[i-1]+distrib.rvs(1)[0])
    return Sn
def hitting_probs(p,a,b):
    N=a+b
    if isinstance(p, (int, float)):
        if p==0.5:
            return a/(a+b)
        return (1-((1-p)/p)**a)/(1-((1-p)/p)**(a+b))
    sum_br=1
    sum_im=1
    s_i=1
    for i in range(1,N):
        s_i *= (1 - p[i - 1]) / p[i - 1]
        if i<a:
            sum_br+=s_i
        sum_im+=s_i

    return sum_br/sum_im
def expected_time(p, a, b, start_at=0):
    N = a + b
    i = a + start_at  # map start in {-a,...,b} to {0,...,N}
    if i <= 0 or i >= N:
        return 0.0  # already at a boundary

    if isinstance(p, (int, float)):
        if abs(p - 0.5) < 1e-12:
            return float(i * (N - i))  # i(N-i)
        r = (1 - p) / p
        return (N / (2*p - 1)) * ((r**i - 1) / (r**N - 1)) - i / (2*p - 1)

    s = [1.0]
    for k in range(1, N):  # k = 1..N-1
        s.append(s[-1] * (1 - p[k-1]) / p[k-1])
    S = [0.0]
    for k in range(N):
        S.append(S[-1] + s[k])
    A = [0.0]
    for m in range(1, N):  # m = 1..N-1
        A.append(A[-1] + 1.0 / (p[m-1] * s[m]))

    B = [s[0] * A[0]]  # = 0
    for m in range(1, N):
        B.append(B[-1] + s[m] * A[m])

    v1 = B[N-1] / S[N]
    return S[i] * v1 - B[i-1]
def max_till(listac,till):
    maxac=listac[0]
    for i in range(till):
        if listac[i]>maxac:
            maxac=listac[i]
    return maxac
def simulate_many_via_calls(T, N, K, seeds=None, scales=None):
    """
    Reuse Wiener2D K times.
    scales: optional length-K array; X,Y increments are multiplied by scales[k].
    """
    if seeds is None:
        seeds = [None]*K
    if scales is None:
        scales = np.ones(K)

    t = None
    pos = np.empty((N+1, K, 2), dtype=float)
    for k in range(K):
        tk, X, Y = Wiener2D(T, N, seed=seeds[k])
        if t is None:
            t = tk
        # re-scale path to model “more excited” (variance multiplier = scale^2)
        pos[:, k, 0] = X * scales[k]
        pos[:, k, 1] = Y * scales[k]
    return t, pos#zagif

def make_gif(positions, fps=30, step_stride=4, tail=200, filename="particles.gif", colors=None):
    T1, K, _ = positions.shape
    idx = np.arange(0, T1, step_stride)
    pos = positions[idx]

    xy = pos.reshape(-1, 2)
    pad = 0.1 + 3*np.std(xy, axis=0).max()
    xmin, ymin = xy.min(axis=0) - pad
    xmax, ymax = xy.max(axis=0) + pad

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pos[0, :, 0], pos[0, :, 1], s=12, c=colors, alpha=0.9)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_title("2D Wiener Particles")
    trails = [ax.plot([], [], lw=1, alpha=0.25, color=(None if colors is None else colors[i]))[0] for i in range(K)]

    def update(f):
        sc.set_offsets(pos[f])
        k0 = max(0, f - tail // step_stride)
        seg = pos[k0:f+1]
        for i, line in enumerate(trails):
            line.set_data(seg[:, i, 0], seg[:, i, 1])
        return [sc, *trails]

    ani = FuncAnimation(fig, update, frames=len(pos), interval=1000/fps, blit=True)
    ani.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return filename
def diagnose_2wiener(path_xy, t=None, eps=1e-12):
    """
    path_xy: array (N,2) of positions (x,y)
    t: None for unit steps, else array (N,) of times (strictly increasing)
    Returns dict of diagnostics.
    """
    P = np.asarray(path_xy, float)
    N = len(P)
    if t is None:
        dt = np.ones(N-1)
    else:
        t = np.asarray(t, float)
        dt = np.diff(t)
    dZ = np.diff(P, axis=0)                       # (N-1, 2)
    dt_col = dt[:, None]                          # (N-1, 1)

    # 1) Drift check: mean of increments per unit time
    mu_hat = (dZ / dt_col).mean(axis=0)           # estimate of drift

    # 2) Estimate diffusion matrix Sigma via MLE:
    #    Sigma_hat = E[ (dZ dZ^T)/dt ]
    Sigma_hat = (dZ.T @ (dZ / dt_col)) / (N-1)    # (2,2)

    # 3) Standardize increments to N(0, I_2) under Sigma_hat
    #    Use symmetric sqrt inverse via eigen-decomp
    evals, evecs = eigh(Sigma_hat + eps*np.eye(2))
    Ssqrt_inv = evecs @ np.diag(1.0/np.sqrt(evals)) @ evecs.T

    U = (Ssqrt_inv @ (dZ / np.sqrt(dt_col)).T).T  # (N-1, 2)

    # 4) Basic moment checks on U ~ N(0, I)
    mean_U = U.mean(axis=0)
    cov_U  = (U.T @ U) / (len(U) - 1)

    # 5) Simple autocorr at lag 1 for standardized components
    def lag1_autocorr(x):
        x = x - x.mean()
        return (x[:-1]*x[1:]).sum() / ( (x**2).sum() - x[-1]**2 )
    acf_u1 = lag1_autocorr(U[:,0])
    acf_u2 = lag1_autocorr(U[:,1])

    # 6) Chi-square radii (should be ~ ChiSq(2): mean~2, var~4)
    r2 = np.sum(U**2, axis=1)
    r2_mean, r2_var = r2.mean(), r2.var()

    # 7) Realized covariance vs total time
    Ttot = dt.sum()
    realized = dZ.T @ dZ / Ttot  # should be close to Sigma_hat

    return {
        "drift_per_unit_time_hat": mu_hat,        # ≈ [0,0]
        "Sigma_hat": Sigma_hat,                   # diffusion estimate
        "std_increments_mean": mean_U,            # ≈ [0,0]
        "std_increments_cov": cov_U,              # ≈ I_2
        "lag1_autocorr_Ux": acf_u1,               # ≈ 0
        "lag1_autocorr_Uy": acf_u2,               # ≈ 0
        "r2_mean": r2_mean,                       # ≈ 2
        "r2_var": r2_var,                         # ≈ 4
        "realized_cov_over_T": realized,          # ≈ Sigma_hat
        "total_time": Ttot,
        "n_increments": len(dZ),
    }

def Relfection(n,a=1,iters=10000): #testiranjena princip za refleksija
    hits_a,ends_a=0,0
    for _ in range(iters):
        t,W=Wiener(1,n)
        if(max(W)>=a):
            hits_a+=1
        if(W[-1]>=a):
            ends_a+=1
    a = np.asarray(a, dtype=float)
    return hits_a/iters,ends_a/iters,2 * norm.sf(a / np.sqrt(1))

if __name__ == '__main__':

    print(Relfection(10000,iters=10000))

    iters=10000


    for (lam,mass_ratio) in [(10000,5),(10000,100),(10000,10)]:
            fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=200)
            path=BrownianLogical(iters,mass_ratio=mass_ratio,lamb_d=lam)
            statistika=BrownianStats(path)
            print(f"({lam},{mass_ratio}",statistika)
            x,y=zip(*path)
            plt.scatter(x, y, c=range(len(x)), cmap="plasma", s=6, edgecolors="black", linewidths=0.2)
            ax1.set_title(f"BrownianSim(lambda={lam},{mass_ratio})", fontsize=14)
            fig1.savefig(f"BrownianSim(lambda={lam},{mass_ratio}).png", dpi=600, bbox_inches="tight")
            plt.close(fig1)



    # fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=200)
    # t = np.arange(len(path))
    # plt.scatter(x, y, c=t, cmap="plasma", s=6, edgecolors="black", linewidths=0.2)
    # ax1.set_title(f"BrownianSim(lambda={lam},{mass_ratio})", fontsize=14)
    # fig1.savefig(f"BrownianSim(lambda={lam},{mass_ratio}).png", dpi=600, bbox_inches="tight")
    # plt.close(fig1)
    # print(f"{lam},{mass_ratio}")
    # mass_part = 1.0
    # iters = 10000
    #
    # path1 = BrownianMotion(num_part=200, mass_polen=10, mass_part=mass_part, iters=iters)
    # path2 = BrownianMotion(num_part=2000, mass_polen=100, mass_part=mass_part, iters=iters)
    # path3 = BrownianMotion(num_part=20000, mass_polen=1000, mass_part=mass_part, iters=iters)
    # path4= BrownianModified(num_part=2000, mass_polen=100, mass_part=mass_part, iters=iters)
    # stats1=BrownianStats(path1)
    # stats2=BrownianStats(path2)
    # stats3=BrownianStats(path3)
    # stats4=BrownianStats(path4)
    # stats1_set = BrownianStats(list(set(map(tuple, path1))))
    # stats2_set = BrownianStats(list(set(map(tuple, path2))))
    # stats3_set = BrownianStats(list(set(map(tuple, path3))))
    # stats4_set = BrownianStats(list(set(map(tuple, path4))))
    # print(stats1)
    # print(stats2)
    # print(stats3)
    # print(stats4)
    #
    # print(stats1_set)
    # print(stats2_set)
    # print(stats3_set)
    # print(stats4_set)
    # #kak vlijaa masa i broj takvi na DX i EX i guess,
    # x, y = zip(*path1)
    # fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=200)
    # ax1.scatter(x, y, alpha=0.4, s=5, color="#1f77b4")
    # ax1.set_title("BrownianSim(200,10)", fontsize=14)
    #
    # fig1.savefig("BrownianSim200.png", dpi=600, bbox_inches="tight")
    # plt.close(fig1)
    #
    # x, y = zip(*path2)
    # fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=200)
    # ax2.scatter(x, y, alpha=0.4, s=5, color="#ff7f0e")
    # ax2.set_title("BrownianSim(2000,100)", fontsize=14)  # adjust label as needed
    #
    # fig2.savefig("BrownianSim2000.png", dpi=600, bbox_inches="tight")
    # plt.close(fig2)
    #
    # x, y = zip(*path3)
    # fig3, ax3 = plt.subplots(figsize=(6, 6), dpi=200)
    # ax3.scatter(x, y, alpha=0.4, s=5, color="#2ca02c")
    # ax3.set_title("BrownianSim(20000,1000)", fontsize=14)
    #
    # fig3.savefig("BrownianSim20000.png", dpi=600, bbox_inches="tight")
    # plt.close(fig3)
    #
    # x, y = zip(*path4)
    # fig3, ax3 = plt.subplots(figsize=(6, 6), dpi=200)
    # ax3.scatter(x, y, alpha=0.4, s=5, color="#2ca02c")
    # ax3.set_title("BrownianSimModified(2000,100)", fontsize=14)
    #
    # fig3.savefig("BrownianSimModified.png", dpi=600, bbox_inches="tight")
    # plt.close(fig3)
    #
    # iters=1000
    # path=BrownianMotion(200,300,1,iters=iters)
    # #kak vlijaa masa i broj takvi na DX i EX i guess,
    # t,wx,wy=Wiener2D(1,iters)
    # pathw=XW = np.array(list(zip(wx, wy)))
    # print(statistiks(pathw),statistiks(path))
    # x,y=zip(*path)
    # plt.figure(figsize=(6, 6), dpi=600)
    # plt.scatter(x,y,alpha=0.4)
    # plt.scatter(wx,wy,alpha=0.4,color="r")
    # plt.show()
    # n = 10000
    # t, X, Y = Wiener2D(1, n)
    #
    # T, N, K = 20.0, 4000, 150
    # # 50 “hot” (more excited) particles with 2x variance (scale=√2), rest normal
    # scales = np.concatenate([np.sqrt(2) * np.ones(50), np.ones(K - 50)])
    # # Optional color coding
    # colors = np.array(["#d62728"] * 50 + ["#1f77b4"] * (K - 50))
    #
    # t, pos = simulate_many_via_calls(T, N, K, seeds=[123 + i for i in range(K)], scales=scales)
    # gif = make_gif(pos, fps=30, step_stride=5, tail=250, filename="wiener_particles.gif", colors=colors)
    # print("Saved:", gif)
    #
    # print("Saved:", gif)
    #
    # n=10000
    # colors = cm.plasma(np.linspace(0, 1, 20))
    # for i in range(20):
    #     plt.plot(X,Y,color=colors[i])
    # plt.show()
    #
    # a, b = 2, 3
    # p_list = [0.60, 0.70, 0.40, 0.55]
    #
    # u = hitting_probs(p_list, a, b)
    # t = expected_time(p_list, a, b)
    # print("Hitting prob (+b before -a):", u)
    # print("Expected time to absorption:", t)
    #
    # n=10000
    # Sn=Lattice(2,n,None,[3,-3,3,-3])
    # print(Sn)
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # x,y=zip(*Sn)
    # z=range(n)
    #
    #
    # ax.scatter(x, y,z , c=z, cmap="viridis", s=40)
    # ax.set_xlabel("X axis")
    # ax.set_ylabel("Y axis")
    # ax.set_zlabel("Z axis")
    # plt.show()
    #
    # gauss=stats.norm(0,31)
    # Snl=NonLattice(n,gauss)
    # print(Snl)
    # plt.scatter(range(n),Snl,c='r')
    # plt.show()
    #
    # n=100000
    #
    # t_rw, X_rw=randomwalk(.5,.5,n)
    # t_rw=np.array(t_rw)
    # #plt.scatter(t_rw, X_rw, color="tab:purple", linewidth=1.0, alpha=0.9, label="Interpolated path")
    # t_sw,X_sw=ScaledWalk(1,100)
    # colorcoded={1:"red",2:"blue",3:"green",4:"yellow",5:"magenta",6:"cyan"}
    # for i in range(2,7):
    #     deg=pow(10,i)
    #     t_sw, X_sw = ScaledWalk(1,deg)
    #     plt.scatter(t_sw*(1/deg), X_sw, color=colors[i-2],s=5, alpha=0.7)
    # patches = [mpatches.Patch(color=color, label=f"$10^{exp}$")
    #            for exp, color in zip(range(2,7), colors)]
    # plt.legend(handles=patches,
    #            title="n",
    #            loc="center left",  # anchor to left side of bbox
    #            bbox_to_anchor=(1.02, 0.5))
    #
    # plt.xlabel("t ∈ [0,1]")
    # plt.ylabel("S_n value") #TODO
    # plt.title("Scaled Random Walks converging to Brownian Motion", fontsize=14)
    # plt.tight_layout()
    # plt.savefig("scaled_walks.png", dpi=600, bbox_inches="tight")
    # plt.show()
    #
    # regular=0
    # reflected=0
    # w=1
    # for i in range(10000):
    #     t_W,W=Wiener(1,100000,None)
    #     maxima=max_till(W,10000)
    #     if i==0:
    #         w=W[-1]
    #     if W[-1]<=w:
    #         regular=regular+1
    #     if W[-1]>=2*maxima-w:
    #         reflected=reflected+1
    # print("regular=",regular,"reflected=",reflected)
    # plt.show()
