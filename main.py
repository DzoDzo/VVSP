import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches

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
def Wiener(T,N,seed):
    if seed is not None:
        np.random.seed(seed)
    dt=T/N
    increments=np.sqrt(dt)*np.random.randn(N)
    W=np.cumsum(increments)
    W=np.insert(W,0,0)
    t=np.linspace(0,T,N+1)
    print((t),len(W),W)
    return t,W
if __name__ == '__main__':

    colors = cm.plasma(np.linspace(0, 1, 5))
    t_rw, X_rw=randomwalk(.5,.5,100000)
    print(t_rw)
    #plt.scatter(t_rw, X_rw, color="tab:purple", linewidth=1.0, alpha=0.9, label="Interpolated path")
    t_sw,X_sw=ScaledWalk(1,100)
    colorcoded={1:"red",2:"blue",3:"green",4:"yellow",5:"magenta",6:"cyan"}
    for i in range(2,7):
        deg=pow(10,i)
        t_sw, X_sw = ScaledWalk(1,deg)
        plt.scatter(t_sw*(1/deg), X_sw, color=colors[i-2],s=5, alpha=0.7)
    patches = [mpatches.Patch(color=color, label=f"$10^{exp}$")
               for exp, color in zip(range(2,7), colors)]

    plt.legend(handles=patches,
               title="n",
               loc="center left",  # anchor to left side of bbox
               bbox_to_anchor=(1.02, 0.5))

    plt.xlabel("t ∈ [0,1]")
    plt.ylabel("S_n value") #TODO
    plt.title("Scaled Random Walks converging to Brownian Motion", fontsize=14)
    plt.tight_layout()
    plt.savefig("scaled_walks.png", dpi=600, bbox_inches="tight")
    plt.show()



