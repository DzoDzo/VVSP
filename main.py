import random
import matplotlib.pyplot as plt
import numpy as np

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
def makeContinues(time,dirs,steps):
    #mu vrakjat, vrednost na Sn, i vreme
    time=np.array(time)*(1/len(time))

    Sn=np.cumsum(dirs)
    np.insert(Sn,0,0)
    print(len(Sn),len(time)) #tuka so zgolemuvanje na N, disperzijata nogo jako raste
    plt.scatter(time,Sn)
    plt.show()
    #do tuka e samo skalirano vremeto
    print(time)
    scaled_dirs=dirs*(1/np.sqrt(len(time)))
    Snt=np.cumsum(scaled_dirs)
    np.insert(Snt,0,0)
    plt.scatter(time, Snt,color='r',s=1)
    plt.show()

    #sakame da ima poveche tochki
    continues=[]
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

    t_rw, dirs,X_rw=randomwalk(.5,.5,1000000)
    t_rw_scaled=np.array(t_rw)*100
    makeContinues(t_rw,dirs,10000)
    # plt.plot(t_cont, X_cont, color="tab:purple", linewidth=1.0, alpha=0.9, label="Interpolated path")
    #
    # # overlay discrete points
    # plt.scatter(t_rw_scaled, X_rw,
    #             s=30, color="white", edgecolors="black", zorder=3,
    #             label="Random walk steps")
    #
    # # highlight start/end
    # plt.scatter(t_rw_scaled[0], X_rw[0], c="green", s=80, edgecolors="black", zorder=4, label="Start")
    # plt.scatter(t_rw_scaled[-1], X_rw[-1], c="red", s=80, edgecolors="black", zorder=4, label="End")
    # t_wiener,X_wiener=Wiener(1,10000,None)
    # plt.plot(np.array(t_wiener), X_wiener, color="tab:orange", lw=1.0, alpha=0.8, label="Wiener (Brownian)")
    #
    # plt.xlabel("t")
    # plt.ylabel("X")
    # plt.title("Random Walk vs Continuous Interpolation (Donsker scaling)")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

