import math
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
import math
import pandas as pd
import numpy as np

def randomwalk(p,q,n): # za grfiranje na 1D talkanje
    dir=[]
    dirs=[-1,1]
    probs=[p,q]
    for i in range(n):
        dir.append(random.choices(dirs,probs,k=1)[0])
        #print(walk)
    rw=np.cumsum(dir)
    np.insert(rw,0,0)
    return range(n),rw
def Lattice(dimensions,iters,probs=None,steps=None): # za dobivanje na lattice pateka na 1d,2d talkanje
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
def NonLattice(iters,distrib): #talkanje so proizvolna distribucija na X
    Sn=[0]
    for i in range(1,iters):
        Sn.append(Sn[i-1]+distrib.rvs(1)[0])
    return Sn
def ScaledWalk(time,steps): #skaliranoto talkanje, za simulaacijata na donskerova teorema
    dir = []
    k=(1/np.sqrt(steps))
    dirs = [-1*k, k]
    for i in range(time*steps):
        dir.append(random.choices(dirs)[0])
    sw = np.cumsum(dir)
    np.insert(sw, 0, 0)
    return np.array(list(range(steps))), sw
def Wiener(T,N,seed=None): #za grafiranje na Vinerov vo edna dimenzija
    if seed is not None:
        np.random.seed(seed)
    dt=T/N
    increments=np.sqrt(dt)*np.random.randn(N)
    W=np.cumsum(increments)
    W=np.insert(W,0,0)
    t=np.linspace(0,T,N+1)
    #print((t),len(W),W)
    return t,W
def Wiener2D(T, N, seed=None): #za grafiranje na Vinerov vo dve dimenzii
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    dX = np.sqrt(dt) * np.random.randn(N)
    dY = np.sqrt(dt) * np.random.randn(N)
    X = np.concatenate(([0.0], np.cumsum(dX)))
    Y = np.concatenate(([0.0], np.cumsum(dY)))
    t = np.linspace(0.0, T, N+1)
    return t, X, Y

def vectoradd(elem1,elem2): #pomoshna
    res=[0]*len(elem1)
    for i in range(len(elem1)):
        res[i]=elem1[i]+elem2[i]
    return res
#slednite dve se za talkanje so 1 i -1 chekori
def hitting_probs(p,a,b): #za presmetuvanje na verojatnost za pogoduvanje b pred -a
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
def expected_time(p, a, b): #za presmetuvanje na ochekuvanoto vreme do pogoduvanje na barierite -a i b
    N = a + b
    if isinstance(p, (int, float)):
        if abs(p - 0.5) < 1e-12:
            return float(a * (N - a))
        r = (1 - p) / p
        return (N / (2*p - 1)) * ((r**a - 1) / (r**N - 1)) - a / (2*p - 1)

    s = [1.0]
    for k in range(1, N):
        s.append(s[-1] * (1 - p[k-1]) / p[k-1])
    S = [0.0]
    for k in range(N):
        S.append(S[-1] + s[k])
    A = [0.0]
    for m in range(1, N):
        A.append(A[-1] + 1.0 / (p[m-1] * s[m]))

    B = [s[0] * A[0]]
    for m in range(1, N):
        B.append(B[-1] + s[m] * A[m])

    v1 = B[N-1] / S[N]
    return S[a] * v1 - B[a-1]
def max_till(listac,till): #pomoshna funkcija
    maxac=listac[0]
    for i in range(till):
        if listac[i]>maxac:
            maxac=listac[i]
    return maxac
def fair_call_price_bs(K,r,T):
    df = pd.read_csv("synthetic_stock_prices.csv", parse_dates=["Date"])
    S = float(df["Close"].iloc[-1]) #cena na akcija vo segashnosta, csv sodrzhi prethodnata godina za nekoja akcija
    logs = np.log(df["Close"]).diff().dropna().to_numpy() # devijacija na logaritmite od odnosite na cena den za den
    sigma = logs.std(ddof=1) * math.sqrt(252) #252 trading denovi ima

    #r vo Black Scholes ne e procent tuku se presmestevua so ln(1+r)
    r=math.log(1+r)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(d2)
def fair_call_price_bin(K,r,T,n=1000):
    df = pd.read_csv("synthetic_stock_prices.csv", parse_dates=["Date"])
    S = float(df["Close"].iloc[-1])
    logs = np.log(df["Close"]).diff().dropna().to_numpy()
    sigma = logs.std(ddof=1) * math.sqrt(252)

    dt=T/n
    u=math.exp(sigma * math.sqrt(dt))
    d=1/u

    p=(math.exp(r*dt)-d)/(u-d)

    C=np.zeros(n+1)
    S_term=S*d**n
    #Ovde, za podobruvanje na algoritmot, gi chuvame samo terminalnite jazli, kafe idneksot na C, se odnesuva kon kolku pati od N cenata se ima nakacheno
    #do posledniot moment
    for i in range(n+1):
        C[i]=max(S_term-K,0)
        S_term*=u/d

    #sega odime nanazad
    for j in range(n-1,-1,-1): #sekoj ciklus ovde se odnesuva na edna iteracija na drvoto nanazad, pr za n-1 bi bilo za jazlite pred terminalnite
        for i in range(j+1): #seko
            C[i]=math.exp(-r*dt)* (p * C[i + 1] + (1.0 - p) * C[i])
    return C[0]
def Relfection(n,a=1,iters=10000): #za empirisko testiranje na teoremata dobiean od principot na reflekcija
    hits_a,ends_a=0,0
    for _ in range(iters):
        t,W=Wiener(1,n)
        if(max(W)>=a):
            hits_a+=1
        if(W[-1]>=a):
            ends_a+=1
    a = np.asarray(a, dtype=float)
    return hits_a/iters,ends_a/iters,float(2 * norm.sf(a / np.sqrt(1)))
if __name__ == '__main__':
    #testiranje na algoritmi


    prob=hitting_probs(0.7,a=5,b=7)
    time=expected_time(0.7,a=5,b=7)
    print(prob,time)
    prob = hitting_probs(0.3, a=5, b=7)
    time = expected_time(0.3, a=5, b=7)
    print(prob, time)
    p = [0.30, 0.336, 0.373, 0.409, 0.445, 0.482, 0.518, 0.555, 0.591, 0.627, 0.70]
    prob=hitting_probs(p,6,6)
    time=expected_time(p,6,6)
    print(prob,time)
    p = [0.70, 0.30, 0.70, 0.30, 0.70, 0.30, 0.70, 0.30, 0.70, 0.30, 0.70]
    prob = hitting_probs(p, 6, 6)
    time = expected_time(p, 6, 6)
    print(prob, time)
    p = [0.500, 0.530, 0.590, 0.650, 0.590, 0.530, 0.530, 0.590, 0.650, 0.590, 0.530]
    prob = hitting_probs(p, 6, 6)
    time = expected_time(p, 6, 6)
    print(prob, time)

    print(fair_call_price_bs(100, 0.03, 3))
    print(fair_call_price_bs(100, 0.05, 3))
    print(fair_call_price_bs(100, 0.08, 3))
    print(fair_call_price_bs(100, 0.12, 3))

    #Ovde interesno e deka donskerovata teorema vazhi, so zgolemuvanje na n ovie vrednosti se doblizhuvat

    print(fair_call_price_bin(100, 0.03, 3))
    print(fair_call_price_bin(100, 0.05, 3))
    print(fair_call_price_bin(100, 0.08, 3))
    print(fair_call_price_bin(100, 0.12, 3))

    for a in [1,1.5,2,2.5]: #primer vrednosti za a
        print(Relfection(n=10000,a=a,iters=10000))
