import random
import matplotlib.pyplot as plt

def randomwalk(weights,n):
    coor=[0, 0]
    options=[-1,1,-2,2]
    xpoints,ypoints=[],[]
    visits=dict()
    for i in range(n):
        dd=random.choices(options,weights=weights,k=1)[0]
        if dd%2==0:
            coor[1]+=int(dd / 2)
        else:
            coor[0]+=dd
        xpoints.append(coor[0])
        ypoints.append(coor[1])
        #iter da povrzam so kordinata
        visits[tuple(coor)] = visits.get(tuple(coor), 0) + 1

    #print(coor)
    points=zip(xpoints,ypoints)
    plt.savefig("randomwalk.png", dpi=600, bbox_inches="tight")
    plt.scatter(xpoints, ypoints, c=range(len(xpoints)), cmap="plasma",
                s=10,  # marker size
                alpha=0.9,  # keep visible but not opaque
                edgecolors="black",  # outline
                linewidths=0.2)
    cbar = plt.colorbar()
    cbar.set_label("Step")

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("2D Coordinate Plot")
    plt.show()
if __name__ == '__main__':
    weights=[.5,.5,.5,.5]
    randomwalk(weights,10000)