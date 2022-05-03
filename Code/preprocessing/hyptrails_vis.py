import matplotlib.pyplot as plt
import random

STATES = 5


def bar_plot(output_path: str = None, prior: bool = False):
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.set_zlim((0, 1))

    xpos = [a for b in [[x for _ in range(STATES)] for x in range(STATES)] for a in b]
    ypos = [a for b in [[y for y in range(STATES)] for _ in range(STATES)] for a in b]
    zpos = [0 for _ in range(STATES*STATES)]
    dx = [0.4 for _ in range(STATES*STATES)]
    dy = [0.4 for _ in range(STATES*STATES)]
    dz = [0.1 for _ in range(STATES*STATES)]

    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='blue')
    if prior:
        zpos = [0.1 for _ in range(STATES*STATES)]
        dz = [round(random.uniform(0.1, 0.5), 1) for _ in range(STATES * STATES)]
        zeros = [i for i, v in enumerate(dz) if v == 0.1]
        print(dz)
        print(zeros)
        print("X:", len(xpos), "Y:", len(ypos), "Z:", len(zpos), "dx:", len(dx), "dy:", len(dy), "dz:", len(dz))
        for zero in zeros:
            del xpos[zero]
            del ypos[zero]
            del zpos[zero]
            del dx[zero]
            del dy[zero]
            del dz[zero]

        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='red')
    if output_path:
        if prior:
            name = "hyptrails_prior.eps"
        else:
            name = "hyptrails_normal.eps"
        plt.savefig(output_path + name, dpi=300)
    plt.show()


if __name__ == '__main__':
    # bar_plot(output_path="./data/images/")
    bar_plot(output_path="./data/images/", prior=True)
