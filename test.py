from src import *

def run_simulator():
    ob = OpenBall3D()
    sim1 = ExitTimeSimulator3D(ob, 10, dt=1e-3, dX=1, n=10)
    sim1.run()

def run_plotter():
    bm1 = BrownianMotion3D(100)
    bm2 = BrownianMotion3D(100)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(bm1.Bts[0,], bm1.Bts[1,], bm1.Bts[2,])
    ax.plot(bm2.Bts[0,], bm2.Bts[1,], bm2.Bts[2,])

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = 10 * np.outer(np.cos(u), np.sin(v))
    y = 10 * np.outer(np.sin(u), np.sin(v))
    z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color="gray", alpha=0.5)

    plt.show()

if __name__ == "__main__":
    var = input("Type \'sim\' or \'plot\': ")
    if var == "sim":
        run_simulator()
    elif var == "plot":
        run_plotter()
    else:
        print("Invalid input")
