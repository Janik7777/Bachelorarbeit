import deepxde as dde
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN
import random
import matplotlib.pyplot as plt
from deepxde.backend import tf


nu = 1e-20

start_time = 0.0
skip_iterations = 10
delta_t = 0.01
num_iterations = 62

num_points = 400

# 1. Das Gebiet
geom = dde.geometry.Disk([0.5, 0.5], 0.45)
timedomain = dde.geometry.TimeDomain(start_time,start_time + delta_t * (float(num_iterations * skip_iterations) - 1.0))
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Die PDG

def ConvDiff_system(x_y_t, u_a):
    """Convdiff
    u_t + a * grad(u) - nu * laplace(u) = s
    """

    u = u_a[:, 0:1]
    a1 = u_a[:, 1:2]
    a2 = u_a[:, 2:3]

    u_x = dde.grad.jacobian(ys=u, xs=x_y_t, j=0)
    u_y = dde.grad.jacobian(ys=u, xs=x_y_t, j=1)
    u_t = dde.grad.jacobian(ys=u, xs=x_y_t, j=2)

    u_xx = dde.grad.hessian(ys=u, xs=x_y_t, i=0, j=0)
    u_yy = dde.grad.hessian(ys=u, xs=x_y_t, i=1, j=1)

    a1_t = dde.grad.jacobian(ys=a1, xs=x_y_t, j=2)
    a2_t = dde.grad.jacobian(ys=a2, xs=x_y_t, j=2)

    return nu * (u_xx + u_yy) - a1 * u_x - a2 * u_y - u_t, a1_t, a2_t


# 3. Weitere Bedingungen

# Simulationsdaten zu u
def gen_traindata(i):
    j = int(start_time / delta_t + i * skip_iterations)

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName("D:\Dokumente\Studium\Bachelorarbeit\data\convdiff_instat\\rotation_2\\1_solution_5_{}_0.vtu".format(j))
    reader.Update()
    data = reader.GetOutput()
    
    points_3D = VN.vtk_to_numpy(data.GetPoints().GetData())
    points_3D[:,2] = points_3D[:,2] + (start_time + i * delta_t * skip_iterations)
    u = VN.vtk_to_numpy(data.GetPointData().GetArray("u"))
    u = u.reshape(u.shape[0], 1)

    shP = points_3D.shape
    shU = u.shape

    points_and_u = np.zeros((shP[0], 4))
    points_and_u[0:shP[0], 0:3] = points_3D
    points_and_u[0:shU[0], 3:4] = u

    x_dist = 0.5 - points_3D[:,0]
    y_dist = 0.5 - points_3D[:,1]
    dist = x_dist * x_dist + y_dist * y_dist
    
    points_and_u = points_and_u[dist < 0.45 * 0.45]

    unique_points_and_u = np.unique(points_and_u, axis=0)

    points = unique_points_and_u[:, 0:3] 
    u = unique_points_and_u[:, 3:4]
    shU = u.shape
    u = u.reshape(shU[0],1)

    return points, u

def load_training_data(num):
    points, u = gen_traindata(0)
    for i in range(1, num_iterations):
        points_i, u_i = gen_traindata(i)
        points = np.concatenate([points, points_i])
        u = np.concatenate([u, u_i])

    indeces = random.sample(range(0, u.shape[0]), num)

    ob_points = np.take(points, indeces, 0) 
    ob_u = np.take(u, indeces, 0)

    return ob_points, ob_u, points, u

ob_points, ob_u, all_points, all_u = load_training_data(num_points)
observe_u = dde.icbc.PointSetBC(ob_points, ob_u, component=0)

# Anfangsbedingungen

def initialFunction(point):
    x1 = point[:, 0:1]
    x2 = point[:, 1:2]
    dist = (x1 - 0.3)**2 + (x2 - 0.3)**2
    return 1.0 - dist

ic = dde.icbc.IC(geomtime, initialFunction, lambda _, on_initial: on_initial)

# 4. Kombination
data = dde.data.TimePDE(
    geomtime,
    ConvDiff_system,
    [ic, observe_u],
    num_domain=200,
    num_initial=100,
    anchors=ob_points
)

# 5. Das neuronale Netz
net = dde.nn.FNN([3] + [40] * 6 + [3], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
model.compile("adam", lr=0.0002)
losshistory, train_state = model.train(epochs=100000)

# 7. Plotten der Ergebnisse
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Vorhersage und Lösung:
prediction = model.predict(all_points)

u_pred = prediction[:, 0]
a1_pred = prediction[:, 1]
a2_pred = prediction[:, 2]
u_true = all_u[:, 0]
ob_x, ob_y, ob_t = all_points[:, 0], all_points[:, 1], all_points[:, 2]

a1_true = 0.5 - ob_y
a2_true = ob_x - 0.5


def plot_pred(i):
    # data for t
    t = start_time + delta_t * i * skip_iterations
    x_at_t = ob_x[ob_t == t]
    y_at_t = ob_y[ob_t == t]
    u_true_at_t = u_true[ob_t == t]
    u_pred_at_t = u_pred[ob_t == t]
    a1_true_at_t = a1_true[ob_t == t]
    a1_pred_at_t = a1_pred[ob_t == t]
    a2_true_at_t = a2_true[ob_t == t]
    a2_pred_at_t = a2_pred[ob_t == t]
    
    # u 
    vmax = max(np.amax(u_pred_at_t), np.amax(u_true_at_t))
    vmin = min(np.amin(u_pred_at_t), np.amin(u_true_at_t))
    fig, ax = plt.subplots(ncols=3, gridspec_kw={"width_ratios":[1,1, 0.05]})
    cntr0 = ax[0].tricontourf(x_at_t, y_at_t, u_pred_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cntr1 = ax[1].tricontourf(x_at_t, y_at_t, u_true_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    fig.colorbar(cntr1, cax=ax[2])

    ax[0].set_title("u: Vorhersage des PINNs" + "(t=" + str(t) + ")", fontsize=10)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=10, family="Arial")
    ax[0].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    ax[1].set_title("u: Simulierte Lösung mit Hiflow³" + "(t=" + str(t) + ")", fontsize=10)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=10, family="Arial")
    ax[1].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    fig.tight_layout()


    # a1
    vmax = max(np.amax(a1_pred_at_t), np.amax(a1_true_at_t))
    vmin = min(np.amin(a1_pred_at_t), np.amin(a1_true_at_t))
    fig, ax = plt.subplots(ncols=3, gridspec_kw={"width_ratios":[1,1, 0.05]})
    cntr0 = ax[0].tricontourf(x_at_t, y_at_t, a1_pred_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cntr1 = ax[1].tricontourf(x_at_t, y_at_t, a1_true_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    fig.colorbar(cntr1, cax=ax[2])

    ax[0].set_title("a1: Vorhersage des PINNs" + "(t=" + str(t) + ")", fontsize=10)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=10, family="Arial")
    ax[0].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    ax[1].set_title("a1: Simulierte Lösung mit Hiflow³" + "(t=" + str(t) + ")", fontsize=10)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=10, family="Arial")
    ax[1].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    fig.tight_layout()


    # a2
    vmax = max(np.amax(a2_pred_at_t), np.amax(a2_true_at_t))
    vmin = min(np.amin(a2_pred_at_t), np.amin(a2_true_at_t))
    fig, ax = plt.subplots(ncols=3, gridspec_kw={"width_ratios":[1,1, 0.05]})
    cntr0 = ax[0].tricontourf(x_at_t, y_at_t, a2_pred_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    cntr1 = ax[1].tricontourf(x_at_t, y_at_t, a2_true_at_t, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
    fig.colorbar(cntr1, cax=ax[2])

    ax[0].set_title("a2: Vorhersage des PINNs" + "(t=" + str(t) + ")", fontsize=10)
    ax[0].axis("scaled")
    ax[0].set_xlabel("X", fontsize=10, family="Arial")
    ax[0].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    ax[1].set_title("a2: Simulierte Lösung mit Hiflow³" + "(t=" + str(t) + ")", fontsize=10)
    ax[1].axis("scaled")
    ax[1].set_xlabel("X", fontsize=10, family="Arial")
    ax[1].set_ylabel("Y", fontsize=10, family="Arial", rotation=0)
    fig.tight_layout()


    plt.show()

i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))
while i != -1:
    plot_pred(i)
    i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))
