from cmath import cos
import deepxde as dde
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN
import random
import matplotlib.pyplot as plt
from deepxde.backend import tf
import re

nu = 1e-20

p1 = dde.Variable(0.4) 
p2 = dde.Variable(0.6) 

p1_true = 0.5
p2_true = 0.5

start_time = 0.0
skip_iterations = 10
delta_t = 0.01
num_iterations = 62

num_points = 400

# 1. Das Gebiet
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])
timedomain = dde.geometry.TimeDomain(start_time,start_time + delta_t * (float(num_iterations * skip_iterations) - 1.0))
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Die PDG

def ConvDiff_system(x_y_t, u):
    """Convdiff
    u_t + a * grad(u) - nu * laplace(u) = s
    """
    x = x_y_t[:, 0:1]
    y = x_y_t[:, 1:2]

    u_x = dde.grad.jacobian(ys=u, xs=x_y_t, j=0)
    u_y = dde.grad.jacobian(ys=u, xs=x_y_t, j=1)
    u_t = dde.grad.jacobian(ys=u, xs=x_y_t, j=2)

    u_xx = dde.grad.hessian(ys=u, xs=x_y_t, i=0, j=0)
    u_yy = dde.grad.hessian(ys=u, xs=x_y_t, i=1, j=1)

    a1 = (p1 - y)
    a2 = (x - p2)

    return nu * (u_xx + u_yy) - a1 * u_x - a2 * u_y - u_t


# 3. Weitere Bedingungen

# Simulationsdaten zu u
def gen_traindata(i):
    j = int(start_time / delta_t + i * skip_iterations)

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName("D:\Dokumente\Studium\Bachelorarbeit\data\convdiff_instat\\rotation\\1_solution_5_{}_0.vtu".format(j))
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

    return ob_points, ob_u

ob_points, ob_u = load_training_data(num_points)
observe_u = dde.icbc.PointSetBC(ob_points, ob_u, component=0)

# Dirichlet Randbedingungen
bc = dde.icbc.DirichletBC(geomtime, lambda _: 0.0, lambda _, on_boundary: on_boundary )

# Anfangsbedingungen
def initial_in_circle(x, on_initial):
    dist = (x[0] - 0.33)**2 + (x[1] - 0.33)**2
    return on_initial and dist < 0.17**2

def initial_rest(x, on_initial):
    return on_initial and not initial_in_circle(x, on_initial)

def initialFunction_in_circle(point):
    x1 = point[:, 0:1]
    x2 = point[:, 1:2]
    dist = (x1 - 0.33)**2 + (x2 - 0.33)**2
    return tf.cos(np.pi / 0.34 * tf.sqrt(dist))

ic_in_circle = dde.icbc.IC(geomtime, initialFunction_in_circle, initial_in_circle)
ic_rest = dde.icbc.IC(geomtime, lambda _: 0.0, initial_rest)

# 4. Kombination
data = dde.data.TimePDE(
    geomtime,
    ConvDiff_system,
    [ic_in_circle, ic_rest, bc, observe_u],
    num_domain=200,
    num_boundary=50,
    num_initial=50,
    anchors=ob_points
)

# 5. Das neuronale Netz
net = dde.nn.FNN([3] + [40] * 6 + [1], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([p1, p2], period=100, filename="variables.dat", precision=4)
model.compile("adam", lr=0.0002, external_trainable_variables=[p1, p2])
losshistory, train_state = model.train(epochs=100000, callbacks=[variable])

# 7. Plotten der Ergebnisse
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Variables:
lines = open("variables.dat", "r").readlines()
var_pred = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
l = var_pred.shape[0]
plt.figure()
plt.semilogy(range(0, l * 100, 100), var_pred[:, 0], "r-")
plt.semilogy(range(0, l * 100, 100), np.ones(var_pred[:, 0].shape) * p1_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen p1")

plt.figure()
plt.semilogy(range(0, l * 100, 100), var_pred[:, 1], "r-")
plt.semilogy(range(0, l * 100, 100), np.ones(var_pred[:, 1].shape) * p2_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen p2")

# Vorhersage und Lösung von u:
ob_points, ob_u = load_training_data(num=33*33*num_iterations)
prediction = model.predict(ob_points)

u_pred = prediction[:, 0]
u_true = ob_u[:, 0]
ob_x, ob_y, ob_t = ob_points[:, 0], ob_points[:, 1], ob_points[:, 2]

errors = np.zeros(num_iterations)
for i in range(0, num_iterations):
    t = start_time + delta_t * i * skip_iterations

    u_pred_t = u_pred[ob_t == t]
    u_true_t = u_true[ob_t == t]

    errors[i] = dde.metrics.l2_relative_error(u_true_t, u_pred_t)
plt.figure()
plt.plot(np.arange(start_time, start_time + delta_t * num_iterations * skip_iterations, delta_t * skip_iterations), errors)
plt.title("Relativer L2-Fehler für u")
plt.xlabel("t")
plt.show()

max_err = np.amax(np.abs(u_true - u_pred))

def plot_u_pred(i):
    # data for t
    t = start_time + delta_t * i * skip_iterations
    x_at_t = ob_x[ob_t == t]
    y_at_t = ob_y[ob_t == t]
    u_true_at_t = u_true[ob_t == t]
    u_pred_at_t = u_pred[ob_t == t]

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

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, np.abs(u_true_at_t - u_pred_at_t), levels=80, cmap="coolwarm", vmin=0, vmax=max_err)
    plt.colorbar(cntr2)
    plt.title("u: Fehler bei t=" + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)
    plt.show()

i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))
while i != -1:
    plot_u_pred(i)
    i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))
