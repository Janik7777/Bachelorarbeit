import deepxde as dde
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN
import random
import matplotlib.pyplot as plt
import re


nu = dde.Variable(0.1)

rho = 1.0
nu_true = 0.1

num_points = 400

# 1. Das Gebiet

geom = dde.geometry.Rectangle([16.0, 1.0], [40.0, 15.0])

# 2. Die PDG
def NavierStokes_system(x, y):

    vx = y[:, 0:1]
    vy = y[:, 1:2]

    dvx_x = dde.grad.jacobian(y, x, i=0, j=0)
    dvx_y = dde.grad.jacobian(y, x, i=0, j=1)

    dvy_x = dde.grad.jacobian(y, x, i=1, j=0)
    dvy_y = dde.grad.jacobian(y, x, i=1, j=1)

    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)

    dvx_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dvx_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dvy_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dvy_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = dvx_x + dvy_y
    x_momentum = (vx * dvx_x + vy * dvx_y) + dp_x / rho - nu * (dvx_xx + dvx_yy)
    y_momentum = (vx * dvy_x + vy * dvy_y) + dp_y / rho - nu * (dvy_xx + dvy_yy)
    return [continuity, x_momentum, y_momentum]


# 3. Weitere Bedingungen

# Simulationsdaten zu u

def read_traindata_for_file(k):
    
    path = "D:\Dokumente\Studium\Bachelorarbeit\data\\nav_sto_stationary\data_lvl_3\\sol_channel_ilupp0800_{}.vtu".format(k)
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()

    points_3D = VN.vtk_to_numpy(data.GetPoints().GetData())
    points_2D = points_3D[:,0:2]
    vx = VN.vtk_to_numpy(data.GetPointData().GetArray("vx"))
    vx = vx.reshape(vx.shape[0], 1)
    vy = VN.vtk_to_numpy(data.GetPointData().GetArray("vy"))
    vy = vy.reshape(vy.shape[0], 1)
    p = VN.vtk_to_numpy(data.GetPointData().GetArray("p"))
    p = p.reshape(p.shape[0], 1)

    num_data_points = points_3D.shape[0]
    allData = np.zeros((num_data_points, 5))

    allData[0:num_data_points, 0:2] = points_2D
    allData[0:num_data_points, 2:3] = vx
    allData[0:num_data_points, 3:4] = vy
    allData[0:num_data_points, 4:5] = p

    allData = allData[allData[:, 0] >= 16.0]
    allData = allData[allData[:, 1] <= 15.0]
    allData = allData[allData[:, 1] >= 1.0]

    return allData

def read_traindata():
    allData = read_traindata_for_file(0)
    for k in range(1, 16):
        allData_i = read_traindata_for_file(k)
        allData = np.concatenate([allData, allData_i])

    allDataUnique = np.unique(allData, axis=0)

    points = allDataUnique[:, 0:2]
    vx = allDataUnique[:, 2:3]
    vy = allDataUnique[:, 3:4]
    p = allDataUnique[:, 4:5]

    return points, vx, vy, p

points, vx, vy, p = read_traindata()

def gen_traindata(num):

    indices = random.sample(range(0, points.shape[0]), num)
    ob_points = np.take(points, indices, 0) 
    ob_vx = np.take(vx, indices, 0)
    ob_vy = np.take(vy, indices, 0)
    ob_p = np.take(p, indices, 0)
    return ob_points, ob_vx, ob_vy, ob_p

ob_points, ob_vx, ob_vy, ob_p = gen_traindata(num_points)


observe_vx = dde.icbc.PointSetBC(ob_points, ob_vx, component=0)
observe_vy = dde.icbc.PointSetBC(ob_points, ob_vy, component=1)

# 4. Kombination
data = dde.data.PDE(
    geom,
    NavierStokes_system,
    [observe_vx, observe_vy],
    num_domain=400,
    num_boundary=200,
    anchors=ob_points,
)

# 5. Das neuronale Netz
net = dde.nn.FNN([2] + [50] * 6 + [3], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([nu], period=100, filename="variables.dat", precision=4)
model.compile("adam", lr=0.002, external_trainable_variables=[nu])
losshistory, train_state = model.train(epochs=10000, callbacks=[variable], display_every=100, disregard_previous_best=True)

model.compile("adam", lr=0.001, external_trainable_variables=[nu])
losshistory, train_state = model.train(epochs=10000, callbacks=[variable], display_every=100, disregard_previous_best=True)

model.compile("adam", lr=0.0001, external_trainable_variables=[nu])
losshistory, train_state = model.train(epochs=10000, callbacks=[variable], display_every=100, disregard_previous_best=True)

# 7. Plotten der Ergebnisse
dde.saveplot(losshistory, train_state, issave=True, isplot=True)


# Variable nu:
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
plt.semilogy(range(0, l * 100, 100), np.ones(var_pred[:, 0].shape) * nu_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen nu")

plt.show()

# Vorhersage und Lösung:
prediction = model.predict(points)
vx_pred = prediction [:, 0]
vy_pred = prediction [:, 1]
p_pred = prediction [:, 2]
vx_true= vx[:, 0]
vy_true= vy[:, 0]
p_true= p[:, 0]
x, y = points[:, 0], points[:, 1]

print("l2 relative error for vx: " + str(dde.metrics.l2_relative_error(vx_true, vx_pred)))
print("l2 relative error for vy: " + str(dde.metrics.l2_relative_error(vy_true, vy_pred)))


    
# Für vx
vmax = max(np.amax(vx_pred), np.amax(vx_true))
vmin = min(np.amin(vx_pred), np.amin(vx_true))

plt.figure()
cntr2 = plt.tricontourf(x, y, vx_pred, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("vx: Vorhersage des PINNs", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

plt.figure()
cntr2 = plt.tricontourf(x, y, vx_true, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("vx: Simulierte Lösung mit Hiflow³", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)


# Für vy
vmax = max(np.amax(vy_pred), np.amax(vy_true))
vmin = min(np.amin(vy_pred), np.amin(vy_true))

plt.figure()
cntr2 = plt.tricontourf(x, y, vy_pred, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("vy: Vorhersage des PINNs", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

plt.figure()
cntr2 = plt.tricontourf(x, y, vy_true, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("vy: Simulierte Lösung mit Hiflow³", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)


# Für p
plt.figure()
cntr2 = plt.tricontourf(x, y, p_pred, levels=80, cmap="rainbow")
plt.colorbar(cntr2)
plt.title("p: Vorhersage des PINNs", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

plt.figure()
cntr2 = plt.tricontourf(x, y, p_true, levels=80, cmap="rainbow")
plt.colorbar(cntr2)
plt.title("p: Simulierte Lösung mit Hiflow³", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

plt.show()
