import deepxde as dde
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN
import random
import matplotlib.pyplot as plt
import re
import imageio
import os

nu = dde.Variable(0.1)
rho = dde.Variable(1.1)

rho_true = 1.0
nu_true = 0.01

start_time = 80.0
skip_iterations = 1
delta_t = 0.1
num_iterations = 31

num_points = 1000

# 1. Das Gebiet
geom = dde.geometry.Rectangle([20.0, 4.0], [30.0, 12.0])
timedomain = dde.geometry.TimeDomain(start_time,start_time + delta_t * (float(num_iterations * skip_iterations) - 1.0))
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Die PDG
def NavierStokes_system(x, y):

    vx = y[:, 0:1]
    vy = y[:, 1:2]

    dvx_x = dde.grad.jacobian(y, x, i=0, j=0)
    dvx_y = dde.grad.jacobian(y, x, i=0, j=1)
    dvx_t = dde.grad.jacobian(y, x, i=0, j=2)

    dvy_x = dde.grad.jacobian(y, x, i=1, j=0)
    dvy_y = dde.grad.jacobian(y, x, i=1, j=1)
    dvy_t = dde.grad.jacobian(y, x, i=1, j=2)

    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_y = dde.grad.jacobian(y, x, i=2, j=1)

    dvx_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    dvx_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dvy_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    dvy_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    continuity = dvx_x + dvy_y
    x_momentum = dvx_t + (vx * dvx_x + vy * dvx_y) + dp_x / rho - nu * (dvx_xx + dvx_yy)
    y_momentum = dvy_t + (vx * dvy_x + vy * dvy_y) + dp_y / rho - nu * (dvy_xx + dvy_yy)
    return [continuity, x_momentum, y_momentum]


# 3. Weitere Bedingungen

# Simulationsdaten zu u

def read_traindata_for_iteration_and_file(i, k):
    j = str(int(start_time / delta_t + i * skip_iterations)).zfill(4)
    path = "D:\Dokumente\Studium\Bachelorarbeit\data\\nav_sto\data_lvl_4\\sol_channel_ilupp{}_{}.vtu".format(j,k)
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    data = reader.GetOutput()

    points_3D = VN.vtk_to_numpy(data.GetPoints().GetData())
    points_3D[:,2] = points_3D[:,2] + (start_time + i * delta_t * skip_iterations)
    vx = VN.vtk_to_numpy(data.GetPointData().GetArray("vx"))
    vx = vx.reshape(vx.shape[0], 1)
    vy = VN.vtk_to_numpy(data.GetPointData().GetArray("vy"))
    vy = vy.reshape(vy.shape[0], 1)
    p = VN.vtk_to_numpy(data.GetPointData().GetArray("p"))
    p = p.reshape(p.shape[0], 1)

    num_data_points = points_3D.shape[0]
    allData = np.zeros((num_data_points, 6))

    allData[0:num_data_points, 0:3] = points_3D
    allData[0:num_data_points, 3:4] = vx
    allData[0:num_data_points, 4:5] = vy
    allData[0:num_data_points, 5:6] = p

    allData = allData[allData[:, 0] >= 20.0]
    allData = allData[allData[:, 0] <= 30.0]
    allData = allData[allData[:, 1] <= 12.0]
    allData = allData[allData[:, 1] >= 4.0]

    return allData

def read_traindata_for_iteration(i):
    allData = read_traindata_for_iteration_and_file(i, 0)
    for k in range(1, 48):
        allData_i = read_traindata_for_iteration_and_file(i, k)
        allData = np.concatenate([allData, allData_i])

    allDataUnique = np.unique(allData, axis=0)

    points = allDataUnique[:, 0:3]
    vx = allDataUnique[:, 3:4]
    vy = allDataUnique[:, 4:5]
    p = allDataUnique[:, 5:6]

    return points, vx, vy, p

def read_traindata():
    points, vx, vy, p = read_traindata_for_iteration(0)
    for i in range(1, num_iterations):
        points_i, vx_i, vy_i, p_i = read_traindata_for_iteration(i)
        points = np.concatenate([points, points_i])
        vx = np.concatenate([vx, vx_i])
        vy = np.concatenate([vy, vy_i])
        p = np.concatenate([p, p_i])

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
observe_p = dde.icbc.PointSetBC(ob_points, ob_p, component=2)

# 4. Kombination
data = dde.data.TimePDE(
    geomtime,
    NavierStokes_system,
    [observe_vx, observe_vy, observe_p],
    num_domain=400,
    num_boundary=200,
    anchors=ob_points,
)

# 5. Das neuronale Netz
net = dde.nn.FNN([3] + [50] * 6 + [3], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([nu, rho], period=100, filename="variables.dat", precision=4)
model.compile("adam", lr=0.002, external_trainable_variables=[nu, rho])
losshistory, train_state = model.train(epochs=10000, callbacks=[variable], display_every=100, disregard_previous_best=True)

model.compile("adam", lr=0.001, external_trainable_variables=[nu, rho])
losshistory, train_state = model.train(epochs=10000, callbacks=[variable], display_every=100, disregard_previous_best=True)

model.compile("adam", lr=0.0005, external_trainable_variables=[nu, rho])
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


plt.figure()
plt.semilogy(range(0, l * 100, 100), var_pred[:, 1], "r-")
plt.semilogy(range(0, l * 100, 100), np.ones(var_pred[:, 1].shape) * rho_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen rho")

plt.show()

# Vorhersage und Lösung:
prediction = model.predict(points)
vx_pred = prediction [:, 0]
vy_pred = prediction [:, 1]
p_pred = prediction [:, 2]
vx_true= vx[:, 0]
vy_true= vy[:, 0]
p_true= p[:, 0]
ob_x, ob_y, ob_t  = points[:, 0], points[:, 1], points[:, 2]

print("l2 relative error for vx: " + str(dde.metrics.l2_relative_error(vx_true, vx_pred)))
print("l2 relative error for vy: " + str(dde.metrics.l2_relative_error(vy_true, vy_pred)))

# make gifs

max_vx = max(np.amax(vx_pred), np.amax(vx_true))
min_vx = min(np.amin(vx_pred), np.amin(vx_true))
max_vy = max(np.amax(vy_pred), np.amax(vy_true))
min_vy = min(np.amin(vy_pred), np.amin(vy_true))
max_p_pred = np.amax(p_pred)
min_p_pred = np.amin(p_pred)
max_p_true = np.amax(p_true)
min_p_true = np.amin(p_true)

filenames = [[],[],[],[],[],[]]

def plot_pred_for_gif(i):
    # data for t
    t = start_time + delta_t * i * skip_iterations
    x_at_t = ob_x[ob_t == t]
    y_at_t = ob_y[ob_t == t]
    vx_pred_at_t = vx_pred[ob_t == t]
    vy_pred_at_t = vy_pred[ob_t == t]
    p_pred_at_t = p_pred[ob_t == t]
    vx_true_at_t= vx_true[ob_t == t]
    vy_true_at_t = vy_true[ob_t == t]
    p_true_at_t = p_true[ob_t == t]
    
    # Für vx
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vx_pred_at_t, levels=80, cmap="rainbow", vmin=min_vx, vmax=max_vx)
    plt.colorbar(cntr2)
    plt.title("vx: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'vx_pred_{i}.png'
    filenames[0].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vx_true_at_t, levels=80, cmap="rainbow", vmin=min_vx, vmax=max_vx)
    plt.colorbar(cntr2)
    plt.title("vx: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'vx_true_{i}.png'
    filenames[1].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()


    # Für vy
    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vy_pred_at_t, levels=80, cmap="rainbow", vmin=min_vy, vmax=max_vy)
    plt.colorbar(cntr2)
    plt.title("vy: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'vy_pred_{i}.png'
    filenames[2].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vy_true_at_t, levels=80, cmap="rainbow", vmin=min_vy, vmax=max_vy)
    plt.colorbar(cntr2)
    plt.title("vy: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'vy_true_{i}.png'
    filenames[3].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()


    # Für p
    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, p_pred_at_t, levels=80, cmap="rainbow")
    plt.colorbar(cntr2)
    plt.title("p: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'p_pred_{i}.png'
    filenames[4].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, p_true_at_t, levels=80, cmap="rainbow")
    plt.colorbar(cntr2)
    plt.title("p: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    # create file name and append it to a list
    filename = f'p_true_{i}.png'
    filenames[5].append(filename)
    
    # save frame
    plt.savefig(filename)
    plt.close()

for i in range(num_iterations):
    plot_pred_for_gif(i)

# build gif
gifnames = ['vx_pred.gif', 'vx_true.gif', 'vy_pred.gif', 'vy_true.gif', 'p_pred.gif', 'p_true.gif']

for i in range(6):
    with imageio.get_writer(gifnames[i], mode='I') as writer:
        for filename in filenames[i]:
            image = imageio.imread(filename)
            writer.append_data(image)
        
    # Remove files
    for filename in set(filenames[i]):
        os.remove(filename)


# show iteration

def plot_pred(i):
    # data for t
    t = start_time + delta_t * i * skip_iterations
    x_at_t = ob_x[ob_t == t]
    y_at_t = ob_y[ob_t == t]
    vx_pred_at_t = vx_pred[ob_t == t]
    vy_pred_at_t = vy_pred[ob_t == t]
    p_pred_at_t = p_pred[ob_t == t]
    vx_true_at_t= vx_true[ob_t == t]
    vy_true_at_t = vy_true[ob_t == t]
    p_true_at_t = p_true[ob_t == t]
    
    # Für vx
    vmax = max(np.amax(vx_pred_at_t), np.amax(vx_true_at_t))
    vmin = min(np.amin(vx_pred_at_t), np.amin(vx_true_at_t))

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vx_pred_at_t, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.colorbar(cntr2)
    plt.title("vx: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vx_true_at_t, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.colorbar(cntr2)
    plt.title("vx: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)


    # Für vy
    vmax = max(np.amax(vy_pred_at_t), np.amax(vy_true_at_t))
    vmin = min(np.amin(vy_pred_at_t), np.amin(vy_true_at_t))

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vy_pred_at_t, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.colorbar(cntr2)
    plt.title("vy: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, vy_true_at_t, levels=80, cmap="rainbow", vmin=vmin, vmax=vmax)
    plt.colorbar(cntr2)
    plt.title("vy: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)


    # Für p
    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, p_pred_at_t, levels=80, cmap="rainbow")
    plt.colorbar(cntr2)
    plt.title("p: Vorhersage des PINNs für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

    plt.figure()
    cntr2 = plt.tricontourf(x_at_t, y_at_t, p_true_at_t, levels=80, cmap="rainbow")
    plt.colorbar(cntr2)
    plt.title("p: Simulierte Lösung mit Hiflow³ für t= " + str(t), fontsize=10)
    plt.axis("scaled")
    plt.xlabel("X", fontsize=10, family="Arial")
    plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)


    plt.show()


i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))
while i != -1:
    plot_pred(i)
    i = int(input("Which iteration do you want to see? Enter int from 0 to {}, Enter -1 to end programm:".format(num_iterations - 1)))