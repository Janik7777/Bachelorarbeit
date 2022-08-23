import deepxde as dde
import numpy as np
from vtk import vtkXMLUnstructuredGridReader
from vtk.util import numpy_support as VN
import matplotlib.pyplot as plt
from deepxde.backend import tf
import re

nu = dde.Variable(0.6)
nu_true = 0.5
a1 = 1.0
a2 = 0

# 1. Das Gebiet
geom = dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0])

# 2. Die PDG
def ConvDiff_system(points, u):
    """Convdiff
    nu * laplace(u) = (a1,a2) * grad(u)
    """
    u_x = dde.grad.jacobian(ys=u, xs=points, j=0)
    u_y = dde.grad.jacobian(ys=u, xs=points, j=1)
    u_xx = dde.grad.hessian(ys=u, xs=points, i=0, j=0)
    u_yy = dde.grad.hessian(ys=u, xs=points, i=1, j=1)

    return nu * (u_xx + u_yy) - a1 * u_x - a2 * u_y


# 3. Weitere Bedingungen

# Simulationsdaten zu u
def read_data():
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName("D:\Dokumente\Studium\Bachelorarbeit\data\convdiff_stat\StabConvDiff_0.5_1_0_quadratic_left.vtu")
    reader.Update()
    data = reader.GetOutput()
    
    points_3D = VN.vtk_to_numpy(data.GetPoints().GetData())
    points_2D = points_3D[:, 0:2]
    u = VN.vtk_to_numpy(data.GetPointData().GetArray("u"))
    u = u.reshape(4096, 1)

    return points_2D, u

def gen_traindata():
    points, u = read_data()
    
    indices = np.array([682])
    for i in range(683, 4096):
        [x, y] = points[i]
        if np.abs(y - 0.5) < 0.001:
            indices = np.append(indices, i)
    '''
    indices = np.array([642])
    for i in range(643, 4096):
        [x, y] = points[i]
        if np.abs(y - 0.5) < 0.1 and np.abs(x - 0.4) < 0.1:
            indices = np.append(indices, i)
    '''
    ob_points = np.take(points, indices, 0) 
    ob_u = np.take(u, indices, 0)
    return ob_points, ob_u


ob_points, ob_u = gen_traindata()
'''
print(ob_points.shape)
plt.scatter(ob_points[:, 0:1], ob_points[:, 1:2])
plt.show()
'''

observe_u = dde.icbc.PointSetBC(ob_points, ob_u, component=0)

# Dirichlet Randbedingungen
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundaryFunction_left(point):
    x2 = point[:, 1:2]
    return 4.0 * (1.0 - x2) * x2

bc_left = dde.icbc.DirichletBC(geom, boundaryFunction_left, boundary_left)

def boundary_rest(x, on_boundary):
    return on_boundary and not np.isclose(x[0], 0)

bc_rest = dde.icbc.DirichletBC(geom, lambda _: 0, boundary_rest)

# 4. Kombination
data = dde.data.PDE(
    geom,
    ConvDiff_system,
    [bc_left, bc_rest, observe_u],
    num_domain=200,
    num_boundary=50,
    anchors=ob_points,
    num_test=1000
)

# 5. Das neuronale Netz
net = dde.nn.FNN([2] + [40] * 4 + [1], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
variable = dde.callbacks.VariableValue([nu], period=100, filename="variables.dat", precision=4)
model.compile("adam", lr=0.0002, external_trainable_variables=[nu])
losshistory, train_state = model.train(epochs=300000, callbacks=[variable])

# 7. Plotten der Ergebnisse
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Variable nu:
lines = open("variables.dat", "r").readlines()
nu_pred = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
l = nu_pred.shape[0]
plt.semilogy(range(0, l * 100, 100), nu_pred, "r-")
plt.semilogy(range(0, l * 100, 100), np.ones(nu_pred.shape) * nu_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen nu")

# Vorhersage und Lösung von u:
points, u = read_data()
u_pred = model.predict(points)[:, 0]
u_true= u[:, 0]
x, y = points[:, 0], points[:, 1]

vmax = max(np.amax(u_pred), np.amax(u_true))
vmin = min(np.amin(u_pred), np.amin(u_true))

print("l2 relative error for u: " + str(dde.metrics.l2_relative_error(u_true, u_pred)))
plt.figure()
cntr2 = plt.tricontourf(x, y, u_pred, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("u: Vorhersage des PINNs", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)

plt.figure()
cntr2 = plt.tricontourf(x, y, u_true, levels=80, cmap="coolwarm", vmin=vmin, vmax=vmax)
plt.colorbar(cntr2)
plt.title("u: Simulierte Lösung mit Hiflow³", fontsize=10)
plt.axis("scaled")
plt.xlabel("X", fontsize=10, family="Arial")
plt.ylabel("Y", fontsize=10, family="Arial", rotation=0)
plt.show()