import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import re

b = dde.Variable(0.6)
b_true = 0.5

# 1. Das Gebiet
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# 2. Die PDG
def pde(x_and_t, u):
    u_x = dde.grad.jacobian(ys=u, xs=x_and_t, j=0)
    u_t = dde.grad.jacobian(ys=u, xs=x_and_t, j=1)
    return u_t + b * u_x

# 3. Weitere Bedingungen
def g(x):
    return np.cos(np.pi * x)

def sol(x_and_t):
    x = x_and_t[:, 0:1]
    t = x_and_t[:, 1:]
    return g(x - b_true * t)
    

def gen_traindata(num):
    xvals = np.linspace(-1, 1, 2*num + 1).reshape(2*num + 1, 1)
    tvals = np.linspace(0,1, num + 1)
    x_and_tvals = np.dstack(np.meshgrid(xvals, tvals)).reshape(-1, 2)
    uvals = sol(x_and_tvals)
    return x_and_tvals, uvals

bc = dde.icbc.DirichletBC(geomtime, sol, lambda _, on_b: on_b)
ob_x_and_t, ob_u = gen_traindata(10) # 10 Punkten pro Einheit, dh. 11*21 = 231 Punkte
observe_u = dde.icbc.PointSetBC(ob_x_and_t, ob_u)

# 4. Kombination
data = dde.data.TimePDE(
    geomtime,
    pde,
    [observe_u, bc],
    num_domain=200,
    num_boundary=20,
    anchors=ob_x_and_t,
    num_test=1000,
)

# 5. Das neuronale Netz
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")

# 6. Der Trainingsprozess
model = dde.Model(data, net)
model.compile("adam", lr=0.01, external_trainable_variables=[b])
variable = dde.callbacks.VariableValue([b], period=20, filename="variables.dat")
losshistory, train_state = model.train(epochs=2000, callbacks=[variable], display_every=100)

# 7. Plotten der Ergebnisse
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Variable b:
lines = open("variables.dat", "r").readlines()
b_pred = np.array(
    [
        np.fromstring(
            min(re.findall(re.escape("[") + "(.*?)" + re.escape("]"), line), key=len),
            sep=",",
        )
        for line in lines
    ]
)
l = b_pred.shape[0]
plt.semilogy(range(0, l * 20, 20), b_pred, "r-")
plt.semilogy(range(0, l * 20, 20), np.ones(b_pred.shape) * b_true, "k--")
plt.legend(["Variable", "echter Wert"], loc="lower right")
plt.xlabel("Epochs")
plt.title("Entwicklung der Variablen b")
plt.show()

# Vorhersage und Lösung von u:
[ob_points, ob_u] = gen_traindata(100)
prediction = model.predict(ob_points)

u_pred = prediction[:, 0]
u_true = ob_u[:, 0]
ob_x, ob_t = ob_points[:, 0], ob_points[:, 1]

fig, ax = plt.subplots(2, 1)
cntr0 = ax[0].tricontourf(ob_x, ob_t, u_pred, levels=80, cmap="coolwarm", vmin=-1, vmax=1)
cntr1 = ax[1].tricontourf(ob_x, ob_t, u_true, levels=80, cmap="coolwarm", vmin=-1, vmax=1)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cntr1, cax=cbar_ax)
ax[0].set_title("u: Vorhersage des PINNs", fontsize=9.5)
ax[0].axis("scaled")
ax[0].set_xlabel("x", fontsize=10, family="Arial")
ax[0].set_ylabel("t", fontsize=10, family="Arial", rotation=0)
ax[1].set_title("u: Lösung der PDG", fontsize=9.5)
ax[1].axis("scaled")
ax[1].set_xlabel("x", fontsize=10, family="Arial")
ax[1].set_ylabel("t", fontsize=10, family="Arial", rotation=0)
fig.tight_layout()
plt.show()