import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors


TIME_SCALE = 0.1
X_SCALE = 0.25
alpha = 4.
s = .5
offset = 5
s2 = 0.

# MESSY FUNCTIONS
def long_riley_function_linear(x, t):
    t_lin = np.where(t < 140., t, 0.)
    t_lin = t_lin.unsqueeze(1) if len(t_lin.shape) < 2 else t_lin
    fx = alpha * (X_SCALE * x - s2 - 0.01 * t_lin) ** 2 + offset
    mask = np.logical_and(t > 140., t < 225.)
    t = np.where(mask, 50., t)
    t = np.where(t < 225., t, -50.)
    fxt = 2 * np.multiply(X_SCALE * x, np.sin(TIME_SCALE * t)) - np.square(np.cos(TIME_SCALE * t))
    return fx + fxt


def min_long_riley_function_linear(x, t):
    t_lin = np.where(t < 140, t, 0.)
    mask = np.logical_and(t > 140., t < 225.)
    t = np.where(mask, 50., t)
    t = np.where(t < 225., t, -50.)
    dfx = 1 / X_SCALE * (s2 + 0.01 * t_lin - 1 / alpha * np.sin(TIME_SCALE * t))
    return dfx


SHOW_XBOUNDS = [-8, 10]
SHOW_YBOUNDS = [-5, 15]
ACQ_BOUNDS = [-5, 9]
TIME_HORIZON = 300
TRAINING_POINTS = 15

x = np.linspace(SHOW_XBOUNDS[0], SHOW_XBOUNDS[1], 100)
t = np.linspace(0, TIME_HORIZON, 100)
X, T = np.meshgrid(x, t)

# multi_modal
output = long_riley_function_linear(X, T)
minimum = min_long_riley_function_linear(x, t)


# FUN PART
venv_name = "visualisation-with-matplotlib"  # change accordingly
c = load_colors(venv_name)

# load params
params = initialize_plot('README')  # specifies font size etc., adjust accordingly
plt.rcParams.update(params)

x, y = set_size(300, fraction=1)
fig = plt.figure(figsize=(x, y))

v = np.linspace(3, 33, 15)
cnt = plt.contourf(T, X, output, v, cmap='viridis', extend='both', vmin=2, vmax=34, )
for ct in cnt.collections:
    ct.set_edgecolor("face")

plt.plot(t, np.ones(len(t)) * ACQ_BOUNDS[0], '-', color=c['schwarz100'], label='$\mathcal{X}$')
plt.plot(t, np.ones(len(t)) * ACQ_BOUNDS[1], '-', color=c['schwarz100'], )
plt.plot(t, minimum, 'w-', label='$x_t^*$', lw=1)
plt.xlim([0, TIME_HORIZON])
plt.ylim(SHOW_XBOUNDS)
plt.ylabel('$x$')
plt.xlabel('$t$')
legend1 = plt.legend(loc="lower right", frameon=True, facecolor=c['blau50'], ncol=4, edgecolor='k', borderaxespad=0,
                     framealpha=1)
frame1 = legend1.get_frame()
frame1.set_boxstyle('Square', pad=0)
frame1.set_linewidth(0.75)

# fraction and pad are magic numbers
fig.colorbar(cnt, fraction=0.046, pad=0.02, ticks=[3, 9, 15, 21, 27, 33], label='$f_t(x)$')
plt.clim(3, 33)  # orientation="horizontal"

for ax in fig.get_axes():
    ax.label_outer()

fig.subplots_adjust(bottom=0.15, top=0.95, left=0.11, right=0.9)

# plt.show()  # comment out, if you want so save the plot

# # to save as pdf
# plt.savefig('../images/contourf_plot.pdf', format='pdf')
#
# # to save as pgf
# plt.savefig('../images/contourf_plot.pgf', format='pgf', backend="pgf")
#
# to save as png (DON'T USE PNG IN PAPERS! USE PDF!) -- for GIF, pngs are needed
plt.savefig('../images/contourf_plot.png', format='png', dpi=300)

plt.close()