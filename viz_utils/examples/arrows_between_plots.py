import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors


# Our 2-dimensional distribution will be over variables X and Y
N = 60
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([0., .5])
Sigma = np.array([[1., -0.5], [-0.5, 1.5]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y


def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)

    return np.exp(-fac / 2) / N


# The distribution on the variables X, Y packed into pos.
Z1 = multivariate_gaussian(pos, mu, Sigma)
Z2 = multivariate_gaussian(pos, mu, Sigma + np.array([[0.5, 1], [1, -0.5]]))

# FUN PART
venv_name = "visualisation-with-matplotlib"  # change accordingly
c = load_colors(venv_name)

# load params
params = initialize_plot('README')  # specifies font size etc., adjust accordingly
plt.rcParams.update(params)


x, y = set_size(300, subplots=(1, 2), fraction=1)
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(x, y * 2))
ax1.contour(X, Y, Z1)
ax1.set_xlim([-3, 3])
ax1.set_ylim([-3, 3])
ax1.set_title("From the window ...")
ax1.set_ylabel("$x_2$")
ax1.set_xlabel("$x_1$")

ax2.contour(X, Y, Z2)
ax2.set_xlim([-3, 3])
ax2.set_ylim([-3, 3])
ax2.set_title(".. to the wall!")
ax2.set_ylabel("$x_2$")
ax2.set_xlabel("$x_1$")

#### BOTTOM ARROW
# Create the arrow
# 1. Get transformation operators for axis and figure
ax0tr = ax1.transData  # Axis 0 -> Display
ax1tr = ax2.transData  # Axis 1 -> Display
figtr = fig.transFigure.inverted()  # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB = figtr.transform(ax0tr.transform((2, -2)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE = figtr.transform(ax1tr.transform((-1., -1.5)))
# 4. Create the patch
from matplotlib.patches import FancyArrowPatch

arrow = FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc="k", connectionstyle="arc3,rad=0.5", arrowstyle='simple', alpha=0.7,
    mutation_scale=20.
)
# 5. Add patch to list of objects to draw onto the figure
fig.patches.append(arrow)

# UPPER ARROW
ax0tr = ax1.transData  # Axis 0 -> Display
ax1tr = ax2.transData  # Axis 1 -> Display
figtr = fig.transFigure.inverted()  # Display -> Figure
# 2. Transform arrow start point from axis 0 to figure coordinates
ptB2 = figtr.transform(ax1tr.transform((-1., 2)))
# 3. Transform arrow end point from axis 1 to figure coordinates
ptE2 = figtr.transform(ax0tr.transform((2, 2)))
arrow2 = FancyArrowPatch(
    ptB2, ptE2, transform=fig.transFigure,  # Place arrow in figure coord system
    fc="k", connectionstyle="arc3,rad=0.3", arrowstyle='simple', alpha=0.7,
    mutation_scale=20.
)
# 5. Add patch to list of objects to draw onto the figure
fig.patches.append(arrow2)

for ax in fig.get_axes():
    ax.label_outer()

# adjust margins, manually adjusting will always yield nicer plots!
fig.subplots_adjust(bottom=0.17, top=0.85, left=0.16, right=0.96)

# plt.show()  # comment out, if you want so save the plot

# # to save as pdf
# plt.savefig('../images/arrow_between_plots.pdf', format='pdf')
#
# # to save as pgf
# plt.savefig('../images/arrow_between_plots.pgf', format='pgf', backend="pgf")
#
# to save as png (DON'T USE PNG IN PAPERS! USE PDF!) -- for GIF, pngs are needed
plt.savefig('../images/arrow_between_plots.png', format='png', dpi=300)

plt.close()