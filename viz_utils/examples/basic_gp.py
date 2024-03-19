import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from viz_utils.calculate_stuff import calculate_gp_posterior
from viz_utils.visualization_utils import set_size, initialize_plot, load_colors


def function(x):
    return np.sin(x)


def kernel(X1, X2, l=0.5, sigma_f=0.5):
    sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * sqdist)


noise_lvl = 0.05
np.random.seed(50)
x_train = np.array([-1, -0.5, 1, ]).reshape(-1, 1)
y_train = function(x_train) + np.random.normal(scale=noise_lvl, size=x_train.shape)
x_test = np.linspace(-2, 2, 100).reshape(-1, 1)
y_test = function(x_test)

n_samples = 3
mean, var, samples = calculate_gp_posterior(x_train, y_train, x_test, kernel, noise_lvl, n_samples=n_samples)
stdv = np.sqrt(np.diag(var))



# FUN PART

venv_name = "viz_utils"  # change accordingly
c = load_colors(venv_name)

# load params
params = initialize_plot('README')  # specifies font size etc., adjust accordingly
plt.rcParams.update(params)

# CDC column width is e.g. 245pt, for double column plot change to 505pt

# 398.33862pt for thesis report
x, y = set_size(300,
                subplots=(1, 1),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
fig = plt.figure(figsize=(x, y))

plt.plot(x_test.reshape(-1), y_test, 'k')
plt.plot(x_train, y_train, 'k*', )

# posterior
plt.plot(x_test.reshape(-1), mean, color=c['bordeaux100'], label='Posterior Mean')
plt.fill_between(x_test.reshape(-1), mean - 2 * stdv, mean + 2 * stdv, alpha=0.7, color=c['bordeaux50'], lw=0)

labels = ['Posterior Samples'] + ["" for i in range(n_samples-1)]
plt.plot(x_test, samples.T, color=c['bordeaux75'], linewidth=0.5, label=labels)

plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.ylabel('$f(x)$')
plt.xlabel('$x$')

plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
           borderaxespad=0, ncol=2, framealpha=1, labelspacing=0.2, )

# adjust margins, manually adjusting will always yield nicer plots!
fig.subplots_adjust(bottom=0.17, top=0.85, left=0.16, right=0.96)

# plt.show()  # comment out, if you want so save the plot

# # to save as pdf
# plt.savefig('../images/basic_gp.pdf', format='pdf')
#
# # to save as pgf
# plt.savefig('../images/basic_gp.pgf', format='pgf', backend="pgf")
#
# to save as png (DON'T USE PNG IN PAPERS! USE PDF!) -- for GIF, pngs are needed
plt.savefig('../images/basic_gp.png', format='png', dpi=300)

plt.close()