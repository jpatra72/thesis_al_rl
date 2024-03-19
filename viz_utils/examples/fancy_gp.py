import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
f_mean, f_var, samples = calculate_gp_posterior(x_train, y_train, x_test, kernel, noise_lvl, n_samples=n_samples)
f_stdv = np.sqrt(np.diag(f_var))



# FUN PART

venv_name = "visualisation-with-matplotlib"  # change accordingly
c = load_colors(venv_name)

# load params
params = initialize_plot('README')  # specifies font size etc., adjust accordingly
plt.rcParams.update(params)

# CDC column width is e.g. 245pt, for double column plot change to 505pt
x, y = set_size(300,
                subplots=(1, 1),  # specify subplot layout for nice scaling
                fraction=1.)  # scale width/height
fig = plt.figure(figsize=(x, y))


def powspace(start, stop, power, num):
    start = np.power(start, 1 / float(power))
    stop = np.power(stop, 1 / float(power))
    return np.power(np.linspace(start, stop, num=num), power)


def fancyGPposterior(x: np.ndarray, mean: np.ndarray, stdv: np.ndarray,
                     color: list,
                     fade_out_color: np.ndarray = None,  # np.array([1, 1, 1]),
                     alpha: float = 0.9,
                     show_percentile=True,
                     plot_settings_mean: dict = {},
                     plot_settings_area: dict = {}):

    # create faded colormap
    N = 256
    color = np.array(color)
    vals = np.ones((N, 4))
    if fade_out_color is None:
        white = np.array([1, 1, 1])
        fade_out_color = white - 0.5 * (white - color)
    vals[:-int(N / 2), 0] = np.linspace(color[0], fade_out_color[0], int(N / 2))[::-1]
    vals[:-int(N / 2), 1] = np.linspace(color[1], fade_out_color[1], int(N / 2))[::-1]
    vals[:-int(N / 2), 2] = np.linspace(color[2], fade_out_color[2], int(N / 2))[::-1]

    vals[int(N / 2):, 0] = np.linspace(color[0], fade_out_color[0], int(N / 2))
    vals[int(N / 2):, 1] = np.linspace(color[1], fade_out_color[1], int(N / 2))
    vals[int(N / 2):, 2] = np.linspace(color[2], fade_out_color[2], int(N / 2))

    vals[int(N / 2):, 3] = powspace(0, alpha, 1, int(N / 2))[::-1]
    vals[:-int(N / 2), 3] = powspace(0, alpha, 1, int(N / 2))
    newcmp = ListedColormap(vals)

    plt.plot(x, mean, color=color, **plot_settings_mean)
    poly = plt.fill_between(x, mean - 3 * stdv, mean + 3 * stdv, color="none", lw=0, **plot_settings_area)
    if show_percentile:
        plt.plot(x, mean - 2 * stdv, color=color, lw=0.5, alpha=2 / 3 * alpha)
        plt.plot(x, mean + 2 * stdv, color=color, lw=0.5, alpha=2 / 3 * alpha)

    verts = np.vstack([p.vertices for p in poly.get_paths()])
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    xmin, xmax = x.min(), x.max()
    gradient = plt.imshow(np.array([np.interp(np.linspace(ymin, ymax, 200), [y1i, y2i], np.arange(2))
                                    for y1i, y2i in zip(mean - 3 * stdv, mean + 3 * stdv)]).T,
                          cmap=newcmp, aspect='auto', origin='lower', extent=[xmin, xmax, ymin, ymax])
    gradient.set_clip_path(poly.get_paths()[0], transform=plt.gca().transData)

plot_settings_mean = {"label": "Posterior Mean"}
fancyGPposterior(x_test.reshape(-1), f_mean, f_stdv, color=c["blau100"], plot_settings_mean=plot_settings_mean)

plt.plot(x_test, y_test, 'k')
labels = ['Posterior Samples'] + ["" for i in range(n_samples-1)]
plt.plot(x_test, samples.T, "k:", label=labels)
plt.errorbar(x_train.reshape(-1), y_train.reshape(-1), yerr=2 * np.ones_like(x_train).reshape(-1) * noise_lvl,
             fmt='.', color="orange", lw=0.5, capsize=4, elinewidth=1, markeredgewidth=1)

fig.subplots_adjust(bottom=-0.1, top=1.1, left=-0.1, right=1.1, )

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
# plt.savefig('../images/fancy_gp.pdf', format='pdf')
#
# # to save as pgf
# plt.savefig('../images/fancy_gp.pgf', format='pgf', backend="pgf")
#
# to save as png (DON'T USE PNG IN PAPERS! USE PDF!) -- for GIF, pngs are needed
plt.savefig('../images/fancy_gp.png', format='png', dpi=300)

plt.close()