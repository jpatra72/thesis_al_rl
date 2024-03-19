import matplotlib.colors as mcolors
import matplotlib.patches as mpatch
import matplotlib.pyplot as plt

from viz_utils.visualization_utils import load_colors


colors = load_colors("visualizations-with-matplotlib")

fig = plt.figure(figsize=[9, 5])
ax = fig.add_axes([0, 0, 1, 1])

n_groups = 3
n_rows = len(colors) // n_groups + 1

for j, color_name in enumerate(colors.keys()):
    color = colors[color_name]

    # Pick text colour based on perceived luminance.
    rgba = color + [0]
    luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]

    col_shift = (j // n_rows) * 3
    y_pos = j % n_rows
    text_args = dict(fontsize=10)
    ax.add_patch(mpatch.Rectangle((0 + col_shift, y_pos), 1, 1, color=color))
    ax.add_patch(mpatch.Rectangle((1 + col_shift, y_pos), 0.5, 1, color=color))
    ax.text(2.2 + col_shift, y_pos + .7, f'"{color_name}"',
            color='k', ha='center', **text_args)
    # ax.text(1.5 + col_shift, y_pos + .7, color_name,
    #         color=css4_text_color, ha='center', **text_args)
    # ax.text(1.5 + col_shift, y_pos + .7, f'  {color_name}', **text_args)

for g in range(n_groups):
    ax.hlines(range(n_rows), 3*g, 3*g + 2.8, color='0.7', linewidth=1)
    ax.text(0.75 + 3*g, -0.3, 'Color', ha='center',weight='bold')
    ax.text(2.2 + 3*g, -0.3, 'Key', ha='center', weight='bold')

ax.set_xlim(0, 3 * n_groups)
ax.set_ylim(n_rows, -1)
ax.axis('off')


# plt.show()
plt.savefig('../images/colors.png', format='png', dpi=300)

plt.close()