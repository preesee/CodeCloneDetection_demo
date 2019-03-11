import math
import numpy as np
import matplotlib.pyplot as plt


def example_plot(ax, fontsize=12):
    ax.plot([1, 2])
    ax.locator_params(nbins=3)
    ax.set_xlabel('x-label', fontsize=fontsize)
    ax.set_ylabel('y-label', fontsize=fontsize)
    ax.set_title('Title', fontsize=fontsize)

import matplotlib.gridspec as gridspec
fig = plt.figure()
gs1 = gridspec.GridSpec(1, 2)
ax1 = fig.add_subplot(gs1[0])
ax2 = fig.add_subplot(gs1[1])

example_plot(ax1)
example_plot(ax2)

gs1.tight_layout(fig, pad=2,  h_pad=5, w_pad=-4.1, rect=[0, 0, 1, 1])


# x = np.linspace(0, 2 * np.pi, 400)
# y = np.sin(x ** 2)
# # Two subplots, unpack the axes array immediately
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# ax1.plot(x, y)
# ax1.set_title('Sharing Y axis')
# ax2.scatter(x, y)
# plt.tight_layout()

# x = np.arange(-10., 10., 0.2)
# sig = sigmoid(x)
#
# fig = plt.figure()
# ax = fig.add_subplot(2, 1, 1)
#
# # Move left y-axis and bottim x-axis to centre, passing through (0,0)
# ax.spines['left'].set_position('center')
# ax.spines['bottom'].set_position('center')
#
# # Eliminate upper and right axes
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
#
# # Show ticks in the left and lower axes only
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')

# plt.plot(x,sig)
plt.show()
