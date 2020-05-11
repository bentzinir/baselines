import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull


def format_ax(ax):
    ax.xaxis.pane.set_edgecolor('#000000')
    ax.yaxis.pane.set_edgecolor('#000000')
    ax.zaxis.pane.set_edgecolor('#000000')

    ax.w_xaxis.set_pane_color((.0, .0, .0, .05))
    ax.w_yaxis.set_pane_color((.0, .0, .0, .05))
    ax.w_zaxis.set_pane_color((.0, .0, .0, .05))
    # ax.w_xaxis.line.set_color("white")
    # ax.w_yaxis.line.set_color("white")
    # ax.w_zaxis.line.set_color("white")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])
    plt.ylim([0, 1])
    # # plt.axis('off')
    # plt.xticks([], [])
    # plt.yticks([], [])
    # ax.set_zticks([])
    #
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    frame1.axes.yaxis.set_ticklabels([])
    frame1.axes.zaxis.set_ticklabels([])


pts = np.random.rand(500, 3)

hull = ConvexHull(pts)

fig = plt.figure('Skeleton Buffer')
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points
ax.scatter(pts.T[0], pts.T[1], pts.T[2], color="grey", alpha=0.4, s=2*np.ones(1000), edgecolor=None)

for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], color='black', s=5)

# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "k-", linewidth=.2)

format_ax(ax)
plt.savefig('metric_replay_buffer.png', bbox_inches='tight', pad_inches=0, dpi=200)
plt.show(block=False)

######################################################################


pts = np.random.rand(500, 3)

hull = ConvexHull(pts)

fig = plt.figure('Metric Replay buffer')
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points
ax.scatter(pts.T[0], pts.T[1], pts.T[2], color="grey", alpha=0.4, s=2*np.ones(1000), edgecolor=None)

for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], color='black', s=5)

# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "k-", linewidth=.2)

format_ax(ax)
plt.savefig('metric_replay_buffer.png', bbox_inches='tight', pad_inches=0, dpi=200)
plt.show(block=False)

######################################################################

pts = 0.1 * np.random.rand(100, 3) + 0.5

hull = ConvexHull(pts)

fig = plt.figure('Replay Buffer Only')
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points
ax.scatter(pts.T[0], pts.T[1], pts.T[2], color="grey", alpha=0.4, s=2*np.ones(1000), edgecolor=None)

format_ax(ax)
plt.savefig('replay_buffer_only.png', bbox_inches='tight', pad_inches=0, dpi=200)
plt.show(block=False)
######################################################################

pts = 0.3 * np.random.rand(100, 3) + 0.5
# pts = np.clip(pts, 0.25, 0.75)

hull = ConvexHull(pts)

fig = plt.figure('Convex Hull Only')
ax = fig.add_subplot(111, projection="3d")

# Plot defining corner points
# ax.scatter(pts.T[0], pts.T[1], pts.T[2], color="grey", alpha=0.4, s=2*np.ones(1000), edgecolor=None)

for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.scatter(pts[s, 0], pts[s, 1], pts[s, 2], color='black', s=5)

# 12 = 2 * 6 faces are the simplices (2 simplices per square face)
for s in hull.simplices:
    s = np.append(s, s[0])  # Here we cycle back to the first coordinate
    ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "k-", linewidth=.1)

format_ax(ax)
plt.savefig('convex_hull_only.png', bbox_inches='tight', pad_inches=0, dpi=200)
plt.show()
