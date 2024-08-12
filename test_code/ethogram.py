import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"
plt.rcParams["font.size"] = "10"

k_preds = np.load(
    "/mnt/home/jwu10/working/ceph/results/vae/mals_64_mf/mals_p1_50/vis_latents/z_550_gmm.npy"
)
color = [
    "#1c86ee",
    "#E31A1C",  # red
    "#3A5311",
    "#6A3D9A",  # purple
    "#FF7F00",  # orange
    "black",
    "#ffbf00",
    "#7ec0ee",
    "#FB9A99",  # lt pink
    "#90EE90",
    "#CAB2D6",  # lt purple
    "#FDBF6F",  # lt orange
    "#b3b3b3",
    "#eee685",
    "maroon",
    "#ff83fa",
    "#ff1493",
    "#0000ff",
    "#36648b",
    "darkturquoise",
    "#00ff00",
    "#8b8b00",
    "#cdcd00",
    "#8b4500",
    "brown",
]

start = 1000
time = np.arange(start, start + 900)
k_preds = k_preds[time]
f, ax = plt.subplots(figsize=(10, 5))
events = []
for i in np.arange(25):
    curr_ind = k_preds == i
    events += [(time[curr_ind] - start) / 90]

    # plt.scatter((time[curr_ind]-start)/90, np.zeros(curr_ind.sum()) + i, linewidths=2)
    # plt.eventplot((time[curr_ind]-start)/90, orientation = "horizontal", lineoffsets=i, colors=color)

ax.eventplot(
    events,
    orientation="horizontal",
    lineoffsets=np.arange(25),
    colors=color,
    linelengths=1,
)
ax.get_yaxis().set_ticks([])
ax.set_title("GMM Cluster Assignment Over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cluster")
ax.set_xlim(left=0, right=10)
ax.spines["left"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
f.tight_layout()
f.savefig("./results/ethogram.png")
