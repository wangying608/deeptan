import seaborn as sns
sns.set_theme(style="white")

df = sns.load_dataset("penguins")

g = sns.JointGrid(data=df, x="body_mass_g", y="bill_depth_mm", space=0)
g.plot_joint(sns.kdeplot,
             fill=True, clip=((2200, 6800), (10, 25)),
             thresh=0, levels=100, cmap="rocket")
g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)
g.set_axis_labels("Body mass (g)", "Bill depth (mm)")
g.figure.set_size_inches(6, 4.5)
g.savefig("penguins.svg", dpi=300, transparent=True, format="svg")
