import umap
import plotly.express as px


def get_reduced(embeddings, n_dims=3):
    return umap.UMAP(n_components=n_dims).fit_transform(embeddings)


def make_plot_3d(data, colors=None, hover_labels=None):
    fig = px.scatter_3d(data, x=0, y=1, z=2, color=colors, hover_name=hover_labels)
    fig.update_traces(marker_size=8)
    return fig
