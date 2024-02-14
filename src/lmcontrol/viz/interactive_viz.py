import argparse
import io
import base64
import pickle

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go

from PIL import Image

import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split

# Contains 100 images for each digit from MNIST
mnist_path = 'datasets/mini-mnist-1000.pickle'

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

def load_mini_mnist():
    with open(mnist_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_data(path):
    npz = np.load(path)
    labels = dict()
    for k in npz.keys():
        if '_labels' in k:
            label = k[:-7]
            labels[label] = {'labels': npz[label+'_labels'], 'classes': npz[label+'_classes']}
    return npz['images'], npz['embedding'], labels


def prob(string):
    ret = float(string)
    if ret > 1.0 or ret < 0.0:
        raise argparse.ArgumentTypeError()

def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='the NumPy file archive containing data for plotting')
    parser.add_argument('-l', '--label', help='the label to use for coloring points', default='time')
    parser.add_argument('-s', '--subsample', help='the fraction to subsample data points to', type=float, default=None)

    args = parser.parse_args(argv)

    images, emb, all_labels = load_data(args.npz)

    if args.subsample:
        idx, _ = train_test_split(np.arange(len(images)), train_size=args.subsample, stratify=all_labels[args.label]['labels'])
        images = images[idx]
        emb = emb[idx]
        for k in all_labels:
            all_labels[k]['labels'] = all_labels[k]['labels'][idx]

    classes = all_labels[args.label]['classes']
    labels = all_labels[args.label]['labels']


    display_text = list()
    for i in range(len(labels)):
        tmp = list()
        for k in all_labels:
            c = all_labels[k]['classes'][all_labels[k]['labels'][i]]
            tmp.append(f"{k}: {c}")

        display_text.append(" | ".join(tmp))

    # Color for each label
    color_map = sns.color_palette(n_colors=len(np.unique(labels)))
    colors = [color_map[label] for label in labels]


    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        autosize=True,
        height=700
    )

    fig = go.Figure(data=[go.Scatter3d(
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=colors,
        )
    )], layout=layout)

    fig.update_traces(
        hoverinfo="none",
        hovertemplate=None,
    )

    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Graph(id="graph-5", figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-5", direction='bottom'),
        ],
    )

    @callback(
        Output("graph-tooltip-5", "show"),
        Output("graph-tooltip-5", "bbox"),
        Output("graph-tooltip-5", "children"),
        Input("graph-5", "hoverData"),
    )
    def display_hover(hoverData):
        if hoverData is None:
            return False, no_update, no_update

        # demo only shows the first point, but other points may also be available
        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        im_matrix = images[num]
        im_url = np_image_to_base64(im_matrix)
        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "50px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(str(display_text[num]), style={'font-weight': 'bold'})
                #html.P("Label: " + str(classes[labels[num]]), style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    app.run(debug=True)

if __name__ == "__main__":
    main()

