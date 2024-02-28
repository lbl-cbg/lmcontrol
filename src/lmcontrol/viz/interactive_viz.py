import argparse
import io
import base64
import pickle

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import plotly.express as px

from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from skimage.transform import rescale

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

    # Subsample data
    if args.subsample:
        idx, _ = train_test_split(np.arange(len(images)), train_size=args.subsample, stratify=all_labels[args.label]['labels'])
        images = images[idx]
        emb = emb[idx]
        for k in all_labels:
            all_labels[k]['labels'] = all_labels[k]['labels'][idx]

    # Compute display label
    display_text = list()
    for i in range(len(emb)):
        tmp = list()
        for k in all_labels:
            c = all_labels[k]['classes'][all_labels[k]['labels'][i]]
            tmp.append(f"{k}: {c}")

        display_text.append(" | ".join(tmp))


    #scatter = go.Scatter
    #fig_kwargs = dict(x=emb[:, 0], y=emb[:, 1])
    #if emb.shape[1]== 3:
    #    fig_kwargs['z'] = emb[:, 2]
    #    scatter = go.Scatter3d

    #colors = dict()
    #dd_options = list()
    #for k in all_labels:
    #    dd_options.append({'label': k, 'value': k})
    #    classes = all_labels[k]['classes']
    #    labels = all_labels[k]['labels']
    #    color_map = sns.color_palette(n_colors=len(classes))
    #    colors[k] = [color_map[label] for label in labels]

    scatter = px.scatter
    data = dict(x=emb[:, 0], y=emb[:, 1])
    if emb.shape[1]== 3:
        data['z'] = emb[:, 2]
        scatter = px.scatter_3d
    fig_kwargs = {k:k for k in data}

    dd_options = list()
    for k in all_labels:
        dd_options.append({'label': k, 'value': k})
        data[k] = all_labels[k]['classes'][all_labels[k]['labels']]

    df = pd.DataFrame(data=data)

    app = Dash(__name__)

    app.layout = html.Div(
        className="container",
        children=[
            dcc.Dropdown(
                id='label-dropdown',
                options=dd_options,
                value='ht'
            ),
            dcc.Graph(id="scatter-plot", clear_on_unhover=True),
            dcc.Tooltip(id="scatter-tooltip", direction='bottom'),
        ],
    )

    @callback(
        Output("scatter-tooltip", "show"),
        Output("scatter-tooltip", "bbox"),
        Output("scatter-tooltip", "children"),
        Input("scatter-plot", "hoverData"),
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
                    style={"width": "200px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(str(display_text[num]), style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('label-dropdown', 'value')]
    )
    def update_scatter_plot(selected_label):
        fig = px.scatter_3d(df, color=selected_label, **fig_kwargs)


        #fig = go.Figure(data=[scatter(
        #    mode='markers',
        #    marker=dict(
        #        size=2,
        #        color=colors[selected_label],
        #    ),
        #    **fig_kwargs
        #)] )#, layout=layout)

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          showlegend=True,
                          autosize=True, height=700)
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        return fig


    app.run(debug=True)

if __name__ == "__main__":
    main()

