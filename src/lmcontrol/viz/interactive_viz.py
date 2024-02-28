import argparse
import io
import base64
import pickle

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go

from PIL import Image

import numpy as np
import pandas as pd
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

current_selected_label = None

def build_app(npz, subsample=None, stratify_label=None):
    """Build a Dash app for interactive viewing of data

    Args:
        npz (str)               : A path to the NPZ file containing data needed for
                                  building interactive scatter plot
        subsample (float)       : the fraction of data to subsample for viewing. This
                                  should be a floating point number between (0.0, 1.0).
                                  By default, no data is subsampled.
        stratify_label (str)    : the label to use for stratifying subsamples. This should
                                  be one of the labels in NPZ files.

    Returns:
        app (dash.Dash)         : a Dash application
    """

    images, emb, all_labels = load_data(npz)

    # Subsample data
    if subsample is not None:
        if not isinstance(subsample, float) or not (subsample > 0.0 and subsample < 1.0):
            raise ValueError("subsample must be a float between (0.0, 1.0)")
        stratify = all_labels[stratify_label]['labels'] if stratify_label is not None else None
        idx, _ = train_test_split(np.arange(len(images)), train_size=subsample, stratify=stratify)
        images = images[idx]
        emb = emb[idx]
        for k in all_labels:
            all_labels[k]['labels'] = all_labels[k]['labels'][idx]

    encoded_images = [np_image_to_base64(img) for img in images]

    # Compute display label
    display_text = list()
    for i in range(len(emb)):
        tmp = list()
        for k in all_labels:
            c = all_labels[k]['classes'][all_labels[k]['labels'][i]]
            tmp.append(f"{k}: {c}")
        display_text.append(" | ".join(tmp))

    # Set up data for graph object
    scatter = go.Scatter
    df_data = dict(x=emb[:, 0], y=emb[:, 1])
    if emb.shape[1]== 3:
        df_data['z'] = emb[:, 2]
        scatter = go.Scatter3d

    fig_vars = list(df_data)  # use this so we know what kwargs to pass into our scatter graph object
    df_data['images'] = encoded_images
    df_data['text'] = display_text
    for k in all_labels:
        df_data[k] = all_labels[k]['labels']
    df = pd.DataFrame(df_data)

    # A map from class name to unique labels for maintaining constency
    # between Traces added in update_scatter_plot and curveNum in hoverData
    # passed into display_hover. This will get used to get
    # label value for each data point
    classes = {k: df[k].unique() for k in all_labels}

    dd_options = [{'label': k, 'value': k} for k in all_labels]

    # Set up our Dash application
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

    # Make a function of updating the hover
    @callback(
        Output("scatter-tooltip", "show"),
        Output("scatter-tooltip", "bbox"),
        Output("scatter-tooltip", "children"),
        Input("scatter-plot", "hoverData"),
    )
    def display_hover(hoverData):
        """Update data displayed when hovering over points"""
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        # curveNumber correspoends to the order in which the Trace was
        # added to the Figure. For example, curveNumber=4 means we need to get
        # data for the fifth Trace that was added to the Figure.
        class_id = hover_data['curveNumber']
        class_val = classes[current_selected_label][class_id]
        mask = df[current_selected_label] == class_val
        pt_series = df[['images', 'text']][mask].iloc[num]
        im_url = pt_series['images']
        disp_txt = pt_series['text']

        children = [
            html.Div([
                html.Img(
                    src=im_url,
                    style={"width": "200px", 'display': 'block', 'margin': '0 auto'},
                ),
                html.P(str(disp_txt), style={'font-weight': 'bold'})
            ])
        ]

        return True, bbox, children

    # Make a function for updating the display when we toggle label
    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('label-dropdown', 'value')]
    )
    def update_scatter_plot(selected_label):
        """Create Figure with scatter plot"""

        # we need to maintain a global state so display_hover
        # knows what label we are currently using
        global current_selected_label
        current_selected_label = selected_label

        fig = go.Figure()
        # add a Trace object for each class label. The order in which
        # we iterate over these values corresponds to curveNum included
        # in hoverData passed into display_hover
        for cls in classes[selected_label]:
            mask = df[selected_label] == cls
            fig_kwargs = {var: df[var][mask] for var in fig_vars}
            fig.add_trace(scatter(
                name=all_labels[selected_label]['classes'][cls],
                mode='markers',
                marker=dict(
                    size=2,
                ),
                **fig_kwargs
            ))

        # Options for rendering the legend
        legend=dict(
            x=0,
            y=1,
            bordercolor="Black",
            borderwidth=2,
            itemsizing='constant',
        )

        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                          showlegend=True, legend=legend,
                          autosize=True, height=700)
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        return fig

    return app

def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument('npz', help='the NumPy file archive containing data for plotting')
    parser.add_argument('-s', '--subsample', help='the fraction to subsample data points to', type=float, default=None)
    parser.add_argument('-l', '--label', help='the label to use for stratifying subsample', default='time')

    args = parser.parse_args(argv)

    app = build_app(args.npz, subsample=args.subsample, stratify_label=args.label)

    app.run(debug=True)


if __name__ == "__main__":
    main()

