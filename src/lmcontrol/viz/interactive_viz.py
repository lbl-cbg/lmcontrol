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
from sklearn.preprocessing import LabelEncoder

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
    images = npz['images']
    emb = npz['embedding']
    predictions = npz['predictions']
    true_labels = npz['true_labels']
    pred_labels = npz['Prediction_labels']
    Prediction_classes = npz['Prediction_classes']
    correct_incorrect = npz.get('Prediction_human_readable_labels')
    time_classes = npz['time_classes']
    time_labels = npz['time_labels']
    return images, emb, predictions, true_labels, pred_labels, correct_incorrect,Prediction_classes, time_classes, time_labels

def prob(string):
    ret = float(string)
    if ret > 1.0 or ret < 0.0:
        raise argparse.ArgumentTypeError()

current_selected_label = None

def build_app(npz, subsample=None, stratify_label=None, **addl_labels):
    """Build a Dash app for interactive viewing of data"""
    images, emb, predictions, true_labels, pred_labels, correct_incorrect, predictions_classes, time_classes, time_labels = load_data(npz)
    # I havent edited the above command as the paramaters passed maybe used later 
    all_labels = {}
    for k in addl_labels:
        all_labels[k] = dict()
        enc = LabelEncoder()
        all_labels[k]['labels'] = enc.fit_transform(addl_labels[k])
        all_labels[k]['classes'] = list(map(str, enc.classes_))

    idx = np.arange(len(images))
    if subsample is not None:
        if not isinstance(subsample, float) or not (subsample > 0.0 and subsample < 1.0):
            raise ValueError("subsample must be a float between (0.0, 1.0)")
        stratify = all_labels[stratify_label]['labels'] if stratify_label is not None else None
        idx, _ = train_test_split(np.arange(len(images)), train_size=subsample, stratify=stratify)
        images = images[idx]
        emb = emb[idx]
        predictions = predictions[idx]
        true_labels = true_labels[idx]
        pred_labels = pred_labels[idx]
        if correct_incorrect is not None:
            correct_incorrect = correct_incorrect[idx]
        for k in all_labels:
            all_labels[k]['labels'] = all_labels[k]['labels'][idx]

    encoded_images = [np_image_to_base64(img) for img in images]

    display_text = [
        f"idx: {i} | true_label: {true_labels[i]} | pred_label: {pred_labels[i]} | " +
        " | ".join([f"{k}: {all_labels[k]['classes'][all_labels[k]['labels'][i]]}" for k in all_labels])
        for i in range(len(emb))
    ]

    scatter = go.Scatter
    df_data = dict(x=emb[:, 0], y=emb[:, 1])
    if emb.shape[1] == 3:
        df_data['z'] = emb[:, 2]
        scatter = go.Scatter3d

    fig_vars = list(df_data)
    df_data['images'] = encoded_images
    df_data['text'] = display_text
    df_data['true_labels'] = true_labels
    df_data['pred_labels'] = pred_labels
    if correct_incorrect is not None:
        df_data['correct_incorrect'] = correct_incorrect
    for k in all_labels:
        df_data[k] = all_labels[k]['labels']
    df = pd.DataFrame(df_data)

    classes = {k: df[k].unique() for k in all_labels}

    dd_options = [{'label': k, 'value': k} for k in all_labels]

    app = Dash("LMControl Viz")
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
        """Update data displayed when hovering over points"""
        if hoverData is None:
            return False, no_update, no_update

        hover_data = hoverData["points"][0]
        bbox = hover_data["bbox"]
        num = hover_data["pointNumber"]

        pt_series = df.iloc[num]
        im_url = pt_series['images']
        disp_txt = pt_series['text']
        true_label = pt_series['true_labels']
        pred_label = pt_series['pred_labels']

        components = [
            html.Img(
                src=im_url,
                style={"width": "200px", 'display': 'block', 'margin': '0 auto'},
            ),
            html.P(f"True Label: {true_label}", style={'font-weight': 'bold'}),
            html.P(f"Prediction Label: {pred_label}", style={'font-weight': 'bold'}),
        ]

        for c in str(disp_txt).split("\n"):
            components.append(html.P(c, style={'font-weight': 'bold'}))

        children = [
            html.Div(components)
        ]

        return True, bbox, children

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('label-dropdown', 'value')]
    )
    def update_scatter_plot(selected_label):
        """Create Figure with scatter plot"""
        global current_selected_label
        current_selected_label = selected_label

        fig = go.Figure()
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

        legend = dict(
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
    parser.add_argument('-P', '--port', help='the port to run the application on', type=int, default=8050)
    parser.add_argument('-p', '--prod', help='do not run Dash app in debug mode', action='store_true', default=False)

    args = parser.parse_args(argv)

    app = build_app(args.npz, subsample=args.subsample, stratify_label=args.label)

    app.run(debug=not args.prod, port=args.port)

if __name__ == "__main__":
    main()