import argparse
import io
import base64
import json

import pickle
import os

from dash import Dash, dcc, html, Input, Output, no_update, callback, State
import plotly.graph_objects as go

from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.colors as pc
import plotly.graph_objects as go
import plotly.express as px

from matplotlib.colors import Normalize, LinearSegmentedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from hdmf_ai import ResultsTable
from hdmf.common import get_hdf5io, EnumData

metadata_info = None

# Helper functions
def np_image_to_base64(im_matrix):
    im = Image.fromarray(im_matrix)
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image
    return im_url

current_selected_label = None

input_path = df = classes = dd_options = all_labels = scatter = fig_vars = None

def to_int(arr):
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)
    return arr

def load_data(path, subsample=None, stratify_label=None, **addl_labels):
    global df
    global classes
    global dd_options
    global all_labels
    global scatter
    global fig_vars

    input_io = get_hdf5io(input_path, 'r')
    dt = input_io.read()

    emb_io = get_hdf5io(path, 'r')
    rt = emb_io.read()

    all_labels = dict()

    for lbl in metadata_info:
        all_labels[lbl] = dict()
        if metadata_info[lbl]['enum']:
            all_labels[lbl]['classes'] = dt[lbl].elements[:]
        all_labels[lbl]['labels'] = dt[lbl].data

    images, emb = dt['images'].data, rt['viz_embedding'].data

    idx = np.arange(len(images))
    # Subsample data
    if subsample is not None and (subsample < 1.0 and subsample > 0.0):
        if not isinstance(subsample, float) or not (subsample > 0.0 and subsample < 1.0):
            raise ValueError("subsample must be a float between (0.0, 1.0)")
        stratify = all_labels[stratify_label]['labels'] if stratify_label is not None else None
        idx, _ = train_test_split(np.arange(len(images)), train_size=subsample, stratify=stratify)
        idx = np.sort(idx)
        images = images[idx]
        emb = emb[idx]
        for k in all_labels:
            all_labels[k]['labels'] = to_int(all_labels[k]['labels'][idx])

    # Read in all data
    else:
        images = images[:]
        emb = emb[:]
        for k in all_labels:
            all_labels[k]['labels'] = to_int(all_labels[k]['labels'][:])

    input_io.close()
    emb_io.close()

    encoded_images = [np_image_to_base64(img) for img in images]

    # Compute display label
    display_text = list()
    for i in range(len(emb)):
        tmp = list()
        for k in all_labels:
            if metadata_info[k]['enum']:
                c = all_labels[k]['classes'][all_labels[k]['labels'][i]]
            else:
                c = all_labels[k]['labels'][i]
            tmp.append(f"{k}: {c}")
        display_text.append(f"idx: {idx[i]}\n" + " | ".join(tmp))


    df_data = dict(x=emb[:, 0], y=emb[:, 1])
    if emb.shape[1]== 3:
        df_data['z'] = emb[:, 2]

    fig_vars = list(df_data)  # use this so we know what kwargs to pass into our scatter graph object
    df_data['images'] = encoded_images
    df_data['text'] = display_text

    df_data1 = []

    for k in all_labels:
        df_data[k] = all_labels[k]['labels']
        if k == 'time':
            df_data1 = all_labels[k]['labels'].tolist()

    df = pd.DataFrame(df_data)
    classes = {k: np.arange(all_labels[k]['classes'].shape[0]) for k in all_labels if 'classes' in all_labels[k]}

    dd_options = [{'label': k, 'value': k} for k in all_labels]

    if 'z' in df:
        scatter = go.Scatter3d
        fig_vars = ['x', 'y', 'z']
    else:
        scatter = go.Scatter
        fig_vars = ['x', 'y']


def list_hdmfai_files(directory):
    return [{'label': f, 'value': f} for f in os.listdir(directory) if f.endswith('.h5') and os.path.join(directory, f) != input_path]


def build_app(directory, subsample=1.0, stratify_label=None, **addl_labels):
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

    hdmfai_files = list_hdmfai_files(directory)

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

        if metadata_info[current_selected_label]['enum']:
            class_id = hover_data['curveNumber']
            class_val = classes[current_selected_label][class_id]
            mask = df[current_selected_label] == class_val
            pt_series = df[['images', 'text']][mask].iloc[num]
        else:
            pt_series = df[['images', 'text']].iloc[num]
        im_url = pt_series['images']
        disp_txt = pt_series['text']

        components = [
                html.Img(
                    src=im_url,
                    style={"width": "200px", 'display': 'block', 'margin': '0 auto'},
                ),
            ]

        for c in str(disp_txt).split("\n"):
            components.append(html.P(c, style={'font-weight': 'bold'}))

        children = [
            html.Div(components)
        ]

        return True, bbox, children

    # Set up our Dash application
    app = Dash("LMControl Viz")
    app.layout = html.Div(
        className="container",
        children=[
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Refresh every 60 seconds
                n_intervals=0
            ),
            dcc.Dropdown(
                id='hdmfai-dropdown',
                options=hdmfai_files,
                placeholder='Select a dataset',
            ),
            html.Div([
                html.Label('Subsample fraction'),
                dcc.Input(
                    id='subsample-input',
                    type='number',
                    min=0.0,
                    max=1.0,
                    value=subsample,
                ),
                html.Button('Load data', id='update-button', n_clicks=0),
            ]),
            dcc.Loading(
                id='loading-label-dropdown',
                type='default',
                children=[
                    dcc.Dropdown(
                        id='label-dropdown',
                        placeholder='Select a label to color points with',
                        value='ht',
                    ),
                ],
            ),
            dcc.Graph(id="scatter-plot", clear_on_unhover=True),
            dcc.Tooltip(id="scatter-tooltip", direction='bottom'),
        ],
    )
    @app.callback(
        Output('hdmfai-dropdown', 'options'),
        Input('interval-component', 'n_intervals')
    )
    def refresh_hdmfai_list(n_intervals):
        # List NPZ files in the directory
        return list_hdmfai_files(directory)

    @app.callback(
        Output('label-dropdown', 'options'),
        Output('label-dropdown', 'value'),
        [Input('update-button', 'n_clicks')],
        [State('hdmfai-dropdown', 'value'), State('subsample-input', 'value')],
    )
    def update_viz_data(n_clicks, selected_file, subsample):
        if selected_file is None:
            return [], None
        selected_file = os.path.join(directory, selected_file)
        load_data(selected_file, subsample=subsample, stratify_label=stratify_label)
        return dd_options, 'ht'

    def get_color_for_label(label_idx, palette=plt.cm.tab20.colors):
        """Assign consistent colors to labels based on their index."""
        # Use modulo operation to ensure color index is within palette range
        color_index = label_idx % len(palette)
        return palette[color_index]

    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('label-dropdown', 'value')],
        [State('scatter-plot', 'relayoutData')]
    )

    def update_scatter_plot(selected_label, relayout_data):
        """Create Figure with scatter plot"""


        if selected_label is None:
            return go.Figure()

        if selected_label is None:
            return go.Figure()

        global current_selected_label
        current_selected_label = selected_label

        fig = go.Figure()
        if selected_label not in classes:

            global_min = df[selected_label].min()
            global_max = df[selected_label].max()
            norm = Normalize(vmin=global_min, vmax=global_max)
            fig_kwargs = {var: df[var] for var in fig_vars}

            custom_cmap = LinearSegmentedColormap.from_list(
                'CustomRdBu',
                [(0, 'black'), (1, 'red')]
            )

            n_colors = 256
            cmap_values = np.linspace(0, 1, n_colors)
            rgb_colors = [custom_cmap(value)[:3] for value in cmap_values]
            plotly_colorscale = [
                [val, f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"]
                for val, (r, g, b) in zip(cmap_values, rgb_colors)
            ]

            fig.add_trace(go.Scatter3d(
                mode='markers',
                name=selected_label,
                marker=dict(
                    size=2,
                    color=df[selected_label].values,
                    colorscale=plotly_colorscale,
                    colorbar=dict(
                        title=selected_label,
                        tickvals=[global_min, global_max],
                        ticktext=[global_min, global_max]
                    ),
                    cmin=global_min,
                    cmax=global_max,
                ),
                **fig_kwargs
            ))

        else:
            for cls in classes[selected_label]:
                mask = df[selected_label] == cls
                fig_kwargs = {var: df[var][mask] for var in fig_vars}

                global label_color_map

                label_color = get_color_for_label(cls)

                fig.add_trace(scatter(
                    name=str(all_labels[selected_label]['classes'][cls]),
                    mode='markers',
                    marker=dict(
                        size=2,
                        color=f'rgba({label_color[0]*255}, {label_color[1]*255}, {label_color[2]*255}, 1)'
                    ),
                    **fig_kwargs
                ))

        legend=dict(
            x=0,
            y=0,
            bordercolor="Black",
            borderwidth=2,
            itemsizing='constant',
            xanchor='left',
            yanchor='bottom'
        )
        camera = None
        if relayout_data:
            camera = relayout_data.get('scene.camera')
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0),
                              showlegend=True, legend=legend,
                              autosize=True, height=700, scene_camera=camera)
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
        )
        return fig

    return app


def main(argv=None):
    global input_path
    global metadata_info

    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', help='the HDMF input table that contains images and metadata')
    parser.add_argument('emb_dir', help='the directory containing HDMF-AI tables with embeddings')
    parser.add_argument('-s', '--subsample', help='the fraction to subsample data points to', type=float, default=None)
    parser.add_argument('-l', '--label', help='the label to use for stratifying subsample', default='time')
    parser.add_argument('-P', '--port', help='the port to run the application on', type=int, default=8050)
    parser.add_argument('-p', '--prod', help='do not run Dash app in debug mode', action='store_true', default=False)

    args = parser.parse_args(argv)

    input_path = args.inputs

    input_io = get_hdf5io(input_path, 'r')
    dt = input_io.read()

    exclude = {'masks', 'images', 'paths', 'raw_images'}

    metadata_info = {c: {'description': dt[c].description, 'enum': isinstance(dt[c], EnumData)} for c in dt.colnames if c not in exclude}

    app = build_app(args.emb_dir, subsample=args.subsample)

    app.run(debug=not args.prod, host='0.0.0.0', port=args.port)


if __name__ == "__main__":
    main()
