from dash import Dash, html, dcc, callback, Output, Input, State, dash_table
from dash_lib.data_utils import query_samples, load_dataset
from dash.exceptions import PreventUpdate
from dash_lib.plotting import (
    create_cd_plot,
    create_xdr_plot,
    create_param_plot,
    create_pl_image_plot,
    create_pl_plot,
    create_pl_variance_plot,
    create_uv_vis_plot,
    create_uv_vis_variance_plot,
)
from natsort import natsort_keygen

import pandas as pd
import json
import plotly.graph_objects as go
from werkzeug.middleware.profiler import ProfilerMiddleware
import os

# Initialize the app
app = Dash(__name__)
server = app.server

# Ref: https://github.com/plotly/dash-core-components/issues/403
# Add this to dcc.Graph config to set image export options
plot_config = {
    "toImageButtonOptions": {
        "format": "jpeg",  # one of png, svg, jpeg, webp
        "scale": 3,  # Multiply title/legend/axis/canvas sizes by this factor
    }
}

# App layout
app.layout = [
    html.Div(children="Large Param Space Data Browser"),
    html.Hr(),
    dcc.Input(
        placeholder="Enter DB path...",
        type="text",
        value="mydb.duckdb",
        id="db-path-input",
        persistence=True,
        persistence_type="session",
    ),
    html.Button("Load samples from DB", id="load-db-btn", n_clicks=0),
    html.Button("Clear Cache", id="clear-cache-btn", n_clicks=0),
    html.Button("Clear Selection", id="clear-selection-btn", n_clicks=0),
    html.Div(
        [
            html.Div(
                dash_table.DataTable(
                    id="samples-table",
                    columns=[],
                    data=[],
                    sort_action="native",
                    row_selectable="multi",
                    hidden_columns=["unique_id"],
                    style_table={
                        "height": "54vh",
                        "overflowY": "auto",
                        "overflowX": "auto",
                    },
                    style_cell={"whiteSpace": "pre-line"},
                    style_header={
                        "textAlign": "center"  # Align header text to the center
                    },
                ),
                style={"width": "49%"},
            ),
            dcc.Graph(id="cd-graph", style={"flex-basis": "49%"}, config=plot_config),
        ],
        style={"display": "flex", "height": "60vh", "width": "100vw"},
    ),
    html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            "Y-axis scale: ",
                                            dcc.RadioItems(
                                                ["linear", "log"],
                                                "linear",
                                                id="xrd-int-scale",
                                                inline=True,
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            "Y-axis mode: ",
                                            dcc.RadioItems(
                                                ["superimpose", "stack"],
                                                "superimpose",
                                                id="xrd-int-y-mode",
                                                inline=True,
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            "Transform: ",
                                            dcc.Checklist(
                                                options=[
                                                    {
                                                        "label": "Self-normalize",
                                                        "value": "normalize",
                                                    },
                                                ],
                                                value=[],
                                                id="xrd-int-transform",
                                                inline=True,
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "align-items": "center",
                                        },
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-evenly",
                                },
                            ),
                            dcc.Graph(id="xrd-int-plot", config=plot_config),
                        ],
                        style={"height": "50%"},
                    ),
                    dcc.Graph(id="pl-img", style={"height": "50%"}, config=plot_config),
                ],
                style={"flex-basis": "50%"},
            ),
            dcc.Graph(
                id="params-graph", style={"flex-basis": "45%"}, config=plot_config
            ),
        ],
        style={"display": "flex", "height": "60vh"},
    ),
    html.Div(
        [
            html.Div(
                dcc.Graph(id="pl-plot", config=plot_config), style={"flex-basis": "50%"}
            ),
            html.Div(
                dcc.Graph(id="pl-variance-plot", config=plot_config),
                style={"flex-basis": "50%"},
            ),
        ],
        style={
            "display": "flex",
        },
    ),
    html.Div(
        [
            html.Div(
                dcc.Graph(id="uv-vis-plot", config=plot_config),
                style={"flex-basis": "50%"},
            ),
            html.Div(
                dcc.Graph(id="uv-vis-variance-plot", config=plot_config),
                style={"flex-basis": "50%"},
            ),
        ],
        style={
            "display": "flex",
        },
    ),
    html.Div(
        html.Pre(id="json_info", children="Click a cell in the table"),
    ),
    dcc.Store(id="samples-data", storage_type="memory", data=[]),
]


@callback(
    Output("samples-data", "data"),
    Output("samples-table", "data"),
    Output("samples-table", "columns"),
    Input("load-db-btn", "n_clicks"),
    State("db-path-input", "value"),
    prevent_initial_call=True,
)
def load_samples(n_clicks, db_path):
    df = query_samples(db_path)
    df = df.reset_index()  # Turn the 'unique_id' index back into a column
    df = df.sort_values(by="sample_name", key=natsort_keygen())

    columns = [
        "unique_id",
        {"name": "param_hash", "id": "config_hash"},
        "sample_name",
        {"name": "annealing\ntemp (°C)", "id": "annealing_temp"},
        {
            "name": "annealing\ntime (s)",
            "id": "annealing_time",
        },
        {"name": "conc. (M)", "id": "concentration"},
        {"name": "dispense\nspeed (uL/s)", "id": "dispense_speed"},
        {"name": "dispense\nvolume (uL)", "id": "dispense_volume"},
        "spin_speed",
        "a_site",
        "wash",
        {
            "name": "homogeneity",
            "id": "homogeneity",
            "type": "numeric",
        },
        {
            "name": "g_factor_max",
            "id": "g_factor_max",
            "type": "numeric",
            "format": {"specifier": ".2e"},
        },
        {
            "name": "asymmetry",
            "id": "asymmetry",
            "type": "numeric",
            "format": {"specifier": ".2e"},
        },
        {
            "name": "ldlb_gmax",
            "id": "ldlb_area_ratio_gmax",
            "type": "numeric",
            "format": {"specifier": ".0%"},
        },
    ]

    table_columns = []
    for col in columns:
        if isinstance(col, dict):
            table_columns.append(col)
        else:
            table_columns.append({"name": col, "id": col})

        # Hide path columns and truncate hashes
    columns_to_hide = [
        col for col in df.columns if col not in map(lambda x: x["id"], table_columns)
    ]
    df = df.drop(columns=columns_to_hide, errors="ignore")

    for col in df.columns:
        if "hash" in col:
            df[col] = df[col].astype(str).str[:8]

    # Format homogeneity to 3 decimal places if present
    if "homogeneity" in df.columns:
        df["homogeneity"] = df["homogeneity"].round(3)
    if "g_factor_max" in df.columns:
        df["g_factor_max"] = df["g_factor_max"].round(6)

    # Load datasets for all samples to speed up subsequent plotting (optional, can be deferred until plot callbacks)
    unique_ids = df["unique_id"].tolist()
    for key in ["cd", "xrd", "cbox_pl", "cbox_uv_vis"]:
        load_dataset(unique_ids, key)

    return df.to_dict("records"), df.to_dict("records"), table_columns


@callback(
    Input("clear-cache-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_cache_callback(n_clicks):
    # Placeholder for clearing any cache if needed
    pass


@callback(
    Output("samples-table", "selected_rows"),
    Input("clear-selection-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selected_rows(n_clicks):
    # Reset any selections in the samples table
    return []


@callback(
    Output("params-graph", "figure"),
    Input("samples-data", "data"),
)
def update_param_plot(data):
    if not data:
        raise PreventUpdate

    df = pd.DataFrame(data)
    return create_param_plot(df)


@callback(
    Output("json_info", "children"),
    Input("samples-table", "selected_rows"),
    State("samples-table", "data"),
)
def update_json(rows, data):
    json_str = "Select a row in the table"
    if rows and data:
        selected_data = [data[i] for i in rows]
        json_str = json.dumps(selected_data, indent=4)
    return json_str


@callback(
    Output("cd-graph", "figure"),
    Input("samples-table", "selected_rows"),
    State("samples-table", "data"),
)
def update_cd_plot(rows, data):
    if not rows or not data:
        raise PreventUpdate

    selected_data = [data[i] for i in rows]
    unique_ids = [sample["unique_id"] for sample in selected_data]
    ds_cd = load_dataset(unique_ids, "cd", no_load=True)

    if ds_cd is not None:
        return create_cd_plot(ds_cd)
    else:
        return go.Figure()


@callback(
    Output("xrd-int-plot", "figure"),
    Input("samples-table", "selected_rows"),
    Input("xrd-int-scale", "value"),
    Input("xrd-int-y-mode", "value"),
    Input("xrd-int-transform", "value"),
    State("samples-table", "data"),
)
def update_xrd_int_plot(rows, y_scale, y_mode, y_transform, data):
    if not rows or not data:
        raise PreventUpdate

    selected_data = [data[i] for i in rows]
    unique_ids = [sample["unique_id"] for sample in selected_data]
    ds_xrd = load_dataset(unique_ids, "xrd", no_load=True)

    if ds_xrd is None:
        return go.Figure()

    # The plotting function now handles all data transformations
    plot = create_xdr_plot(
        ds_xrd,
        y_mode=y_mode,
        y_transform=y_transform,
        y_scale=y_scale,
    )

    is_normalized = "normalize" in y_transform

    if y_scale == "log":
        plot.update_yaxes(
            type="log", autorangeoptions={"minallowed": -2.1 if is_normalized else 1.99}
        )
    else:
        plot.update_yaxes(type="linear", autorangeoptions={"minallowed": 0.0})

    return plot


@callback(
    Output("pl-img", "figure"),
    Input("samples-table", "selected_rows"),
    State("samples-table", "data"),
)
def update_pl_img(rows, data):
    if not rows or not data:
        raise PreventUpdate

    selected_data = [data[i] for i in rows]
    unique_ids = [sample["unique_id"] for sample in selected_data]
    ds_pl = load_dataset(unique_ids, "cbox_photo", no_load=True)

    if ds_pl is not None:
        # The plotting function is now responsible for extracting images and metrics
        return create_pl_image_plot(ds_pl)
    else:
        return go.Figure()


@callback(
    Output("pl-plot", "figure"),
    Output("pl-variance-plot", "figure"),
    Input("samples-table", "selected_rows"),
    State("samples-table", "data"),
)
# def update_pl_plot(rows, y_scale, y_mode, y_transform, data):
def update_pl_plot(rows, data):
    if not rows or not data:
        raise PreventUpdate

    selected_data = [data[i] for i in rows]
    unique_ids = [sample["unique_id"] for sample in selected_data]
    ds_pl = load_dataset(unique_ids, "cbox_pl", no_load=True)

    if ds_pl is None:
        return go.Figure()

    # The plotting function now handles all data transformations
    plot = create_pl_plot(ds_pl)
    variance_plot = create_pl_variance_plot(ds_pl, wavelength_range=(480, 670))

    return plot, variance_plot


@callback(
    Output("uv-vis-plot", "figure"),
    Output("uv-vis-variance-plot", "figure"),
    Input("samples-table", "selected_rows"),
    State("samples-table", "data"),
)
def update_uv_vis_plot(rows, data):
    if not rows or not data:
        raise PreventUpdate

    selected_data = [data[i] for i in rows]
    unique_ids = [sample["unique_id"] for sample in selected_data]
    ds_uv_vis = load_dataset(unique_ids, "cbox_uv_vis", no_load=True)

    if ds_uv_vis is None:
        return go.Figure()

    # The plotting function now handles all data transformations
    plot = create_uv_vis_plot(ds_uv_vis)
    variance_plot = create_uv_vis_variance_plot(ds_uv_vis)

    return plot, variance_plot


# Run the app
if __name__ == "__main__":
    if os.getenv("PROFILER", None):
        app.server.config["PROFILE"] = True
        app.server.wsgi_app = ProfilerMiddleware(
            app.server.wsgi_app,
            restrictions=[50],  # Restrict output to top 50 most expensive calls
            profile_dir="profiler_stats",  # Directory to save stats files
        )
    app.run(debug=True, threaded=False)
