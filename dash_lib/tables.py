from dash import dash_table

available_columns = [
    "filename",
    "filepath",
    "measurement_name",
    "sample",
    "session",
    "user_name",
    "timestamp",
    "annealing_temp",
    "annealing_time",
    "concentration",
    "config_hash",
    "dispense_speed",
    "dispense_volume",
    "spin_speed",
]
visible_columns = ["session", "sample", "config_hash"]

h5_files_tbl = dash_table.DataTable(
    data=[],
    id="data-table",
    row_selectable="multi",
    sort_action="native",
    hidden_columns=list(set(available_columns) - set(visible_columns)),
    selected_rows=[],
    style_table={"height": "38vh", "overflowY": "scroll"},
)


def prepare_table_data(metadatas):
    first_entry = metadatas[0]
    columns = [{"name": k, "id": k} for k in first_entry.keys()]

    for entry in metadatas:
        entry["config_hash"] = entry["config_hash"][:8]  # Shorten hash for display

    return metadatas, columns
