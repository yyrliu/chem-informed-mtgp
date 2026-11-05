const set_y_scale = function(figure, scale) {
    if(figure === undefined) {
        return {'data': [], 'layout': {}};
    }
    const fig = Object.assign({}, figure, {
        'layout': {
            ...figure.layout,
            'yaxis': {
                ...figure.layout.yaxis, type: scale,
                'autorangeoptions': {
                    'minallowed': (scale === 'log' ? 1.99 : 0.0)
                }
            }
        }
    });
    return fig;
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
    clientside: {
        set_y_scale
    }
});

// python code
// clientside_callback(
//     ClientsideFunction(
//         namespace='clientside',
//         function_name='set_y_scale'
//     ),
//     Output("xrd-int-plot", "figure"),
//     Input("clientside-xrd-int-plot-store", "data"),
//     Input("xrd-int-scale", "value"),
// )
