import dash
from dash import html, dcc
import plotly.graph_objs as go
import numpy as np
from dash.dependencies import Input, Output
from waitress import serve

# Advanced Morris-Thorne Metric Function with logarithmic scaling and full GUU tensor
def advanced_morris_thorne_metric_log(r, b0, phi0, spin, exotic_factor):
    try:
        b_r = np.log1p(abs(b0 * np.exp(-r**2 / b0**2) + exotic_factor * np.sin(r)))
        phi_r = np.log1p(abs(phi0 * np.exp(-r**2 / b0**2)))
        omega = np.log1p(abs(spin * r**2))  # Spin term with logarithmic scaling
        g_tt = -np.exp(2 * phi_r)
        g_rr = 1 / (1 - b_r/r)
        g_thth = np.log1p(abs(r**2))
        g_phiphi = g_thth * (np.sin(np.pi / 2)**2 + omega)
        return np.array([[g_tt, 0, 0, 0], [0, g_rr, 0, 0], [0, 0, g_thth, 0], [0, 0, 0, g_phiphi]])
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        return None

# Function to create and update the wormhole visualization in Plotly with GUU tensor
def create_update_wormhole_log(b0, phi0, spin, exotic_factor):
    r_range = np.linspace(-10, 10, 100)
    guu_tensors = [advanced_morris_thorne_metric_log(r, b0, phi0, spin, exotic_factor) for r in r_range]

    # Create subplots for each component of GUU
    fig = go.Figure()
    for i in range(4):
        for j in range(4):
            zs = [guu[i, j] for guu in guu_tensors if guu is not None]
            fig.add_trace(go.Scatter3d(x=r_range[:len(zs)], y=[i]*len(zs), z=zs, mode='lines', name=f'g{i}{j}'))

    # Update layout for readability
    fig.update_layout(scene=dict(
                        xaxis_title='R',
                        yaxis_title='GUU Component',
                        zaxis_title='Value'),
                      margin=dict(l=0, r=0, b=0, t=0))
    return fig

# Dash app layout
app = dash.Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='wormhole-plot'),
    html.Label('B0:'),
    dcc.Slider(
        id='b0-slider',
        min=1,
        max=10,
        value=1,
        step=0.1,
        marks={i: str(i) for i in range(1, 11)}
    ),
    html.Label('Phi0:'),
    dcc.Slider(
        id='phi0-slider',
        min=0,
        max=10,
        value=0,
        step=0.1,
        marks={i: str(i) for i in range(11)}
    ),
    html.Label('Spin:'),
    dcc.Slider(
        id='spin-slider',
        min=0,
        max=10,
        value=0,
        step=0.1,
        marks={i: str(i) for i in range(11)}
    ),
    html.Label('Exotic Factor:'),
    dcc.Slider(
        id='exotic-factor-slider',
        min=0,
        max=10,
        value=0,
        step=0.1,
        marks={i: str(i) for i in range(11)}
    )
])

# Callback for updating the graph
@app.callback(
    Output('wormhole-plot', 'figure'),
    [Input('b0-slider', 'value'),
     Input('phi0-slider', 'value'),
     Input('spin-slider', 'value'),
     Input('exotic-factor-slider', 'value')]
)
def update_graph(b0, phi0, spin, exotic_factor):
    return create_update_wormhole_log(b0, phi0, spin, exotic_factor)

# Waitress server setup
if __name__ == '__main__':
    serve(app.server, host='0.0.0.0', port=8080)