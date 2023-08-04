from typing import List, Dict, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px


def histogram_figure(cost_data: Dict[str, List[float]],
                     title: str,
                     color_list: List[str] = px.colors.qualitative.T10
                     ) -> go.FigureWidget:
    """ Create a cost density plot

    Args:
        cost_data (Dict[str, List[float]]): Mapping of agents to cost list
        title (str): Title of the plot
        color_list (List[str], optional): Color list. Defaults to px.colors.qualitative.T10.

    Returns:
        go.FigureWidget: Plot widget
    """
    cost_label_pair = list(cost_data.items())
    costs = [item[1] for item in cost_label_pair]
    labels = [item[0] for item in cost_label_pair]
    colors = [color_list[index % len(color_list)] for index in range(len(labels))]

    fig = ff.create_distplot(
        costs,
        group_labels=labels,
        colors=colors,
        bin_size=4,
        show_rug=False)
    for color, cost_list in zip(colors, costs):
        fig.add_vline(
            x=np.median(cost_list),
            line_width=3,
            line_dash="dash",
            line_color=color
        )

    common_axis_layout = dict(
        showline=True,
        linecolor="#a2a2a2",
        linewidth=1,
        showgrid=True,
        gridcolor="#a2a2a2",
    )
    fig.update_layout(
        template="plotly_white",
        width=700,
        height=400,
        title=dict(text=f"{title}", x=0.5),
        yaxis=dict(
            **common_axis_layout,
            title=dict(text="density"),
            # type="log"
        ),
        xaxis=dict(
            **common_axis_layout,
            title=dict(text="cost"),
            # type="log"
        ),
        bargap=0.1,
        font=dict(
            size=12,
            color="Black"
        )
    )
    return fig


def tube_figure(cost_data: Dict[str, Dict[int, List[float]]],
                title: str,
                color_list: List[str] = px.colors.qualitative.T10,
                percentiles: Tuple[int] = (20, 80),
                log_xaxis: bool = False,
                log_yaxis: bool = False,
                xaxis_name: str = "uncertainty radius",
                ) -> go.FigureWidget:
    """ Make error plot as in Figure 2.a and 2.b

    Arg:s
        cost_data (Dict[str, Dict[int, List[float]]]): Dictionary of costs per rho
        title (str): Title of the plot
        color_list (List[str], optional): Color list. Defaults to px.colors.qualitative.T10.
        percentiles (Tuple[int]): Lower and Upper percentiles.

    Returns:
        go.FigureWidget: Plot widget
    """
    fig = go.FigureWidget()
    cost_label_pair = list(cost_data.items())
    cost_data = [item[1] for item in cost_label_pair]
    labels = [item[0] for item in cost_label_pair]
    colors = [color_list[index % len(color_list)] for index in range(len(labels))]

    percentile_lower, percentile_up = percentiles
    for color, cost_dict, label in zip(colors, cost_data, labels):
        rho_values = {rho: np.percentile(cost_list, [percentile_lower, 50, percentile_up])
                      for rho, cost_list in cost_dict.items()}
        fig.add_trace(go.Scatter(
            x=list(rho_values.keys()),
            y=[item[1] for item in rho_values.values()],
            line=dict(color=color),
            mode="lines",
            name=label,
            legendgroup=label
        ))
        fig.add_trace(
            go.Scatter(
                name="Upper Bound",
                x=list(rho_values.keys()),
                y=[item[2] for item in rho_values.values()],
                mode="lines",
                marker=dict(color=color),
                line=dict(width=0),
                showlegend=False,
                legendgroup=label
            ))
        fig.add_trace(
            go.Scatter(
                name="Lower Bound",
                x=list(rho_values.keys()),
                y=[item[0] for item in rho_values.values()],
                marker=dict(color=color),
                line=dict(width=0),
                mode="lines",
                # fillcolor=color,
                opacity=0.5,
                fill="tonexty",
                showlegend=False,
                legendgroup=label
            ))

    common_axis_layout = dict(
        showline=True,
        linecolor="#a2a2a2",
        linewidth=1,
        showgrid=True,
        gridcolor="#a2a2a2",
    )
    fig.update_layout(
        template="plotly_white",
        width=700,
        height=400,
        title=dict(text=f"{title}", x=0.5),
        yaxis=dict(
            **common_axis_layout,
            title=dict(text="costs"),
            type="log" if log_yaxis else None
        ),
        xaxis=dict(
            **common_axis_layout,
            title=dict(text=xaxis_name),
            type="log" if log_xaxis else None
        ),
        font=dict(
            size=12,
            color="Black"
        )
    )
    return fig
