import pandas
import plotly

from plotly import graph_objects, subplots


# --- Configuration ---
CONFIG_YEAR = {
    'color_col': 'Year',
    'x_col': 'Interpolation_w2',
    'y_col': 'Value',
    'metric_filter': 'NDCG10',
    'template': 'simple_white',
}

CONFIG_MONTH = {
    'color_col': 'Month',
    'x_col': 'Interpolation_w2',
    'y_col': 'Value',
    'metric_filter': 'NDCG10',
    'template': 'simple_white',
}

colors = plotly.colors.qualitative.Bold

# --- Helper Functions ---


def add_year_subplot_traces(fig, df, df_base, row_num, col_num, config, show_year_legend=False):
    df = df[df['Metric'] == config['metric_filter']]
    df_base = df_base[df_base['Metric'] == config['metric_filter']]
    if df.empty:
        print("Warning: No data for year subplot.")
        return

    years = sorted(df[config['color_col']].unique())
    year_to_color_idx = {year: idx for idx, year in enumerate(years)}

    for year in years:
        df_year = df[df[config['color_col']] == year]
        if not len(df_year):
            continue
        max_idx = df_year[config['y_col']].idxmax()
        base = df_base[df_base[config['color_col']]
                       == year][config['y_col']].values
        color = colors[year_to_color_idx[year] % len(colors)]

        fig.add_trace(
            graph_objects.Scatter(
                x=df_year[config['x_col']],
                y=df_year[config['y_col']],
                name=f"{year}",
                # legendgroup='Year',
                showlegend=show_year_legend,
                # showlegend=False,
                mode='lines',
                line=dict(color=color),
            ),
            row=row_num, col=col_num
        )
        fig.add_trace(
            graph_objects.Scatter(
                x=[df_year.loc[max_idx, config['x_col']]],
                y=[df_year.loc[max_idx, config['y_col']]],
                name=f"Max_{year}",
                # legendgroup='Year',
                # showlegend=show_year_legend,
                showlegend=False,
                mode='markers',
                marker=dict(
                    symbol='star', size=8,
                    color=color),
            ),
            row=row_num, col=col_num
        )


def add_month_subplot_traces(fig, df, df_base, row_num, col_num, config, show_month_legend=False):
    color_map = colors[::-1]
    df = df[df['Metric'] == config['metric_filter']]
    df_base = df_base[df_base['Metric'] == config['metric_filter']]
    if df.empty:
        print("Warning: No data for month subplot.")
        return

    # months = list(calendar.month_abbr)[1:]
    months = df[config['color_col']].unique()
    month_to_color_idx = {month: idx for idx, month in enumerate(months)}

    for month in months:
        df_month = df[df[config['color_col']] == month]
        if not len(df_month):
            continue
        max_idx = df_month[config['y_col']].idxmax()
        color = color_map[month_to_color_idx[month] % len(colors)]

        fig.add_trace(
            graph_objects.Scatter(
                x=df_month[config['x_col']],
                y=df_month[config['y_col']],
                name=f"{month}",
                # legendgroup='Month',
                showlegend=show_month_legend,
                # showlegend=False,
                mode='lines',
                line=dict(color=color),
            ),
            row=row_num, col=col_num
        )
        fig.add_trace(
            graph_objects.Scatter(
                x=[df_month.loc[max_idx, config['x_col']]],
                y=[df_month.loc[max_idx, config['y_col']]],
                name=f"Max_{month}",
                # legendgroup='Month',
                # showlegend=show_month_legend,
                showlegend=False,
                mode='markers',
                marker=dict(
                    symbol='star', size=8,
                    color=color),
            ),
            row=row_num, col=col_num
        )


if __name__ == "__main__":
    # --- Load Data ---
    df_year_time = pandas.read_csv('../data/year_time_melt.csv',)
    df_year_notime = pandas.read_csv('../data/year_notime_melt.csv')
    df_month_time = pandas.read_csv('../data/month_jandec_time_melt.csv')
    df_month_notime = pandas.read_csv('../data/month_jandec_notime_melt.csv')
    df_year_base_time = pandas.read_csv('../data/baseline_yt.csv')
    df_year_base_notime = pandas.read_csv('../data/baseline_yn.csv')
    df_month_base_time = pandas.read_csv('../data/baseline_mt.csv')
    df_month_base_notime = pandas.read_csv('../data/baseline_mn.csv')

    # --- Setup Subplot ---
    fig = subplots.make_subplots(
        rows=1, cols=4,
        shared_xaxes=True,
        shared_yaxes=True,
        y_title="nDCG@10",
        column_titles=['Explicit', 'Implicit',
                       'Explicit', 'Implicit'],
        horizontal_spacing=0.03
    )

    fig.update_layout(
        template='simple_white',
        width=700,
        height=170,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.26,
            xanchor='left',
            x=0.14,
            title=None
        ),
        font=dict(
            family='Times New Roman',
            # size=14
        ),
        margin=dict(l=60, r=0, t=40, b=0)
    )
    fig.update_xaxes(
        range=[-.05, 1.05]
        # No need to specify row/col when shared_xaxes=True
    )

    fig.add_annotation(
        text="<b>Yearly Interpolation</b>",  # Use HTML for bold
        align='center',
        x=0.13, y=1.25,  # Adjusted position relative to paper
        xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=16)
    )
    fig.add_annotation(
        text="<b>Monthly Interpolation</b>",  # Use HTML for bold
        align='center',
        x=0.88, y=1.25,  # Adjusted position relative to paper
        xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=16)
    )

    # --- Subplot Data Definition ---
    subplot_definitions = [
        {'type': 'year', 'df': df_year_time, 'df_base': df_year_base_time,
            'row': 1, 'col': 1, 'show_legend': False},
        {'type': 'year', 'df': df_year_notime, 'df_base': df_year_base_notime,
            'row': 1, 'col': 2, 'show_legend': True},
        {'type': 'month', 'df': df_month_time, 'df_base': df_month_base_time,
            'row': 1, 'col': 3, 'show_legend': False},
        {'type': 'month', 'df': df_month_notime, 'df_base': df_month_base_notime,
            'row': 1, 'col': 4, 'show_legend': True}
    ]

    for plot_info in subplot_definitions:
        if plot_info['type'] == 'year':
            add_year_subplot_traces(
                fig, plot_info['df'],
                plot_info['df_base'],
                plot_info['row'], plot_info['col'],
                CONFIG_YEAR, show_year_legend=plot_info['show_legend']
            )
        else:
            add_month_subplot_traces(
                fig, plot_info['df'],
                plot_info['df_base'],
                plot_info['row'], plot_info['col'],
                CONFIG_MONTH, show_month_legend=plot_info['show_legend']
            )

    # --- Save or Show ---
    try:
        fig.write_image("summary_ndcg10.pdf", format='pdf', engine="kaleido")
    except Exception as e:
        print(f"Warning: Could not save PDF: {e}")

    # fig.show()
