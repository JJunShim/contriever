import pandas
import plotly

from plotly import graph_objects, subplots

# --- Configuration ---
CONFIG = {
    'subplot': [1, 4],
    'template': 'simple_white',
    'color': plotly.colors.qualitative.Bold,
    'x_col': 'Interpolation_w2',  # Make sure this matches CSV header
    'y_col': 'Value',
    'color_col': 'Year',
    'category_order': ['1.0:0.0', '0.9:0.1', '0.8:0.2', '0.7:0.3', '0.6:0.4', '0.5:0.5', '0.4:0.6', '0.3:0.7', '0.2:0.8', '0.1:0.9', '0.0:1.0']
}

# --- Plotting Function ---


def add_year_subplot_traces(fig, df, metric_filter, row_num, col_num, config, show_year_legend=False):
    """
    Filters data and adds traces for a single subplot (Year version).
    """
    df_filtered = df[df['Metric'] == metric_filter].copy()
    if df_filtered.empty:
        print(f"Warning: No data found for metric '{metric_filter}'")
        return

    x_col = config['x_col']
    y_col = config['y_col']
    color_col = config['color_col']
    colors = config['color']

    # Consistent color mapping
    unique_years = sorted(df_filtered[color_col].unique())
    year_to_color_idx = {year: i for i, year in enumerate(unique_years)}

    # 1. Add Mean Line
    # Ensure grouping column exists and is correct
    if x_col in df_filtered.columns:
        mean_val = df_filtered.groupby(x_col, observed=True)[
            y_col].mean()  # observed=True for category
        fig.add_trace(
            graph_objects.Scatter(
                x=mean_val.index,
                y=mean_val.values,
                mode='lines',
                name='Mean',  # Useful for hover
                line=dict(color='lightgray', dash='dash', width=3),
                showlegend=False
            ),
            row=row_num, col=col_num
        )
    else:
        print(f"Warning: Column '{x_col}' not found for mean calculation.")

    # 2. Add Lines for each Year
    for year in unique_years:
        year_df = df_filtered[df_filtered[color_col] == year]
        if year_df.empty:
            continue
        color_idx = year_to_color_idx[year]
        fig.add_trace(
            graph_objects.Scatter(
                x=year_df[x_col],
                y=year_df[y_col],
                mode='lines',
                name=str(year),  # Legend entry for the year
                line=dict(color=colors[color_idx % len(colors)]),
                legendgroup=str(year),  # Group traces for the same year
                showlegend=show_year_legend  # Control legend visibility
            ),
            row=row_num, col=col_num
        )

    # 3. Add Max Value Markers for each Year
    if not df_filtered.empty:
        # Use idxmax for robustness with categorical index if mean calculation worked
        try:
            max_indices = df_filtered.loc[df_filtered.groupby(color_col)[
                y_col].idxmax()]
            for _, row_data in max_indices.iterrows():
                year = row_data[color_col]
                color_idx = year_to_color_idx[year]
                fig.add_trace(
                    graph_objects.Scatter(
                        x=[row_data[x_col]],
                        y=[row_data[y_col]],
                        mode='markers',
                        name=f'Max {year}',  # Useful for hover text
                        marker=dict(
                            symbol='star',
                            size=8,
                            color=colors[color_idx % len(colors)]
                        ),
                        # Associate with the year's line color
                        legendgroup=str(year),
                        showlegend=False  # Hide legend entry for max points
                    ),
                    row=row_num, col=col_num
                )
        except KeyError as e:
            print(
                f"Warning: Could not find max index, possibly due to grouping issues. Error: {e}")


if __name__ == "__main__":
    # --- Load Data Once ---
    try:
        df_time = pandas.read_csv('year_time_melt.csv')
    except FileNotFoundError:
        print("Error: year_time_melt.csv not found.")
        df_time = pandas.DataFrame()  # Create empty df to avoid later errors

    try:
        df_notime = pandas.read_csv('year_notime_melt.csv')
    except FileNotFoundError:
        print("Error: year_notime_melt.csv not found.")
        df_notime = pandas.DataFrame()  # Create empty df

    # --- Figure Setup ---
    fig = subplots.make_subplots(
        rows=CONFIG['subplot'][0], cols=CONFIG['subplot'][1],
        column_titles=['nDCG@5', 'nDCG@10', 'nDCG@5', 'nDCG@10'],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=.03
    )

    # --- Layout Updates ---
    fig.update_layout(
        width=800,
        height=270,  # Slightly increased height for legend/annotations
        template=CONFIG['template'],
        legend=dict(
            orientation='h',
            yanchor='bottom',  # Anchor below annotations/titles
            y=1.3,         # Position above plots (adjust as needed)
            xanchor='left',
            x=0,
            # Dynamic legend title 'Year'
            title_text=CONFIG['color_col'].capitalize()
        ),
        font=dict(
            family="serif"
        ),
        margin=dict(l=70, r=70, t=50, b=30)  # Adjust margins (esp. top)
    )

    # Apply category order to all shared x-axes
    fig.update_xaxes(
        # categoryorder='array',
        # categoryarray=CONFIG['category_order']
        range=[-.05, 1.05]
        # No need to specify row/col when shared_xaxes=True
    )

    # --- Annotations ---
    fig.add_annotation(
        text="<b>With Timestamp</b>",
        align='center',
        # Adjust position relative to paper (centered over first two plots)
        x=0.14, y=1.15,
        xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=14)
    )
    fig.add_annotation(
        text="<b>Without Timestamp</b>",
        align='center',
        # Adjust position relative to paper (centered over last two plots)
        x=.87, y=1.15,
        xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=14)
    )

    # --- Data Sources and Plotting Loop ---
    subplot_definitions = [
        {'df': df_time, 'metric': 'NDCG5', 'row': 1, 'col': 1, 'show_legend': False},
        {'df': df_time, 'metric': 'NDCG10', 'row': 1,
            'col': 2, 'show_legend': False},
        {'df': df_notime, 'metric': 'NDCG5',
            'row': 1, 'col': 3, 'show_legend': False},
        {'df': df_notime, 'metric': 'NDCG10', 'row': 1, 'col': 4,
            'show_legend': True},  # Show legend only here
    ]

    for plot_info in subplot_definitions:
        if not plot_info['df'].empty:  # Check if dataframe was loaded successfully
            add_year_subplot_traces(
                fig,
                plot_info['df'],
                plot_info['metric'],
                plot_info['row'],
                plot_info['col'],
                CONFIG,
                show_year_legend=plot_info['show_legend']
            )

    # --- Final Output ---
    try:
        fig.write_image("year_wise_ndcg.pdf", format='pdf', engine="kaleido")
    except Exception as e:
        print(
            f"Could not save PDF, maybe Orca or Kaleido is not installed? Error: {e}")

    # fig.show()
