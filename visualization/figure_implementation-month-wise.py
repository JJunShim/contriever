import calendar
import pandas
import plotly

from plotly import graph_objects, subplots

# --- Configuration ---
CONFIG = {
    'subplot': [1, 4],
    'template': 'simple_white',
    'color': plotly.colors.qualitative.Bold,
    'metric_filter': 'NDCG5',
    'x_col': 'Interpolation_w2',
    # 'x_col': 'Interpolation',
    'y_col': 'Value',
    'color_col': 'Month',
    'category_order': ['1.0:0.0', '0.9:0.1', '0.8:0.2', '0.7:0.3', '0.6:0.4', '0.5:0.5', '0.4:0.6', '0.3:0.7', '0.2:0.8', '0.1:0.9', '0.0:1.0']
}

# --- Plotting Function ---


def add_subplot_traces(fig, csv_path, row_num, col_num, config, show_month_legend=False):
    """
    Loads data from CSV, filters, and adds traces for a single subplot.

    Args:
        fig: The plotly figure object.
        csv_path (str): Path to the CSV file.
        row_num (int): Subplot row number.
        col_num (int): Subplot column number.
        config (dict): Configuration dictionary.
        show_month_legend (bool): Whether to show legend entries for months in this subplot.
    """
    try:
        df = pandas.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return  # Skip this subplot if file not found

    # Filter data
    # Use .copy() to avoid SettingWithCopyWarning
    df_filtered = df[df['Metric'] == config['metric_filter']].copy()
    if df_filtered.empty:
        print(
            f"Warning: No data found for metric '{config['metric_filter']}' in {csv_path}")
        return  # Skip if no data after filtering

    x_col = config['x_col']
    y_col = config['y_col']
    color_col = config['color_col']
    colors = config['color']

    # Ensure consistent month order for coloring
    months = list(calendar.month_abbr)[1:]
    month_to_color_idx = {month: i for i, month in enumerate(months)}

    # 1. Add Mean Line
    mean_val = df_filtered.groupby(x_col)[y_col].mean()
    fig.add_trace(
        graph_objects.Scatter(
            x=mean_val.index,
            y=mean_val.values,
            mode='lines',
            name='Mean',  # Name is useful even if legend hidden for hover
            line=dict(color='lightgray', dash='dash', width=3),
            showlegend=False  # Typically hide the mean legend per subplot
        ),
        row=row_num, col=col_num
    )

    # 2. Add Lines for each Month
    for month in months:
        month_df = df_filtered[df_filtered[color_col] == month]
        color_idx = month_to_color_idx[month]
        fig.add_trace(
            graph_objects.Scatter(
                x=month_df[x_col],
                y=month_df[y_col],
                mode='lines',
                name=str(month),  # Legend entry for the month
                # Cycle through colors
                line=dict(color=colors[color_idx % len(colors)]),
                legendgroup=str(month),  # Group traces for the same month
                showlegend=show_month_legend  # Control legend visibility
            ),
            row=row_num, col=col_num
        )

    # 3. Add Max Value Markers for each Month
    # Find index of max value for each month group
    max_indices = df_filtered.loc[df_filtered.groupby(color_col)[
        y_col].idxmax()]

    for _, row_data in max_indices.iterrows():
        month = row_data[color_col]
        color_idx = month_to_color_idx[month]
        fig.add_trace(
            graph_objects.Scatter(
                x=[row_data[x_col]],
                y=[row_data[y_col]],
                mode='markers',
                name=f'Max {month}',  # Useful for hover text
                marker=dict(
                    symbol='star',
                    size=8,
                    color=colors[color_idx % len(colors)]
                ),
                # Associate with the month's line color
                legendgroup=str(month),
                showlegend=False  # Hide legend entry for max points
            ),
            row=row_num, col=col_num
        )


if __name__ == "__main__":
    # --- Figure Setup ---
    fig = subplots.make_subplots(
        rows=CONFIG['subplot'][0], cols=CONFIG['subplot'][1],
        # Dynamic y-title
        y_title=f'{CONFIG["metric_filter"].replace("NDCG", "nDCG@")}',
        column_titles=['Jan to Jun', 'Jun to Dec', 'Jan to Jun', 'Jun to Dec'],
        # column_widths=[0.22, 0.22, 0.22, 0.22],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=.03
    )

    # --- Layout Updates ---
    fig.update_layout(
        width=800,
        height=270,  # Slightly increased height for better legend/title spacing
        template=CONFIG['template'],
        legend=dict(
            orientation='h',
            yanchor='bottom',  # Anchor legend below annotations/titles
            y=1.3,           # Position legend above plots
            xanchor='left',
            x=0,
            title_text=CONFIG['color_col'].capitalize()  # Dynamic legend title
        ),
        font=dict(
            family="serif"
        ),
        # Adjust margins (esp. top for annotations/legend)
        margin=dict(l=70, r=70, t=50, b=30)
    )

    # Apply category order to all shared x-axes
    fig.update_xaxes(
        # categoryorder='category descending',
        # categoryarray=CONFIG['category_order'],
        range=[-.05, 1.05]
        # No need to specify row/col when shared_xaxes=True
    )

    # --- Annotations ---
    fig.add_annotation(
        text="<b>With Timestamp</b>",  # Use HTML for bold
        align='center',
        x=0.14, y=1.15,  # Adjusted position relative to paper
        xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=14)
    )
    fig.add_annotation(
        text="<b>Without Timestamp</b>",  # Use HTML for bold
        align='center',
        x=0.87, y=1.15,  # Adjusted position relative to paper
        xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=14)
    )

    # --- Data Sources and Plotting Loop ---
    subplot_data = [
        {'csv': 'month_janjun_time_melt.csv', 'row': 1, 'col': 1,
            'show_legend': False},  # Show legend only once
        {'csv': 'month_jundec_time_melt.csv',
            'row': 1, 'col': 2, 'show_legend': False},
        {'csv': 'month_janjun_notime_melt.csv',
            'row': 1, 'col': 3, 'show_legend': False},
        {'csv': 'month_jundec_notime_melt.csv',
            'row': 1, 'col': 4, 'show_legend': True},
    ]

    for data_info in subplot_data:
        add_subplot_traces(
            fig,
            data_info['csv'],
            data_info['row'],
            data_info['col'],
            CONFIG,
            show_month_legend=data_info['show_legend']
        )

    # --- Final Output ---
    # Save to PDF (optional)
    try:
        fig.write_image("month_wise_ndcg.pdf", format='pdf', engine="kaleido")
    except Exception as e:
        print(
            f"Could not save PDF, maybe Orca or Kaleido is not installed? Error: {e}")

    # fig.show()
