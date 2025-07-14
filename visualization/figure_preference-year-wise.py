import pandas

from plotly import graph_objects, subplots

FILE = 'temp.csv'
CONFIG = {
    'template': 'simple_white',
    'subplot': (1, 4),
    'fig_size': (800, 180),
    'margin': dict(l=0, r=0, t=30, b=30),
    'font_family': "serif",
    'color_scale': "sunsetdark",
    'years': [2018, 2019, 2020, 2021],
    'slice_length': 4
}

if __name__ == "__main__":
    # Load data
    df = pandas.read_csv(FILE, header=None)

    # Titles and indices to plot
    title = [u"\U0001D6FC"+" = "+str(i/10) for i in range(11)]
    selected_idxs = [0, 3, 7, 10]

    # Create subplots
    fig = subplots.make_subplots(
        rows=CONFIG['subplot'][0], cols=CONFIG['subplot'][1],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=0.03, vertical_spacing=0.05,
        subplot_titles=[title[i] for i in selected_idxs]
    )

    # Add heatmaps
    for idx, (row, col) in enumerate([
        (i, j)
        for i in range(1, CONFIG['subplot'][0]+1)
        for j in range(1, CONFIG['subplot'][1]+1)
    ]):
        data_idx = selected_idxs[idx]
        temp = df.iloc[
            data_idx * CONFIG['slice_length']: (data_idx + 1) * CONFIG['slice_length'],
            :4
        ]

        heatmap = graph_objects.Heatmap(
            z=temp.values,
            x=CONFIG['years'],
            y=CONFIG['years'],
            colorscale=CONFIG['color_scale'],
            showscale=False,
            texttemplate="%{z:.2f}"
        )
        fig.add_trace(heatmap, row=row, col=col)

    # Layout settings
    fig.update_layout(
        width=CONFIG['fig_size'][0],
        height=CONFIG['fig_size'][1],
        template=CONFIG['template'],
        font=dict(family=CONFIG['font_family']),
        margin=CONFIG['margin']
    )
    fig.update_yaxes(autorange="reversed")

    # Save and show
    fig.write_image("year_wise_document_preference.pdf", format='pdf')
    # fig.show()
