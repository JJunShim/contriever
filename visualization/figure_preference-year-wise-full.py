import pandas

from plotly import graph_objects, subplots

FILE = 'temp.csv'
CONFIG = {
    'subplot': [6, 4],
    'spacing': [0.03, 0.05],
    'fig_size': (700, 750),
    'margin': dict(l=0, r=0, t=50, b=0),
    'font': dict(family="Times New Roman"),
    'template': 'simple_white',
    'color_scale': "sunsetdark",
    'axis': [2018, 2019, 2020, 2021],
}
titles = [
    '1.0:0.0', '0.4:0.6', '1.0:0.0', '0.4:0.6', '0.9:0.1', '0.3:0.7', '0.9:0.1', '0.3:0.7',
    '0.8:0.2', '0.2:0.8', '0.8:0.2', '0.2:0.8', '0.7:0.3', '0.1:0.9', '0.7:0.3', '0.1:0.9',
    '0.6:0.4', '0.0:1.0', '0.6:0.4', '0.0:1.0', '0.5:0.5', '', '0.5:0.5', ''
]
titles = [u"\U0001D6FC"+" = "+t.split(':')[-1] if t else '' for t in titles]

if __name__ == "__main__":
    # Load data
    df = pandas.read_csv(FILE, header=None)

    # 서브플롯 생성
    fig = subplots.make_subplots(
        rows=CONFIG['subplot'][0],
        cols=CONFIG['subplot'][1],
        shared_xaxes=True, shared_yaxes=True,
        horizontal_spacing=CONFIG['spacing'][0],
        vertical_spacing=CONFIG['spacing'][1],
        subplot_titles=titles
    )

    fig.update_layout(
        width=CONFIG['fig_size'][0], height=CONFIG['fig_size'][1],
        template=CONFIG['template'],
        font=CONFIG['font'],
        margin=CONFIG['margin'],
    )
    fig.update_yaxes(autorange="reversed")

    # 타이틀 추가
    fig.add_annotation(
        text="Explicit", x=0.19, y=1.03,
        xref="paper", yref="paper",
        xanchor="left", yanchor="bottom",
        showarrow=False, font=dict(size=16)
    )
    fig.add_annotation(
        text="Implicit", x=0.8, y=1.03,
        xref="paper", yref="paper",
        xanchor="right", yanchor="bottom",
        showarrow=False, font=dict(size=16)
    )

    # 히트맵 추가 (한 번에)
    for side in range(2):  # 0 = With Timestamp, 1 = Without Timestamp
        idx = 0
        for j in range(1 + side*2, 3 + side*2):  # 왼쪽(1,2), 오른쪽(3,4)
            for i in range(1, CONFIG['subplot'][0]+1):
                if idx >= 11:
                    break  # 0~11 (12개)
                temp = df.iloc[idx*4:(idx+1)*4,
                               :4] if side == 0 else df.iloc[idx*4:(idx+1)*4, 4:]
                heatmap = graph_objects.Heatmap(
                    z=temp.values,
                    x=CONFIG['axis'],
                    y=CONFIG['axis'],
                    colorscale=CONFIG['color_scale'],
                    showscale=False,
                    texttemplate="%{z:.2f}"
                )
                fig.add_trace(heatmap, row=i, col=j)
                idx += 1

    # 결과 저장
    fig.write_image("year_wise_document_preference_full.pdf", format='pdf')
    # fig.show()
