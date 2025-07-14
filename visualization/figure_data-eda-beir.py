import numpy
import pandas
import statsmodels.api as sm

from plotly import express, graph_objects

FILE = "temp.csv"

if __name__ == "__main__":
    df = pandas.read_csv(FILE)

    df = pandas.concat([df[1:].idxmax(), df[:1].T], axis=1)
    df.columns = ["idx", "year"]
    df = df.astype(
        {
            "idx": "int",
            "year": "int"
        }
    )
    df["idx"] = (df.idx - df.idx.min()) / (df.idx.max() - df.idx.min())

    # 1. OLS fit
    X = df["year"]
    y = df["idx"]
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # 2. 부드러운 x 구간 생성 (예: 2015~2021까지 100개 점)
    x_fit = numpy.linspace(2015.25, 2020.75, 100)
    x_fit_with_const = sm.add_constant(x_fit)
    y_fit = model.predict(x_fit_with_const)

    # 3. Plotly Express로 본 plot 생성
    fig = express.scatter(
        df,
        x="year",
        y="idx",
        text=df.index,
        color="idx",
        color_continuous_scale="sunsetdark"
    )

    # 4. trendline 직접 추가 (부드러운 선)
    fig.add_trace(graph_objects.Scatter(
        x=x_fit,
        y=y_fit,
        mode="lines",
        name="Trendline",
        line=dict(color="rgb(227, 79, 111)", width=2),
        showlegend=False
    ))

    # 5. 스타일 적용
    fig.update_traces(
        textposition=[
            "middle left", "bottom left", "bottom center", "top center", "top left",
            "bottom left", "bottom right", "middle left", "top center", "bottom center",
            "bottom center", "bottom center", "top left", "bottom center"
        ]
    )
    fig.update_layout(
        width=350,
        height=200,
        margin=dict(l=50, r=0, t=0, b=0),
        xaxis_title="Dataset Creation Year",
        # yaxis_title=r"Best Interpolation (α for 2021 Year)",
        yaxis_title="Best Interpolation ("+u"\U0001D6FC"+")",
        font=dict(
            family='serif',
            # size=14
        ),
        template='simple_white',
        showlegend=False,
        coloraxis_showscale=False
    )
    fig.update_xaxes(
        range=[2015, 2021],
        tickvals=[2016, 2017, 2018, 2019, 2020],
    )
    fig.write_image("beir_best_interpolation.pdf",
                    format='pdf', engine="kaleido")
    # fig.show()
