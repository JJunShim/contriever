import pandas

FILE = "temp.csv"

if __name__ == "__main__":
    df = pandas.read_csv(FILE, header=None)

    df.columns = pandas.MultiIndex.from_arrays([
        ["Interpolation", "Interpolation"] + ["NDCG@5"] * 4 + ["NDCG@10"] * 4 +
        ["Recall@5"] * 4 + ["Recall@10"] * 4,
        ["w1", "w2"] + [2018, 2019, 2020, 2021] * 4
    ])
    df.columns = [
        '{}_{}'.format(str(col[0]).replace(' ', '').replace('@', ''), col[1])
        if isinstance(col, tuple) else col
        for col in df.columns
    ]

    df_long = df.melt(
        id_vars=["Interpolation_w1", "Interpolation_w2"],
        var_name="Metric_Year",
        value_name="Value"
    )

    # 분리: Metric, Year
    df_long[["Metric", "Year"]] = df_long["Metric_Year"].str.extract(
        r"([A-Za-z0-9]+)_([0-9]{4})")
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Interpolation"] = df_long.apply(
        lambda row: f"{row['Interpolation_w1']}:{row['Interpolation_w2']}",
        axis=1
    )

    df_long.to_csv('year_time_melt.csv')
