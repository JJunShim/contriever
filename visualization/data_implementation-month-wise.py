import pandas

FILE = "temp.csv"

if __name__ == "__main__":
    df = pandas.read_csv(FILE, header=None)

    df.columns = pandas.MultiIndex.from_arrays([
        ["Interpolation"] * 2 + ["NDCG@5"] * 3 + ["NDCG@10"] * 3,
        ["w1", "w2"] + ['Jan', 'Jun', 'Dec'] * 2
    ])
    df.columns = [
        '{}_{}'.format(str(col[0]).replace(' ', '').replace('@', ''), col[1])
        if isinstance(col, tuple) else col
        for col in df.columns
    ]
    df_long = df.melt(
        id_vars=["Interpolation_w1", "Interpolation_w2"],
        var_name="Metric_Month",
        value_name="Value"
    )

    df_long[["Metric", "Month"]] = df_long["Metric_Month"].str.extract(
        r"([A-Za-z0-9]+)_([A-Za-z]{3})")
    df_long["Interpolation"] = df_long.apply(
        lambda row: f"{row['Interpolation_w1']}:{row['Interpolation_w2']}",
        axis=1
    )

    df_long.to_csv('month_jundec_notime_melt.csv')
