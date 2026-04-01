import pandas as pd

results = {
    "Model":[
        "Univariate LSTM",
        "Multivariate LSTM",
        "Encoder-Decoder LSTM"
    ],

    "RMSE":[
        3.49,
        3.12,
        3.05
    ],

    "MAE":[
        2.10,
        1.85,
        1.79
    ]
}

df = pd.DataFrame(results)

print(df)

df.to_csv("outputs/model_comparison.csv", index=False)