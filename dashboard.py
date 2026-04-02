import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

##################################
# PAGE CONFIG
##################################

st.set_page_config(
page_title="TransferIQ",
layout="wide"
)

st.title("TransferIQ – AI Player Transfer Value Prediction")

st.markdown(
"AI ensemble system combining **LSTM time-series forecasting**, **XGBoost regression** and **sentiment analysis**."
)

##################################
# CACHE LOADING
##################################

@st.cache_data
def load_data():
    return pd.read_csv("final_ensemble_dataset.csv")

df = load_data()

##################################
# SIDEBAR
##################################

st.sidebar.header("Prediction Controls")

selected_player = st.sidebar.selectbox(
"Player",
sorted(df["player_name"].unique())
)

player_df = df[df["player_name"]==selected_player]

selected_season = st.sidebar.selectbox(
"Season",
sorted(player_df["season"].unique())
)

future_years = st.sidebar.slider(
"Future Seasons",
1,
5,
2
)

predict = st.sidebar.button("Generate Prediction")

##################################
# FORECAST FUNCTION
##################################

def future_forecast(values,n):

    if len(values)<2:
        return [values[-1]]*n

    trend=np.polyfit(
    range(len(values)),
    values,
    1
    )

    future=[]

    last=len(values)

    for i in range(n):

        pred=trend[0]*(last+i)+trend[1]

        future.append(pred)

    return future

##################################
# MAIN OUTPUT
##################################

if predict:

    player_data = player_df[
    player_df["season"]==selected_season
    ]

    if player_data.empty:

        st.warning("No data available")

    else:

##################################
# PLAYER HEADER
##################################

        st.subheader(f"{selected_player} Performance Dashboard")

##################################
# PROFILE
##################################

        col1,col2,col3,col4 = st.columns(4)

        if "age" in player_data.columns:
            col1.metric("Age", int(player_data["age"].values[0]))
        else:
            col1.metric("Age","N/A")

        if "goals" in player_data.columns:
            col2.metric("Goals",int(player_data["goals"].values[0]))
        else:
            col2.metric("Goals","N/A")

        if "assists" in player_data.columns:
            col3.metric("Assists",int(player_data["assists"].values[0]))
        else:
            col3.metric("Assists","N/A")

        if "minutes" in player_data.columns:
            col4.metric("Minutes",int(player_data["minutes"].values[0]))
        else:
            col4.metric("Minutes","N/A")

##################################
# PREDICTION SUMMARY
##################################

        st.subheader("Value Prediction")

        current_value = player_data[
        "market_value_eur"
        ].values[0]

        ensemble_pred = player_data[
        "final_prediction"
        ].values[0]

        change = ensemble_pred-current_value

        if current_value != 0:
            growth = (change/current_value)*100
        else:
            growth = 0

        col1,col2,col3,col4,col5 = st.columns(5)

        col1.metric(
        "Current Value",
        f"€{current_value:,.0f}"
        )

        col2.metric(
        "Ensemble Prediction",
        f"€{ensemble_pred:,.0f}"
        )

        col3.metric(
        "Change",
        f"€{change:,.0f}"
        )

        col4.metric(
        "Growth",
        f"{growth:.2f}%"
        )


##################################
# FUTURE FORECAST
##################################

        st.subheader("Future Forecast")

        future_predictions = future_forecast(

        player_df["final_prediction"].values,

        future_years

        )

        last_season = player_df["season"].sort_values().iloc[-1]

        start=int(last_season[:4])

        future_seasons=[]

        for i in range(future_years):

            y1=start+i+1
            y2=y1+1

            future_seasons.append(

            f"{y1}/{str(y2)[2:]}"

            )

        future_df=pd.DataFrame({

        "Season":future_seasons,

        "Predicted Value":future_predictions

        })

        st.dataframe(future_df)

##################################
# TREND GRAPH
##################################

        st.subheader("Market Trend")

        fig,ax=plt.subplots(figsize=(10,5))

        ax.plot(

        player_df["season"],

        player_df["market_value_eur"],

        marker="o",

        label="Actual"

        )

        ax.plot(

        player_df["season"],

        player_df["final_prediction"],

        linestyle="dashed",

        marker="o",

        label="Model"

        )

        ax.plot(

        future_seasons,

        future_predictions,

        linestyle="dotted",

        marker="o",

        label="Forecast"

        )

        ax.legend()

        plt.xticks(rotation=45)

        st.pyplot(fig)

##################################
# SENTIMENT ANALYSIS
##################################

        st.subheader("Sentiment Analysis")

        fig2,ax1=plt.subplots(figsize=(10,5))

        ax1.plot(

        player_df["season"],

        player_df["market_value_eur"],

        marker="o"

        )

        ax1.set_ylabel("Market Value")

        ax2=ax1.twinx()

        ax2.plot(

        player_df["season"],

        player_df["vader_compound_score"],

        marker="o",

        color="orange"

        )

        ax2.set_ylabel("Sentiment")

        plt.xticks(rotation=45)

        st.pyplot(fig2)

##################################
# INSIGHTS
##################################

        sentiment=player_df[
        "vader_compound_score"
        ]

        value=player_df[
        "market_value_eur"
        ]

        if len(sentiment)>1:

            corr=sentiment.corr(value)

        else:

            corr=0

        col1,col2=st.columns(2)

        col1.metric(

        "Avg Sentiment",

        round(sentiment.mean(),3)

        )

        col2.metric(

        "Sentiment Impact",

        round(corr,2)

        )

##################################
# INSIGHT MESSAGE
##################################

        if growth>8:

            st.success(
            "Strong upward market trajectory"
            )

        elif growth>2:

            st.info(
            "Moderate growth expected"
            )

        elif growth>-3:

            st.warning(
            "Stable valuation expected"
            )

        else:

            st.error(
            "Possible decline in value"
            )

##################################
# DOWNLOAD
##################################

        st.download_button(

        "Download Prediction",

        future_df.to_csv(index=False).encode("utf-8"),

        "prediction.csv"

        )

else:

    st.info(
    "Select player and click Generate Prediction"
    )
