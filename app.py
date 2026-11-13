import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(
    page_title="Stock Market Risk Prediction",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("stock_risk_random_forest_bundle.pkl")

model_bundle = load_model()

model = model_bundle["model"]

feature_columns = model_bundle["feature_columns"]

df = pd.read_csv(
    "nse_stock_historical_data_cleaned.csv"
)


st.set_page_config(
    page_title="Stock Market Risk Prediction",
    page_icon="📈",
    layout="wide"
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a Page",
    [
        "Business Problem",
        "Data Insights",
        "Prediction"
    ]
)




# PAGE 1: BUSINESS PROBLEM

if page == "Business Problem":

    st.title("Stock Market Risk Prediction")
    st.caption("Machine Learning Based Financial Risk Analytics")

    st.divider()

    # 1. Project Overview
    st.header("Project Overview")
    st.write("Introduce your NSE stock risk prediction project.")

    # 2. Business Problem
    st.header("Business Problem")
    st.write("Explain why investors need to assess stock market risk.")

    # 3. Analytics Objective
    st.header("Analytics Objective")
    st.write("Explain how your Random Forest model classifies stock risk.")

    st.divider()

    # 4. Project Information
    st.header("Project Information")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Observations", f"{df.shape[0]:,}")
    col2.metric("Dataset Variables", df.shape[1])
    col3.metric("ML Algorithm", "Random Forest")

    # 5. Dataset Information
    st.header("Dataset Information")
    st.dataframe(df.head(), use_container_width=True)

    # 6. Target Variable
    st.subheader("Target Variable")
    st.write("Risk_Category")
    st.write("Classes: Low, Medium and High")

    # 7. Business Decision
    st.subheader("Business Decision")
    st.write("Explain how investors can use the predicted risk category.")

    st.info(
        "The model prediction is a decision-support input "
        "and should not be treated as a guaranteed investment outcome."
    )


# PAGE 2: DATA INSIGHTS

elif page == "Data Insights":

    st.title("Stock Market Data Insights")
    st.caption("Exploratory Data Analysis | NSE Historical Stock Data")

    st.divider()

    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"])

    # Stock Selection
    st.subheader("Select Stock")

    stock_list = sorted(df["Stock"].dropna().unique())

    selected_stock = st.selectbox(
        "Choose a stock",
        stock_list
    )

    # Filter selected stock
    stock_df = df[df["Stock"] == selected_stock].copy()

    stock_df = stock_df.sort_values("Date")

    # Feature Engineering for Analysis
    stock_df["Daily_Return"] = (
        stock_df["Close"].pct_change() * 100
    )

    stock_df["Volatility_20"] = (
        stock_df["Daily_Return"].rolling(20).std()
    )

    stock_df["Volatility_60"] = (
        stock_df["Daily_Return"].rolling(60).std()
    )

    stock_df["Moving_Average_20"] = (
        stock_df["Close"].rolling(20).mean()
    )

    stock_df["Moving_Average_60"] = (
        stock_df["Close"].rolling(60).mean()
    )

    # Key Statistics
    st.subheader("Key Financial Statistics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Closing Price",
        f"₹{stock_df['Close'].mean():,.2f}"
    )

    col2.metric(
        "Highest Price",
        f"₹{stock_df['High'].max():,.2f}"
    )

    col3.metric(
        "Lowest Price",
        f"₹{stock_df['Low'].min():,.2f}"
    )

    col4.metric(
        "Average Volume",
        f"{stock_df['Volume'].mean():,.0f}"
    )

    st.divider()

    # Chart 1: Historical Closing Price
    st.subheader("1. Historical Stock Price Trend")

    fig1 = px.line(
        stock_df,
        x="Date",
        y="Close",
        title=f"{selected_stock} - Closing Price Trend",
        labels={
            "Date": "Date",
            "Close": "Closing Price (₹)"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Chart 2: Moving Average
    st.subheader("2. Moving Average Analysis")

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["Close"],
        name="Closing Price"
    ))

    fig2.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["Moving_Average_20"],
        name="20-Day Moving Average"
    ))

    fig2.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["Moving_Average_60"],
        name="60-Day Moving Average"
    ))

    fig2.update_layout(
        title="Closing Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (₹)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Chart 3: Daily Return Distribution
    st.subheader("3. Daily Return Distribution")

    return_df = stock_df.dropna(subset=["Daily_Return"])

    fig3 = px.histogram(
        return_df,
        x="Daily_Return",
        nbins=50,
        title="Distribution of Daily Stock Returns",
        labels={"Daily_Return": "Daily Return (%)"}
    )

    st.plotly_chart(fig3, use_container_width=True)

    # Chart 4: Rolling Volatility
    st.subheader("4. Stock Volatility Analysis")

    volatility_df = stock_df.dropna(
        subset=["Volatility_20", "Volatility_60"]
    )

    fig4 = go.Figure()

    fig4.add_trace(go.Scatter(
        x=volatility_df["Date"],
        y=volatility_df["Volatility_20"],
        name="20-Day Volatility"
    ))

    fig4.add_trace(go.Scatter(
        x=volatility_df["Date"],
        y=volatility_df["Volatility_60"],
        name="60-Day Volatility"
    ))

    fig4.update_layout(
        title="Rolling Volatility Trend",
        xaxis_title="Date",
        yaxis_title="Standard Deviation of Daily Returns (%)"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # Data Preview
    st.divider()

    st.subheader("Selected Stock Data Preview")

    st.dataframe(
        stock_df,
        use_container_width=True
    )



elif page == "Prediction":

    st.title("Stock Risk Prediction")
    st.caption("Random Forest Based Financial Risk Classification")

    st.divider()

    # 1. STOCK SELECTION
    st.subheader("Select Stock")

    stock_list = sorted(df["Stock"].dropna().unique())

    selected_stock = st.selectbox(
        "Choose a stock",
        stock_list
    )

    # 2. FILTER SELECTED STOCK
    stock_df = df[df["Stock"] == selected_stock].copy()

    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df = stock_df.sort_values("Date").reset_index(drop=True)

    # 3. DATE SELECTION
    available_dates = stock_df["Date"].iloc[60:]

    if len(available_dates) == 0:
        st.warning("Not enough historical data for this stock.")
    else:

        selected_date = st.selectbox(
            "Choose Historical Date",
            available_dates.dt.strftime("%Y-%m-%d").tolist()
        )

        selected_date = pd.to_datetime(selected_date)

        # 4. FILTER DATA UP TO SELECTED DATE
        prediction_df = stock_df[
            stock_df["Date"] <= selected_date
        ].copy()

        prediction_df = prediction_df.sort_values("Date")

        # 5. FEATURE ENGINEERING
        prediction_df["Daily_Return"] = (
            prediction_df["Close"].pct_change()
        )

        prediction_df["Volatility_20"] = (
            prediction_df["Daily_Return"]
            .rolling(20).std() * np.sqrt(252)
        )

        prediction_df["Volatility_60"] = (
            prediction_df["Daily_Return"]
            .rolling(60).std() * np.sqrt(252)
        )

        prediction_df["MA20"] = (
            prediction_df["Close"].rolling(20).mean()
        )

        prediction_df["MA60"] = (
            prediction_df["Close"].rolling(60).mean()
        )

        prediction_df["Price_to_MA20"] = (
            prediction_df["Close"] / prediction_df["MA20"] - 1
        )

        prediction_df["Price_to_MA60"] = (
            prediction_df["Close"] / prediction_df["MA60"] - 1
        )

        prediction_df["Volume_Change"] = (
            prediction_df["Volume"].pct_change()
        )

        prediction_df["Log_Volume_Change"] = (
            np.sign(prediction_df["Volume_Change"])
            * np.log1p(np.abs(prediction_df["Volume_Change"]))
        )

        # 6. DISPLAY CALCULATED FEATURES
        st.divider()
        st.subheader("Calculated Model Features")

        input_data = prediction_df[feature_columns].tail(1)

        st.dataframe(
            input_data,
            use_container_width=True
        )

        # 7. CHECK MISSING VALUES
        if input_data.isna().any().any():

            st.warning(
                "Some features are missing. "
                "Please select another historical date."
            )

        else:

            # 8. PREDICTION
            st.divider()
            st.subheader("Stock Risk Prediction")

            if st.button("Predict Risk", type="primary"):

                prediction = model.predict(input_data)
                probabilities = model.predict_proba(input_data)

                predicted_category = prediction[0]

                st.subheader("Prediction Result")

                st.write(
                    f"Selected Stock: {selected_stock}"
                )

                st.write(
                    f"Selected Date: {selected_date.date()}"
                )

                st.write(
                    f"Predicted Risk Category: {predicted_category}"
                )

                # 9. PROBABILITY TABLE
                probability_df = pd.DataFrame({
                    "Risk Category": model.classes_,
                    "Probability": probabilities[0]
                })

                probability_df["Probability (%)"] = (
                    probability_df["Probability"] * 100
                ).round(2)

                st.subheader("Risk Category Probabilities")

                st.dataframe(
                    probability_df[
                        ["Risk Category", "Probability (%)"]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

                # 10. PROBABILITY CHART
                st.subheader("Prediction Probability Chart")

                st.bar_chart(
                    probability_df.set_index("Risk Category")[
                        "Probability (%)"
                    ]
                )

                # 11. RISK INTERPRETATION
                st.divider()
                st.subheader("Risk Interpretation")

                if predicted_category == "Low":

                    st.success(
                        "The model classifies this stock "
                        "in the Low volatility-risk category."
                    )

                elif predicted_category == "Medium":

                    st.warning(
                        "The model classifies this stock "
                        "in the Medium volatility-risk category."
                    )

                else:

                    st.error(
                        "The model classifies this stock "
                        "in the High volatility-risk category."
                    )

                st.caption(
                    "This is a model-based volatility-risk estimate, "
                    "not a guarantee of future market performance "
                    "or an investment recommendation."
                )

