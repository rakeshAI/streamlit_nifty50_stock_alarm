import streamlit as st

# Use the full page instead of a narrow central column
st.set_page_config(layout="wide")
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import base64
import datetime

st.sidebar.title("NIFTY_50")


@st.cache
def load_data():
    url = "https://en.wikipedia.org/wiki/NIFTY_50"
    html = pd.read_html(url, header=0)
    df = html[1]
    return df


df = load_data()


companies = st.beta_expander("Company selection with sectors", expanded=True)
with companies:

    col1, col2 = st.beta_columns(2)
    sector = df.groupby("Sector")

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df["Sector"].unique())
    # st.write(sorted_sector_unique)
    selected_sector = col1.multiselect(
        "Sector", sorted_sector_unique, sorted_sector_unique
    )

    # Filtering data
    df_selected_sector = df[(df["Sector"].isin(selected_sector))]
    # if st.checkbox("Display Companies in Selected Sector"):
    #    st.header("Display Companies in Selected Sector")
    #    st.dataframe(df_selected_sector)
    col1.write("Number of companies: " + str(df_selected_sector.shape[0]))

    col2.dataframe(df_selected_sector)


stock_history_expander = st.beta_expander(
    "NIFTY50 stock history from yahoo finance", expanded=True
)
with stock_history_expander:

    data = yf.download(
        tickers=list(df_selected_sector[:].Symbol),
        period="ytd",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        prepost=True,
        threads=True,
        proxy=None,
    )
    st.write(data)

# st.info("Algorithmic trading strategy")
st.info("Data is updated till {}".format(data.index[-1]))

def buy_sell(signal):
    Buy = []
    Sell = []
    flag = -1

    for i in range(0, len(signal)):
        if signal["MACD"][i] > signal["Signal line"][i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(signal["Close"][i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif signal["MACD"][i] < signal["Signal line"][i]:
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(signal["Close"][i])
                flag = 0
            else:
                Sell.append(np.nan)
        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy, Sell)


def MACD(symbol):
    # st.header(symbol)
    df = pd.DataFrame(data[symbol].Close)
    df["Date"] = df.index
    # calculate the MACD and signal line indicators
    # calculate the short term exponential moving average
    ShortEMA = df.Close.ewm(span=12, adjust=False).mean()
    # st.write(ShortEMA)
    # calculate the long term exponential moving average
    LongEMA = df.Close.ewm(span=26, adjust=False).mean()
    # calculate the MACD line
    MACD = ShortEMA - LongEMA
    # calculate the singnal line
    signal = MACD.ewm(span=9, adjust=False).mean()

    df["MACD"] = MACD
    df["Signal line"] = signal

    a = buy_sell(df)
    df["Buy_Signal_Price"] = a[0]
    df["Sell_Signal_Price"] = a[1]
    return df


def MFI(symbol):
    # st.header(symbol)
    df = pd.DataFrame(data[symbol])
    df["Date"] = df.index
    typical_price = (df["Close"] + df["High"] + df["Low"]) / 3
    df["Typical price"] = typical_price
    # Get the period
    period = 14
    # calculate the money flow
    money_flow = typical_price * df["Volume"]

    # get all the positive and negative money flow
    positive_flow = []
    negative_flow = []
    # loop through the typical price
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            negative_flow.append(money_flow[i-1])
            positive_flow.append(0)
        else:
            negative_flow.append(0)
            positive_flow.append(0)
    # get all the positive and negative money flows within the time period
    positive_mf = []
    negative_mf = []
    for i in range(period-1, len(positive_flow)):
        positive_mf.append( sum(positive_flow[i+1-period: i+1]))
    for i in range(period-1, len(negative_flow)):
        negative_mf.append( sum(negative_flow[i+1-period: i+1]))

    # calculate the money flow index
    mfi = 100 * (np.array(positive_mf) / np.array(positive_mf) + np.array(negative_mf) )
    df2 = pd.DataFrame()
    df2['MFI'] = mfi
    # create the plot
    fig = plt.figure()
    #plt.figure(figsize=(12,4))
    plt.plot(df2["MFI"], label='MFI')
    plt.axhline(10, linestyle= '--', color='orange')
    plt.axhline(20, linestyle= '--', color='blue')
    plt.axhline(80, linestyle= '--', color='blue')
    plt.axhline(90, linestyle= '--', color='orange')
    plt.title("MFI")
    plt.ylabel("MFI values")
    st.pyplot(fig)


def MACD_plot(df):
    col1, col2 = st.beta_columns(2)
    col1.write(df)
    fig = plt.figure()
    plt.scatter(
        df.index,
        df["Buy_Signal_Price"],
        color="green",
        label="Buy",
        marker="^",
        alpha=1,
    )
    plt.scatter(
        df.index,
        df["Sell_Signal_Price"],
        color="red",
        label="Sell",
        marker="v",
        alpha=1,
    )
    plt.plot(df["Close"], label="Close Price", alpha=0.50)
    plt.plot(df["MACD"], label="MACD", color="red", alpha=0.35)
    plt.plot(df["Signal line"], label="Signal Line", color="black", alpha=0.35)
    plt.title("CLose Price Buy & Sell Signals")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc="best")
    return col2.pyplot(fig)


algo_trad_strat_radio = st.sidebar.radio(
    "Select the Algorithmic trading strategy", ("MACD", "MFI", "Moving Average")
)
if algo_trad_strat_radio == "MACD":
    st.sidebar.info("MACD strategy")
    st.sidebar.write("Short EMA : 12, Long EMA : 26, Signal : 9")
    BUY = []
    SELL = []
    for i in list(df_selected_sector.Symbol)[:]:
        df_MACD = MACD(i)
        buy = df_MACD["Buy_Signal_Price"][-1]
        sell = df_MACD["Sell_Signal_Price"][-1]
        if not (np.isnan(buy)):
            BUY.append(i)
        elif not (np.isnan(sell)):
            SELL.append(i)
    date_now = datetime.datetime.now()
    st.write(date_now)
    col1, col2 = st.beta_columns(2)
    col1.success("Buy")
    col2.error("Sell")
    col1.write(BUY)
    col2.write(SELL)
    if st.checkbox("show graphs"):
        for num, i in enumerate(list(df_selected_sector.Symbol)[:]):
            text = str(num + 1) + "." + i
            df_MACD = MACD(i)
            buy = df_MACD["Buy_Signal_Price"][-1]
            sell = df_MACD["Sell_Signal_Price"][-1]
            expanded = False
            if not (np.isnan(buy)):
                expanded = True
            elif not (np.isnan(sell)):
                expanded = True
            company_expander = st.beta_expander(text, expanded=expanded)
            with company_expander:
                if not (np.isnan(buy)):
                    st.success("BUY")
                elif not (np.isnan(sell)):
                    st.error("SELL")
                MACD_plot(df_MACD)

elif algo_trad_strat_radio == "MFI":
    st.sidebar.info("MFI strategy")
    st.write("Coming soon..")
    #BUY = []
    #SELL = []
    #for i in list(df_selected_sector.Symbol)[:]:
    #    MFI(i)

elif algo_trad_strat_radio == "Moving Average":
    # st.info("Moving Average of closing price")
    st.write("Coming soon..")
