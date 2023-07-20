### Library
from dash import Dash, dcc, html, Input, Output, callback, dash_table, State

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import date
from datetime import timedelta
import random
import calendar
import requests


from azure.storage.blob import BlobServiceClient


import plotly.express as px
import plotly.graph_objects as go


#####################################################################################################################################
# Helper Functions
####################################################################################################################################

# only works for yyyy-mm-dd
def StringToDate(string, seperation_character='-'):
    date_list = string.split(seperation_character)
    return(date(int(date_list[0]), int(date_list[1]), int(date_list[2])))


def npDatetimeToDate(inDate):
    return pd.Timestamp(inDate).to_pydatetime().date()

def add_month(number_of_months, month, year):
    if month + number_of_months > 12 :
        return (month + number_of_months) % 12, year + (month + number_of_months) // 12
    else:
        return month + number_of_months, year
    
def GetPrevDates(inDate, number_of_previous_dates):
    prevDates = []
    i = 1
    while(i <= number_of_previous_dates):
        prevDates.append(inDate - timedelta(days=i))
        i+=1
    return prevDates

def GetPrevMonths(inDate, number_of_previous_months, flag = 1):
    prevMonths = []
    i = 1
    year = inDate.year
    month = inDate.month

    while(i <= number_of_previous_months):
        month = month - 1
        if month < 1:
            month += 12
            year -=1
        prevMonths.append((year, month))
        i+=1
    
    if flag == 1: 
        return prevMonths
    else:
        return prevMonths[len(prevMonths) - 1]
    
#### To write logs into ADLS
def write_dataframe_to_datalake(df, StockSymbol="admin"):
    STORAGEACCOUNTURL = "https://adlszeus.blob.core.windows.net"
    STORAGEACCOUNTKEY = "ksL9a2OZFCiKFYPn6hzTNJcY4WI2Nq2xSsRlUD8cDH3dBBEvePAhJqErSP6QKN27so/2ayW3DnO7O8s4uPtUZA=="
    CONTAINERNAME = "price-prediction"
    # BLOBNAME = "AzureFunctionLogs/temp1.csv"
    BLOBNAME = "MaxStockStore" + "\\" + StockSymbol + "_" + ".json"

    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

    blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)

    blob_client_instance.upload_blob(data=df.to_json(orient='records'))

    return True

#### To write logs into ADLS
def read_dataframe_from_datalake(StockSymbol="admin"):
    STORAGEACCOUNTURL = "https://adlszeus.blob.core.windows.net"
    STORAGEACCOUNTKEY = "ksL9a2OZFCiKFYPn6hzTNJcY4WI2Nq2xSsRlUD8cDH3dBBEvePAhJqErSP6QKN27so/2ayW3DnO7O8s4uPtUZA=="
    CONTAINERNAME = "price-prediction"
    # BLOBNAME = "AzureFunctionLogs/temp1.csv"
    BLOBNAME = "MaxStockStore" + "\\" + StockSymbol + "_" + ".json"

    blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)

    blob_client_instance = blob_service_client_instance.get_blob_client(CONTAINERNAME, BLOBNAME, snapshot=None)

    # blob_client_instance.upload_blob(data=df.to_json(orient='records'))
    in_file = blob_client_instance.download_blob()

    return pd.read_json(in_file, orient="records") # type: ignore

#####################################################################################################################################


#####################################################################################################################################
# Functions
###################################################################################################################################

def CalculatePosition_BuyLastDayOfMonth(prev_n_months, input_df, amount=1000, inDate=date.today()):
    # initial variables
    number_of_stocks = 0
    amount_invested = 0

    Bought_at_df = pd.DataFrame()

    if amount < 0:
        return (0, 0, 0, Bought_at_df)


    stocks_bought_at = GetPrevMonths(inDate=inDate, number_of_previous_months=prev_n_months)

    current_value = input_df[input_df['Date'] == inDate]['Close'].to_list()

    if len(current_value) == 0:
        current_value = input_df.sort_values(by='Date')[input_df['Date'] < inDate].tail(1)['Close'].to_list()
        if len(current_value) == 0:
            return

    for month in stocks_bought_at:
        last_day = calendar.monthrange(month[0], month[1])[1]
        number_of_stocks_bought = 0

        if len(input_df[input_df['Date'] == date(month[0], month[1], last_day)]['Close'].values) == 0:
            selected_row_s = input_df.sort_values(by='Date')[input_df['Date'] < date(month[0], month[1], last_day)].tail(1)

            number_of_stocks_bought = amount // selected_row_s['Close'].values[0]

            amount_invested += selected_row_s['Close'].values[0] * number_of_stocks_bought
            selected_row_s['Number_of_Stocks_Bought'] = number_of_stocks_bought

            selected_row_s['Amount_Invested'] = round(selected_row_s['Close'].values[0] * number_of_stocks_bought, 2)
            selected_row_s['Current_Value'] = round(current_value[0] * number_of_stocks_bought, 2)


            selected_row_s['Mark_to_Market'] = round(current_value[0] * number_of_stocks_bought - selected_row_s['Close'].values[0] * number_of_stocks_bought,2)
            selected_row_s['%'] = round((current_value[0] * number_of_stocks_bought - selected_row_s['Close'].values[0] * number_of_stocks_bought) * 100 / (selected_row_s['Close'].values[0] * number_of_stocks_bought), 2)

            Bought_at_df = pd.concat([Bought_at_df, selected_row_s], ignore_index=True)

        else:
            selected_row_s = input_df[input_df['Date'] == date(month[0], month[1], last_day)]

            number_of_stocks_bought = amount // selected_row_s['Close'].values[0]

            amount_invested += selected_row_s['Close'].values[0] * number_of_stocks_bought
            selected_row_s['Number_of_Stocks_Bought'] = number_of_stocks_bought

            selected_row_s['Amount_Invested'] = round(selected_row_s['Close'].values[0] * number_of_stocks_bought, 2)
            selected_row_s['Current_Value'] = round(current_value[0] * number_of_stocks_bought, 2)

            selected_row_s['Mark_to_Market'] = round(current_value[0] * number_of_stocks_bought - selected_row_s['Close'].values[0] * number_of_stocks_bought, 2)
            selected_row_s['%'] = round((current_value[0] * number_of_stocks_bought - selected_row_s['Close'].values[0] * number_of_stocks_bought) * 100 / (selected_row_s['Close'].values[0] * number_of_stocks_bought), 2)


            Bought_at_df = pd.concat([Bought_at_df, selected_row_s], ignore_index=True)
        
        number_of_stocks += number_of_stocks_bought 


    try:
        return current_value[0] * number_of_stocks, amount_invested, current_value[0] * number_of_stocks - amount_invested, Bought_at_df
    except:
        return (0, 0, 0, Bought_at_df)


# 0 for Peak / 1 for Dip
# Also checks for % higher or lower condition
def CheckPeakOrDip(past_val, current_val, future_val, amount_per_stock, buy_percent=10, sell_percent=10,  flag=0, leeway = 5):
    # Peak
    if flag == 0:
        if current_val > past_val and current_val > future_val:
            if (current_val - amount_per_stock) * 100/amount_per_stock > sell_percent:
                return True
            elif (current_val - amount_per_stock) * 100/amount_per_stock < leeway:
                return True
            else:
                return False
        else:
            return False
    # Dip
    elif flag == 1:
        if current_val < past_val and current_val < future_val:
            if current_val < amount_per_stock*(1 - (buy_percent/100)):
                return True
            elif current_val < amount_per_stock*(1 + (leeway/100)):
                return True
            else: 
                return False
        else:   
            return False
        

def CalculatePosition_PeaksAndDips(prev_n_months, input_df, inDate=date.today(), initial_amount = 1000, buy_percent=10, sell_percent=10):
    stocks_bought_at = GetPrevMonths(inDate=inDate, number_of_previous_months=prev_n_months, flag=0)

    investment_date = date(int(stocks_bought_at[0]), int(stocks_bought_at[1]), int(inDate.day + 1))

    target_df = input_df[input_df['Date'] >= investment_date]
    target_df = target_df[input_df['Date'] <= inDate]

    target_df['Avg'] = target_df.apply(lambda row: round((row['High'] + row['Low']) / 2, 2), axis=1)

    target = target_df.sort_values(by=['Date']).to_dict(orient='records')

    # initial variables
    output = []
    number_of_stocks = 0
    initial_amount_invested = None

    amount_invested = 0
    amount_per_stock = 0
    amount_in_hand = 0

    past_val = None
    current_val = None
    future_val = None
    # Flag for Buy or Sell of Stocks
    # 0 - Bought - Check for peak and Sell
    # 1 - Sold - Check for dip and Buy
    BuySell = 0

    for data_point in target:
        #Setup
        if initial_amount_invested == None:
            # Initial Buy - Buys a set number of initial stocks
            number_of_stocks = initial_amount // data_point['Avg']
            initial_amount_invested = data_point['Avg'] * number_of_stocks
            amount_invested = data_point['Avg'] * number_of_stocks
            amount_per_stock = data_point['Avg']
            amount_in_hand = initial_amount - initial_amount_invested

            past_val = data_point['Avg']

            # print(f"amount_invested : {amount_invested}, amount_per_stock : {amount_per_stock}, amount_in_hand : {amount_in_hand}, number_of_stocks : {number_of_stocks}")
            temp = data_point
            temp['Amount_Invested'] = round(amount_invested,2)
            temp['Amount_Per_Stock'] = round(amount_per_stock,2)
            temp['Amount_In_Hand'] = round(amount_in_hand, 2)
            temp['Number_Of_Stocks_Bought'] = number_of_stocks
            temp['Mark_To_Market'] = round(amount_invested + amount_in_hand - initial_amount_invested, 2)
            temp['%'] = round((amount_invested + amount_in_hand - initial_amount_invested) * 100 / initial_amount_invested, 2)

            output.append(temp)

            # set flag to detect peak
            BuySell = 0 
            continue

        elif current_val == None:
            current_val = data_point['Avg']
            continue

        elif future_val == None:
            future_val = data_point['Avg']

            # Checking for Peak or Dip
            if CheckPeakOrDip(past_val=past_val, current_val=current_val, future_val=future_val, amount_per_stock=amount_per_stock, flag=BuySell, buy_percent=buy_percent, sell_percent=sell_percent):
                # selling
                if BuySell == 0:
                    amount_invested = 0
                    amount_per_stock = future_val
                    amount_in_hand += number_of_stocks * future_val
                    number_of_stocks = 0
                    BuySell = 1
                    # print(f"amount_invested : {amount_invested}, amount_per_stock : {amount_per_stock}, amount_in_hand : {amount_in_hand}, number_of_stocks : {number_of_stocks}, BuySell : {BuySell}")

                    temp = data_point
                    temp['Amount_Invested'] = round(amount_invested, 2)
                    temp['Amount_Per_Stock'] = round(amount_per_stock, 2)
                    temp['Amount_In_Hand'] = round(amount_in_hand, 2)
                    temp['Number_Of_Stocks_Bought'] = number_of_stocks
                    temp['Mark_To_Market'] = round(amount_invested + amount_in_hand - initial_amount_invested, 2)
                    temp['%'] = round((amount_invested + amount_in_hand - initial_amount_invested) * 100 / initial_amount_invested, 2)

                    output.append(temp)
                # buying 
                elif BuySell == 1:
                    number_of_stocks = int(amount_in_hand // future_val)
                    amount_invested += number_of_stocks * future_val
                    amount_per_stock = future_val
                    amount_in_hand -= amount_invested
                    BuySell = 0
                    # print(f"amount_invested : {amount_invested}, amount_per_stock : {amount_per_stock}, amount_in_hand : {amount_in_hand}, number_of_stocks : {number_of_stocks}, BuySell : {BuySell}")

                    temp = data_point
                    temp['Amount_Invested'] = round(amount_invested, 2)
                    temp['Amount_Per_Stock'] = round(amount_per_stock,2)
                    temp['Amount_In_Hand'] = round(amount_in_hand, 2)
                    temp['Number_Of_Stocks_Bought'] = number_of_stocks
                    temp['Mark_To_Market'] = round(amount_invested + amount_in_hand - initial_amount_invested, 2)
                    temp['%'] = round((amount_invested + amount_in_hand - initial_amount) * 100 / initial_amount, 2)

                    output.append(temp)
                
                past_val = current_val
                current_val = future_val
                future_val = None
            
            else:
                past_val = current_val
                current_val = future_val
                future_val = None
                    
    return round(initial_amount,2), round(amount_in_hand + amount_invested, 2), round(amount_in_hand + amount_invested - initial_amount), pd.DataFrame.from_records(output)



#####################################################################################################################################
# Global Variable
####################################################################################################################################

qqq_df = pd.read_csv("Data\\QQQ_max.csv")
qqq_df['Date'] = qqq_df.apply(lambda row: StringToDate(row['Date']), axis=1)
qqq_df['StockName'] =  'QQQ'

global_click_month = None
global_click_peaks = None
month_combined_details_df = None
peaks_dips_combined_details_df = None

month_combined_graph_df = None
peaks_dips_combined_graph_df = None

####################################################################################################################################


app = Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions=True

app.layout = html.Div([
    # one
    html.H2('Stock Monitor', style={'text-align' : 'center'}),

    html.Div([

        html.Div([
            html.Div("Symbols", style={'display' : 'flex', 'size' : '14px', 'padding' : '5px', 'margin' : 'initial'}),
            dcc.Checklist(
                id="StockSymbols",
                options=['QQQ', 'QLD', 'IBM', 'TSLA', 'RIVN', 'COIN', 'CGC', 'AZN', 'RIOT', 'MARA'],
                value = ['QQQ'],
                style= {'overflow' : 'overlay'},
                inputStyle={'padding' : '5px'},
                labelStyle={'padding' : '5px'}
            ),
        ], style={'display' : 'flex', 'height' : '20vh', 'flex-direction' : 'column', 'padding' : '10px'}),

        html.Div([
            html.Div(["Time Horizon"], className="Input_Box_Heading"),
            html.Div([dcc.Input(placeholder="Enter Time Horizon", id = 'time_input', type='number')], style={'display' : 'flex'}),
        ], style={'display' : 'flex', 'height' : '20vh', 'flex-direction' : 'column', 'padding' : '10px'}),

        html.Div([

            html.Div(["Monthly Investment"], className="Input_Box_Heading"),
            html.Div([dcc.Input(placeholder="Enter Monthly Investment", id = 'month_input', type='number')], style={'display' : 'flex'}),

            html.Div([html.Button("Monthly Purchase", style={'padding' : '5px'})], style={'overflow' : 'overlay', 'display' : 'flex', 'margin-top' : '10px'}),

        ], style={'display' : 'flex', 'height' : '20vh', 'flex-direction' : 'column', 'padding' : '10px'}),

        html.Div([

            html.Div(["Initial Investment"], className="Input_Box_Heading"),
            html.Div([dcc.Input(placeholder="Enter Initial Investment", id = 'peaks_dips_input', type='number')], style={'display' : 'flex'}),

            html.Div([
                html.Div([
                    html.Div(["Buy Trigger"], className="Input_Box_Heading"),
                    html.Div([dcc.Input(placeholder="%", id = 'buy_trigger', type='number')], style={'display' : 'flex'}),
                ], style={'display' : 'flex', 'flex-direction' : 'column'}),

                html.Div([
                    html.Div(["Sell Trigger"], className="Input_Box_Heading"),
                    html.Div([dcc.Input(placeholder="%", id = 'sell_trigger', type='number')], style={'display' : 'flex'}),
                ], style={'display' : 'flex', 'flex-direction' : 'column'})

            ], style={'display' : 'flex', 'flex-direction' : 'row'}),

            html.Div([html.Button("Peaks / Valley", style={'padding' : '5px'})], style={'overflow' : 'overlay', 'display' : 'flex', 'margin-top' : '10px'}),


        ], style={'display' : 'flex', 'height' : '20vh', 'flex-direction' : 'column', 'padding' : '10px'}),

        html.Div([

            html.Div([html.Button("Calculate", style={'padding' : '5px'})], id = 'calculate_on_click', style={'overflow' : 'overlay', 'display' : 'flex', 'margin-top' : '10px'}),

        ], style={'display' : 'flex', 'height' : '20vh', 'flex-direction' : 'column-reverse', 'padding' : '10px'}),

    ], style={'display' : 'flex', 'flex-direction' : 'row'}),

    html.Div([

        html.Div([

            html.Div([
                "Monthly Averaging : ", 
            ], style={'padding' : '10px'}),

            html.Div([
                ""
            ], id = 'month_output', style={'padding' : '10px'}),


        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

        html.Div([

            html.Div([
                "Dips and Peaks : ", 
            ], style={'padding' : '10px'}),

            html.Div([
                ""
            ], id = 'dips_peaks_output', style={'padding' : '10px'}),


        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'})

    ], style={'display' : 'flex', 'flex-direction' : 'row', 'width' : '98vw'}),

    html.Div([

        html.Div([

            html.Div([
                "Details : ", 
            ], style={'padding' : '10px'}),

            html.Div([""
            ], id = 'month_detail_output', style={'padding' : '10px'})


        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

        html.Div([
            
            html.Div([
                "Details : ", 
            ], style={'padding' : '10px'}),

            html.Div([""
            ], id = 'dips_peaks_detail_output', style={'padding' : '10px'})

        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

    ], style={'display' : 'flex', 'flex-direction' : 'row', 'width' : '98vw'}),

    html.Div([

        html.Div([

            html.Div([""
            ], id = 'month_detail_graph_output', style={'padding' : '10px'}),


        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

        html.Div([
            
           html.Div([""
            ], id = 'dips_peaks_detail_graph_output', style={'padding' : '10px'}),

        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

    ], style={'display' : 'flex', 'flex-direction' : 'row', 'width' : '98vw'}),

    html.Div([

        html.Div([

            html.Div([""
            ], id = 'month_detail_graph_selected_output', style={'padding' : '10px'}),


        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

        html.Div([
            
           html.Div([""
            ], id = 'dips_peaks_detail_graph_selected_output', style={'padding' : '10px'}),

        ], style={'display' : 'flex', 'flex-direction' : 'column', 'width' : '50%', 'overflow' : 'auto', 'padding' : '10px', 'margin' : '12px'}),

    ], style={'display' : 'flex', 'flex-direction' : 'row', 'width' : '98vw'})
])


# @callback(
#     Output(component_id='month_output', component_property='children'),
#     Output(component_id='month_detail_output', component_property='children'),

    
#     Input(component_id='time_input', component_property='value'),
#     Input(component_id='month_input', component_property='value'),
#     Input(component_id='StockSymbols', component_property='value'),
#     Input(component_id='calculate_on_click', component_property='n_clicks_timestamp')
# )
# def Monthly_Averaging_Callback(months, amount, stocksymbols, click):
#     global global_click_month 
#     if months == None or amount == None or stocksymbols == None or click == global_click_month:
#         return f'Monthly Averaging : ', f""

#     global_click_month = click
    
#     if len(stocksymbols) == 1:
#         data_df = read_dataframe_from_datalake(stocksymbols[0])
#         data_df['Date'] = data_df.apply(lambda row: npDatetimeToDate(row['Date']), axis=1)
#         returns, invested, profit, details_df = CalculatePosition_BuyLastDayOfMonth(months, data_df, amount=amount)
#     else:
#         returns, invested, profit, details_df = CalculatePosition_BuyLastDayOfMonth(months, qqq_df, amount=amount)

#     return f'Monthly Averaging : || Invested : ${round(invested,2)} || Returns : ${round(returns,2)} || Profit : ${round(profit,2)}', dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])


# @callback(
#     Output(component_id='dips_peaks_output', component_property='children'),
#     Output(component_id='dips_peaks_detail_output', component_property='children'),

    
#     Input(component_id='time_input', component_property='value'),
#     Input(component_id='peaks_dips_input', component_property='value'),
#     Input(component_id='buy_trigger', component_property='value'),
#     Input(component_id='sell_trigger', component_property='value'),
#     Input(component_id='calculate_on_click', component_property='n_clicks_timestamp')
# )
# def Peaks_Dips_Callback(months, amount, buy_percent, sell_percent, click):
#     global global_click_peaks 
#     if months == None or amount == None or buy_percent == None or sell_percent == None or click == global_click_peaks:
#         return f'Peaks and Dips : ', f""
    

#     global_click_peaks = click
    
#     invested, details_df = CalculatePosition_PeaksAndDips(months, qqq_df, inDate=date.today(), initial_amount = amount, buy_percent=buy_percent, sell_percent=sell_percent)

#     return f'Peaks and Dips : || Invested : ${round(invested,2)} ||', dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])


#######################################################
# Init Table Calculations Calculation
#####################################################

@callback(
    Output(component_id='month_output', component_property='children'),
    Output(component_id='month_detail_output', component_property='children', allow_duplicate=True),
    Output(component_id='dips_peaks_output', component_property='children'),
    Output(component_id='dips_peaks_detail_output', component_property='children', allow_duplicate=True),

    
    State(component_id='time_input', component_property='value'),
    State(component_id='month_input', component_property='value'),
    State(component_id='StockSymbols', component_property='value'),
    State(component_id='peaks_dips_input', component_property='value'),
    State(component_id='buy_trigger', component_property='value'),
    State(component_id='sell_trigger', component_property='value'),
    Input(component_id='calculate_on_click', component_property='n_clicks_timestamp'),

    prevent_initial_call=True
)

def Calculate(months, month_amount, stocksymbols, peak_amount, buy_percent, sell_percent, click):

    global global_click_month 

    global month_combined_details_df 
    global peaks_dips_combined_details_df 

    global month_combined_graph_df 
    global peaks_dips_combined_graph_df 

    output_month_detail = " "
    output_peaks_dips_detail = " "
    output_month = ' '
    output_peaks_dips = ' '


    if click != global_click_month:

        # Monthly Averaging
        if months == None or month_amount == None or stocksymbols == None:
            output_month = ' '

        else:
            if len(stocksymbols) == 1:
                data_df = read_dataframe_from_datalake(stocksymbols[0])
                data_df['Date'] = data_df.apply(lambda row: npDatetimeToDate(row['Date']), axis=1)
                returns, invested, profit, details_df = CalculatePosition_BuyLastDayOfMonth(months, data_df, amount=month_amount)

                output_month = f'|| Invested : ${round(invested,2)} || Returns : ${round(returns,2)} || Profit : ${round(profit,2)} || % : {round((profit * 100)/invested, 2)} ||'
                output_month_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])

            elif len(stocksymbols) >= 1:
                month_combined_details_df = None
                month_combined_graph_df = None
                output = []
                for symbol in stocksymbols:
                    data_df = read_dataframe_from_datalake(symbol)
                    data_df['Date'] = data_df.apply(lambda row: npDatetimeToDate(row['Date']), axis=1)
                    returns, invested, profit, details_df = CalculatePosition_BuyLastDayOfMonth(months, data_df, amount=month_amount)

                    # Details Table
                    if type(month_combined_details_df) == type(None):
                        month_combined_details_df = details_df
                    else:
                        # month_combined_details_df = month_combined_details_df.append(other=details_df, ignore_index = True)
                        month_combined_details_df = pd.concat([month_combined_details_df, details_df], ignore_index=True)

                    # Graph
                    if type(month_combined_graph_df) == type(None):
                        month_combined_graph_df = data_df
                    else:
                        # month_combined_graph_df = month_combined_graph_df.append(other=data_df, ignore_index = True)
                        month_combined_graph_df = pd.concat([month_combined_graph_df, data_df], ignore_index=True)

                    output.append({
                        'Stock Symbol' : symbol,
                        'Invested' :  round(invested,2),
                        'Returns' : round(returns,2),
                        'Profit' : round(profit,2),
                        '%' : round((profit * 100)/invested, 2)
                    })

                output_df = pd.DataFrame(output)

                output_month = dash_table.DataTable(output_df.to_dict("records"), [{"name": i, "id": i} for i in output_df.columns], id='month_tbl')
                output_month_detail = ""
            else:
                returns, invested, profit, details_df = CalculatePosition_BuyLastDayOfMonth(months, qqq_df, amount=month_amount)

                output_month = f'|| Invested : ${round(invested,2)} || Returns : ${round(returns,2)} || Profit : ${round(profit,2)} || % : {round((profit * 100)/invested, 2)} ||'
                output_month_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])


        # Peaks and Dips
        if months == None or peak_amount == None or buy_percent == None or sell_percent == None:
            output_peaks_dips = ' '
        
        else:        
            if len(stocksymbols) == 1:
                data_df = read_dataframe_from_datalake(stocksymbols[0])
                data_df['Date'] = data_df.apply(lambda row: npDatetimeToDate(row['Date']), axis=1)
                invested, returns, profit, details_df = CalculatePosition_PeaksAndDips(months, data_df, inDate=date.today(), initial_amount = peak_amount, buy_percent=buy_percent, sell_percent=sell_percent)

                output_peaks_dips = f'|| Invested : ${round(invested,2)} || Returns : ${round(returns,2)} || Profit : ${round(profit,2)} || % : {round((profit * 100)/invested, 2)} ||'
                output_peaks_dips_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])
            
            elif len(stocksymbols) >= 1:
                peaks_dips_combined_details_df = None
                peaks_dips_combined_graph_df = None
                output = []
                for symbol in stocksymbols:
                    data_df = read_dataframe_from_datalake(symbol)
                    data_df['Date'] = data_df.apply(lambda row: npDatetimeToDate(row['Date']), axis=1)
                    invested, returns, profit, details_df = CalculatePosition_PeaksAndDips(months, data_df, inDate=date.today(), initial_amount = peak_amount, buy_percent=buy_percent, sell_percent=sell_percent)

                    # Graph
                    if type(peaks_dips_combined_graph_df) == type(None):
                        peaks_dips_combined_graph_df = data_df
                    else:
                        # peaks_dips_combined_graph_df = peaks_dips_combined_graph_df.append(other=data_df, ignore_index = True)
                        peaks_dips_combined_graph_df = pd.concat([peaks_dips_combined_graph_df, data_df], ignore_index=True)

                    # Details Table
                    if type(peaks_dips_combined_details_df) == type(None):
                        peaks_dips_combined_details_df = details_df
                    else:
                        # peaks_dips_combined_details_df = peaks_dips_combined_details_df.append(other=details_df, ignore_index = True)
                        peaks_dips_combined_details_df = pd.concat([peaks_dips_combined_details_df, details_df], ignore_index=True)


                    output.append({
                        'Stock Symbol' : symbol,
                        'Invested' :  round(invested,2),
                        'Returns' : round(returns,2),
                        'Profit' : round(profit,2),
                        '%' : round((profit * 100)/invested, 2)
                    })

                output_df = pd.DataFrame(output)

                output_peaks_dips = dash_table.DataTable(output_df.to_dict("records"), [{"name": i, "id": i} for i in output_df.columns], id='peaks_dips_tbl')
                output_peaks_dips_detail = ""
            else:
                invested, returns, profit, details_df = CalculatePosition_PeaksAndDips(months, qqq_df, inDate=date.today(), initial_amount = peak_amount, buy_percent=buy_percent, sell_percent=sell_percent)

                output_peaks_dips = f'|| Invested : ${round(invested,2)} || Returns : ${round(returns,2)} || Profit : ${round(profit,2)} || % : {round((profit * 100)/invested, 2)} ||'
                output_peaks_dips_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])
            


        global_click_month = click
        return output_month, output_month_detail, output_peaks_dips, output_peaks_dips_detail
    elif click == None:
        return output_month, output_month_detail, output_peaks_dips, output_peaks_dips_detail


###########################################
# Month Init Graph
#########################################

@callback(
    Output(component_id='month_detail_output', component_property='children'),
    Output(component_id='month_detail_graph_output', component_property='children'),
    
    State(component_id='StockSymbols', component_property='value'),
    Input(component_id='month_tbl', component_property='active_cell'),
)
def Populate_Month_Details(stocksymbols, month_active):
    if len(stocksymbols) >= 1:
        output_month_detail = " "
        if type(month_combined_details_df) != type(None) and month_active != None and type(month_combined_graph_df) != type(None):
            details_df = month_combined_details_df[month_combined_details_df['StockSymbol'] == stocksymbols[month_active['row']]]
            output_month_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])

            data_df = month_combined_graph_df[month_combined_graph_df['StockSymbol'] == stocksymbols[month_active['row']]]
            data_df = data_df[data_df['Date'] >= details_df.tail(1)['Date'].values[0]]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_df['Date'], y=data_df['Close'],
                                mode='lines',
                                name='lines'))
            fig.add_trace(go.Scatter(x=details_df['Date'], y=details_df['Close'],
                                mode='markers', name='Buys', marker=dict(symbol = 'circle-dot', line_width=2, size=10)))
            
            fig.add_trace(go.Candlestick(x=details_df['Date'],
                open=details_df['Open'],
                high=details_df['High'],
                low=details_df['Low'],
                close=details_df['Close']))
            
            fig.update_layout(clickmode='event+select')

        
            return output_month_detail, dcc.Graph(figure=fig, id="month_graph")
    
    return " ", " "
    


##############################################
# Peaks and Dips Init Graph
############################################
@callback(
    Output(component_id='dips_peaks_detail_output', component_property='children'),
    Output(component_id='dips_peaks_detail_graph_output', component_property='children'),

    State(component_id='StockSymbols', component_property='value'),
    Input(component_id='peaks_dips_tbl', component_property='active_cell')
)
def Populate_Peaks_Dips_Details(stocksymbols, peaks_dips_active):
    if len(stocksymbols) >= 1:
        output_peaks_dips_detail = ""
        if type(peaks_dips_combined_details_df) != type(None) and peaks_dips_active != None and type(peaks_dips_combined_graph_df) != type(None):
            details_df = peaks_dips_combined_details_df[peaks_dips_combined_details_df['StockSymbol'] == stocksymbols[peaks_dips_active['row']]]
            output_peaks_dips_detail = dash_table.DataTable(details_df.to_dict("records"), [{"name": i, "id": i} for i in details_df.columns])

            data_df = peaks_dips_combined_graph_df[peaks_dips_combined_graph_df['StockSymbol'] == stocksymbols[peaks_dips_active['row']]]
            data_df = data_df[data_df['Date'] >= details_df.head(1)['Date'].values[0]]

            data_df['Avg'] = data_df.apply(lambda row: round((row['High'] + row['Low']) / 2, 2), axis=1)
            details_df['Avg'] = details_df.apply(lambda row: round((row['High'] + row['Low']) / 2, 2), axis=1)

            # details_df.drop(column='index')
                    
            details_df = details_df.reset_index(drop=True)
            details_df = details_df.reset_index()

            buy_df = details_df[details_df['index'] % 2 == 0]
            sell_df = details_df[details_df['index'] % 2 == 1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data_df['Date'], y=data_df['Avg'],
                                mode='lines',
                                name='lines'))
            fig.add_trace(go.Scatter(x=buy_df['Date'], y=buy_df['Avg'],
                                     mode='markers', name='Buys', marker=dict(symbol = 'circle-dot', line_width=2, size=10, color='lightgreen')))
            fig.add_trace(go.Scatter(x=sell_df['Date'], y=sell_df['Avg'],
                                     mode='markers', name='Sells', marker=dict(symbol = 'circle-dot', line_width=2, size=10, color='red')))
            
            fig.update_layout(clickmode='event+select')

        
            return output_peaks_dips_detail, dcc.Graph(figure=fig, id="peaks_dips_graph")
        
        return " ", " "
    

############################################################
# Month Detail Graph
##########################################################

@callback(
Output(component_id='month_detail_graph_selected_output', component_property='children'),

State(component_id='StockSymbols', component_property='value'),
State(component_id='month_tbl', component_property='active_cell'),
Input(component_id='month_graph', component_property='selectedData'),
)
def Populate_Month_Selected_Graph(stocksymbols, month_active, selected_data):
    if len(stocksymbols) >= 1:
        if type(month_combined_details_df) != type(None) and month_active != None and type(selected_data) != type(None):
            details_df = month_combined_details_df[month_combined_details_df['StockSymbol'] == stocksymbols[month_active['row']]]

            data_df = month_combined_graph_df[month_combined_graph_df['StockSymbol'] == stocksymbols[month_active['row']]]
            data_df = data_df[data_df['Date'] >= details_df.tail(1)['Date'].values[0]]

            selected_date = selected_data['points'][0]['x']
            selected_date = StringToDate(selected_date)


            selected_df = pd.concat([data_df[data_df['Date'] == selected_date][['Date', 'Open', 'High', 'Low', 'Close']], data_df[['Date', 'Open', 'High', 'Low', 'Close']].head(1)])

            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=selected_df['Date'],
                open=selected_df['Open'],
                high=selected_df['High'],
                low=selected_df['Low'],
                close=selected_df['Close']))
            
        
            return dcc.Graph(figure=fig)



####################################################
# Peaks and Dips Details Graph
###############################################
@callback(
Output(component_id='dips_peaks_detail_graph_selected_output', component_property='children'),

State(component_id='StockSymbols', component_property='value'),
State(component_id='peaks_dips_tbl', component_property='active_cell'),
Input(component_id='peaks_dips_graph', component_property='selectedData'),
)
def Populate_Peaks_Dips_Selected_Graph(stocksymbols, peaks_dips_active, selected_data):
    if len(stocksymbols) >= 1:
        if type(peaks_dips_combined_details_df) != type(None) and peaks_dips_active != None and type(selected_data) != type(None):
            details_df = peaks_dips_combined_details_df[peaks_dips_combined_details_df['StockSymbol'] == stocksymbols[peaks_dips_active['row']]]

            data_df = peaks_dips_combined_graph_df[peaks_dips_combined_graph_df['StockSymbol'] == stocksymbols[peaks_dips_active['row']]]
            data_df = data_df[data_df['Date'] >= details_df.head(1)['Date'].values[0]]

            selected_date = selected_data['points'][0]['x']
            selected_date = StringToDate(selected_date)


            selected_df = pd.concat([data_df[data_df['Date'] == selected_date][['Date', 'Open', 'High', 'Low', 'Close']], data_df[['Date', 'Open', 'High', 'Low', 'Close']].head(1)])

            fig = go.Figure()

            fig.add_trace(go.Candlestick(x=selected_df['Date'],
                open=selected_df['Open'],
                high=selected_df['High'],
                low=selected_df['Low'],
                close=selected_df['Close']))
            
        
            return dcc.Graph(figure=fig)



app.run_server(debug=False,dev_tools_ui=False,dev_tools_props_check=False)