#Import all necessary libraries
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import inventorize3 as inv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

#Title of the App
st.write ("## INVENTORY MANAGEMENT DASHBOARD")
st.markdown('''This is a dashboard showing the Historical Items sales visualization, ABC Analysis of Acive/Current Inventory,Historical Inventory Turnover Ratio (KPI) &
Machine Learning Predictive Analysis.''')


st.header('Historical Sales Items Visualization')
st.markdown('**------------------------------------------------------------------------------------------------------------------------------**')

sales_records = pd.read_csv("Saleskaggle3.csv")
sales_records_hist = sales_records[['SKU_number','SoldCount','PriceReg','ItemCount']][sales_records['File_Type'] == 'Historical']
top_sales = sales_records_hist.sort_values(['SoldCount'], ascending=False).head(20)
buttom_sales = sales_records_hist.sort_values(['SoldCount'], ascending=True).head(20)

if st.checkbox('Show Detailed Historical Top 20 SKU with highest Sold Count'):
    st.subheader('Detailed: SKU with Highest SoldCount Inventory')
    st.write(top_sales)
    
st.subheader('SKU with Highest SoldCount Inventory')
st.bar_chart(top_sales['SoldCount'])

if st.checkbox('Show Detailed Historical Top 20 SKU with Lowest Sold Count'):
    st.subheader('Detailed: SKU with Lowest Historical SoldCount Inventory')
    st.write(buttom_sales)


st.header('Historical Records: Turnover Ratio Metric')
st.markdown('**------------------------------------------------------------------------------------------------------------------------------**')
sales_records = pd.read_csv("C:/Users/robou/OneDrive/Documents/CETMPROM02_Project/Project Dissertation/ML_Pipeline/Data_Set/Saleskaggle3.csv")
kp_indicator = sales_records[['SoldCount','PriceReg','ItemCount','File_Type']][sales_records['File_Type'] == 'Historical']
kp_indicator['COGS'] = kp_indicator['SoldCount']*kp_indicator['PriceReg']
kp_indicator['ItemCount_AfterSales'] = kp_indicator['ItemCount']-kp_indicator['SoldCount']
kp_indicator['Beg_Inv'] = kp_indicator['ItemCount']*kp_indicator['PriceReg']
kp_indicator['End_Inv'] = kp_indicator['ItemCount_AfterSales']*kp_indicator['PriceReg']
Tot_COGS = kp_indicator['COGS'].sum()
Tot_Beg_Inv = kp_indicator['Beg_Inv'].sum()
Tot_End_Inv = kp_indicator['End_Inv'].sum()
Ave_Inv = (Tot_Beg_Inv + Tot_End_Inv)/2
Inv_turnOver = Tot_COGS/Ave_Inv
st.markdown('***The inventory turnover ratio is a measure of how many times the inventory is sold and replaced over a given period (Case Study - Six(6) Months).***')
st.markdown('* Low inventory turnover : **A rate of 1 or less: You have excess inventory**  ')
st.markdown('* High inventory turnover : **A rate of 11 and above: Insufficient inventory to support sales at that rate**  ')
st.markdown('* Good inventory turnover : **A rate of 5 to 10: Sell and Restock of inventory evenly distributed**  ')

st.markdown('The Inventory turnover ratio is shown below:')
st.write(Inv_turnOver)



st.header('Active/Current Items Inventory Visualization')
st.markdown('**------------------------------------------------------------------------------------------------------------------------------**')

sales_records_act = sales_records[['SKU_number','SoldCount','PriceReg','ItemCount']][sales_records['File_Type'] == 'Active']
top_count = sales_records_act.sort_values(['ItemCount'], ascending=False).head(20)
buttom_count = sales_records_act.sort_values(['ItemCount'], ascending=True).head(20)

if st.checkbox('Show Detailed Active Top 20 SKU with Highest ItemCount Inventory'):
    st.subheader('Detailed: SKU with Highest ItemCount Inventory')
    st.write(top_count)

st.subheader('SKU with Highest ItemCount Inventory')    
fig = px.pie(top_count, values = 'ItemCount', names  = 'SKU_number')
st.write(fig)

if st.checkbox('Show Detailed Active Top 20 SKU with Lowest ItemCount Inventory'):
    st.subheader('Detailed: SKU with Lowest Active ItemCount Inventory')
    st.write(buttom_count)
    


st.header('ABC Analysis of Active/Current Inventory')
st.markdown('**------------------------------------------------------------------------------------------------------------------------------**')

sales_records = pd.read_csv("C:/Users/robou/OneDrive/Documents/CETMPROM02_Project/Project Dissertation/ML_Pipeline/Data_Set/Saleskaggle3.csv")
sales_records_act = sales_records[['SKU_number','PriceReg','ItemCount','File_Type']][sales_records['File_Type'] == 'Active']
sales_records_act['AddCost'] = sales_records_act['PriceReg'] * sales_records_act['ItemCount']
sales_records_act_1 = sales_records_act.groupby(['SKU_number']).agg(Volume=('ItemCount',np.sum),Revenue=('AddCost',np.sum)).reset_index()
sales_records_act_abc = inv.ABC(sales_records_act_1[['SKU_number', 'Volume']])
st.dataframe(sales_records_act_abc)
sales_records_act_summary = sales_records_act_abc.groupby('Category').agg(Count=('Category', np.count_nonzero), Percentage=('Percentage', np.sum))
sales_records_act_summary['Percentage']=sales_records_act_summary['Percentage']*100
st.dataframe(sales_records_act_summary)
ABC_Analysis = pd.DataFrame(sales_records_act_summary['Count'])
st.bar_chart(ABC_Analysis)

with st.sidebar:
    st.subheader('About:')
    st.markdown('This Inventory MGT. Dashboard is made by **PATRICK UGBUGBA**, using **Streamlit**')
   


st.header('Machine Learning Predictive Analysis')
st.markdown('**------------------------------------------------------------------------------------------------------------------------------**')

model = pickle.load(open('C:/Users/robou/OneDrive/Documents/CETMPROM02_Project/Project Dissertation/ML_Pipeline/SoldFlag_pred_model','rb'))


def main():
    st.subheader('Machine Learning Prediction')

    #input Variables
    ReleaseNumber = st.text_input("ReleaseNumber")
    New_Release_Flag = st.text_input("New_Release_Flag")
    StrengthFactor = st.text_input("StrengthFactor")
    PriceReg = st.text_input("PriceReg")
    ItemCount = st.text_input("ItemCount")
    LowUserPrice = st.text_input("LowUserPrice")
    LowNetPrice = st.text_input("LowNetPrice")
    Historical = st.text_input("Historical")

    #prediction code
    if st.button('Predict'):
        makepredictions = model.predict([[ReleaseNumber, New_Release_Flag, StrengthFactor,
        PriceReg, ItemCount, LowUserPrice, LowNetPrice, Historical]])
        output =round(makepredictions[0],2)
        st.success('Indication of sale or no-sale in six months is: {}'.format(output))

if __name__ == "__main__":
   main()

    
    















    
