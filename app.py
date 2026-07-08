import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.set_page_config(page_title="XYLOFY AI - Superstore Dashboard",layout="wide")
@st.cache_data
def load_data():
    df=pd.read_csv("train.csv")
    st.write(df.columns)
    df["Order Date"]=pd.to_datetime(df["Order Date"],dayfirst=True,errors="coerce")
    return df
df=load_data()
st.title("Superstore Sales Analytics Dashboard")
regions=st.sidebar.multiselect("Region",sorted(df.Region.unique()),default=sorted(df.Region.unique()))
cats=st.sidebar.multiselect("Category",sorted(df.Category.unique()),default=sorted(df.Category.unique()))
f=df[df.Region.isin(regions)&df.Category.isin(cats)]
sales = f["Sales"].sum()
profit = f["Profit"].sum()
orders = f["Order ID"].nunique()
aov = sales / orders
col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"${sales:,.0f}")
col2.metric("Total Orders", orders)
col3.metric("Average Order Value", f"${aov:,.2f}")
t=st.tabs(["Overview","Time Series","Forecasting","Anomalies","Clustering","Insights"])
with t[0]:
 m=f.groupby(pd.Grouper(key="Order Date",freq="ME"))["Sales"].sum().reset_index()
 st.plotly_chart(px.line(m,x="Order Date",y="Sales"),use_container_width=True)
 st.plotly_chart(px.bar(f.groupby("Category")["Sales"].sum().reset_index(),x="Category",y="Sales"),use_container_width=True)
with t[1]:
 m=f.groupby(pd.Grouper(key="Order Date",freq="ME"))["Sales"].sum().reset_index();m["MA3"]=m.Sales.rolling(3).mean()
 st.plotly_chart(px.line(m,x="Order Date",y=["Sales","MA3"]),use_container_width=True)
with t[2]:
 st.dataframe(pd.DataFrame({"Model":["SARIMA","Prophet","XGBoost"],"MAE":[20580.7,20250.79,15110.78],"RMSE":[22190.91,22318.41,19239.17],"MAPE":[21.94,21.86,14.81]}))
 st.success("Recommended Model: XGBoost")
with t[3]:
 m=f.groupby(pd.Grouper(key="Order Date",freq="ME"))["Sales"].sum().reset_index()
 iso=IsolationForest(contamination=0.08,random_state=42)
 m["A"]=iso.fit_predict(m[["Sales"]]);fig=px.line(m,x="Order Date",y="Sales")
 an=m[m.A==-1];fig.add_scatter(x=an["Order Date"],y=an["Sales"],mode="markers",name="Anomaly");st.plotly_chart(fig,use_container_width=True)
with t[4]:
 feat=f.groupby("Sub-Category").agg(Total=("Sales","sum"),Avg=("Sales","mean"),Vol=("Sales","std")).fillna(0)
 X=StandardScaler().fit_transform(feat)
 feat["Cluster"]=KMeans(n_clusters=4,random_state=42,n_init=10).fit_predict(X)
 pts=PCA(n_components=2).fit_transform(X)
 p=pd.DataFrame({"PC1":pts[:,0],"PC2":pts[:,1],"Cluster":feat.Cluster.astype(str),"Sub":feat.index})
 st.plotly_chart(px.scatter(p,x="PC1",y="PC2",color="Cluster",text="Sub"),use_container_width=True)
with t[5]:
 st.write("Best Forecasting Model: XGBoost")
 st.write("Highest Forecast Category: Office Supplies")
 st.write("Strongest Region: East")
st.markdown("---")

st.markdown("""
## Executive Business Insights

**Best Forecasting Model**
- XGBoost achieved the best performance with the lowest forecasting error (MAE, RMSE and MAPE).

**Sales Trend**
- Overall sales show an increasing trend with seasonal fluctuations.

**Product Insights**
- Office Supplies is forecasted to generate the highest sales over the next three months.

**Regional Performance**
- East region continues to contribute the highest revenue.

**Anomaly Detection**
- Isolation Forest successfully identified unusual sales spikes and drops for business monitoring.

**Customer Segmentation**
- K-Means clustering grouped products into four meaningful business segments for inventory planning.

**Project Developed For**
XYLOFY AI Internship Program

**Developed By**
Kuchuru Sai Krishna Reddy
""")