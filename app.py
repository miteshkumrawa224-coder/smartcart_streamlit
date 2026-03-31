import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

st.set_page_config(page_title="SmartCart AI", page_icon="🛒", layout="wide")

# Copy of your ML logic directly in Streamlit!
def process_data(df):
    df_cleaned = df.copy()
    if "Income" in df_cleaned.columns:
        df_cleaned["Income"] = df_cleaned["Income"].fillna(df_cleaned["Income"].median())
        
    if "Year_Birth" in df_cleaned.columns:
        df_cleaned["Age"] = 2026 - df_cleaned["Year_Birth"]
        
    if "Dt_Customer" in df_cleaned.columns:
        df_cleaned["Dt_Customer"] = pd.to_datetime(df_cleaned["Dt_Customer"], dayfirst=True, errors="coerce")
        reference_date = df_cleaned["Dt_Customer"].max()
        df_cleaned["Customer_Tenure_Days"] = (reference_date - df_cleaned["Dt_Customer"]).dt.days
    
    df_cleaned["Total_Spending"] = df_cleaned.get("MntWines", 0) + df_cleaned.get("MntFruits", 0) + \
                                   df_cleaned.get("MntMeatProducts", 0) + df_cleaned.get("MntFishProducts", 0) + \
                                   df_cleaned.get("MntSweetProducts", 0) + df_cleaned.get("MntGoldProds", 0)
    
    if "Education" in df_cleaned.columns:
        df_cleaned["Education"] = df_cleaned["Education"].replace({
            "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
            "Graduation": "Graduate",
            "Master": "Postgraduate", "PhD": "Postgraduate"
        })
    
    if "Marital_Status" in df_cleaned.columns:
        df_cleaned["Living_With"] = df_cleaned["Marital_Status"].replace({
            "Married": "Partner", "Together": "Partner",
            "Single": "Alone", "Divorced": "Alone",
            "Widow": "Alone", "Absurd": "Alone", "YOLO": "Alone"
        })
        
    cols_to_drop = [c for c in ["ID", "Year_Birth", "Marital_Status", "Kidhome", "Teenhome", "Dt_Customer", 
                               "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"] 
                    if c in df_cleaned.columns]
    
    df_f = df_cleaned.drop(columns=cols_to_drop).dropna()
    
    if "Age" in df_f.columns:
        df_f = df_f[df_f["Age"] < 100]
    if "Income" in df_f.columns:
        df_f = df_f[df_f["Income"] < 600000]
        
    if "Education" in df_f.columns and "Living_With" in df_f.columns:
        X = pd.get_dummies(df_f, columns=["Education", "Living_With"])
    else:
        X = df_f.copy()
        
    scaler = StandardScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    df_f["Segment"] = ["Cluster " + str(c + 1) for c in clusters]
    return df_f

# UI Design
st.title("🛒 SmartCart AI - Customer Segmentation")
st.markdown("Easily uncover hidden purchasing patterns in your dataset using advanced Machine Learning.")

uploaded_file = st.file_uploader("Upload your SmartCart CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")
    
    if st.button("Run K-Means Clustering"):
        with st.spinner("Processing Data and Running ML Pipeline..."):
            processed_df = process_data(df)
            
            st.markdown("---")
            st.header("📊 Segmentation Results")
            
            # KPI Cards
            cols = st.columns(4)
            summaries = processed_df.groupby("Segment").mean(numeric_only=True)
            
            for i, segment in enumerate(sorted(processed_df["Segment"].unique())):
                with cols[i % 4]:
                    st.metric(f"{segment} Avg Income", f"${int(summaries.loc[segment, 'Income']):,}")
                    st.metric(f"{segment} Spend", f"${int(summaries.loc[segment, 'Total_Spending']):,}")
            
            st.markdown("---")
            st.subheader("Segment Distribution")
            
            # Pie Chart
            fig_pie = px.pie(processed_df, names='Segment', title="Cluster Size Breakdown", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader("Income vs Spending by Segment")
            # Scatter Plot
            fig_scatter = px.scatter(processed_df, x="Income", y="Total_Spending", color="Segment",
                                     title="Income relative to Total Spending", hover_data=["Age"])
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Processed Data Sample")
            st.dataframe(processed_df.head(50))
