import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Data Sweeper", page_icon="🧹", layout="wide")

# Sidebar for options
st.sidebar.title("Data Sweeper Options")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])

# Main area
st.title("🧹 Data Sweeper")
st.write("Upload your dataset and let Data Sweeper clean it for you!")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Dataset")
    st.write(df)

    # Sidebar options for data cleaning
    st.sidebar.subheader("Data Cleaning Options")
    
    # Drop missing values
    if st.sidebar.checkbox("Drop missing values"):
        df = df.dropna()
        st.success("Missing values dropped!")

    # Fill missing values
    if st.sidebar.checkbox("Fill missing values"):
        fill_value = st.sidebar.text_input("Enter fill value", "0")
        df = df.fillna(fill_value)
        st.success(f"Missing values filled with {fill_value}!")

    # Remove duplicates
    if st.sidebar.checkbox("Remove duplicates"):
        df = df.drop_duplicates()
        st.success("Duplicates removed!")

    # Display cleaned data
    st.subheader("Cleaned Dataset")
    st.write(df)

    # Sidebar options for data visualization
    st.sidebar.subheader("Data Visualization Options")

    # Histogram
    if st.sidebar.checkbox("Show Histogram"):
        column = st.sidebar.selectbox("Select column for histogram", df.columns)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[column], kde=True)
        st.pyplot(plt)

    # Scatter plot
    if st.sidebar.checkbox("Show Scatter Plot"):
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=df[x_axis], y=df[y_axis])
        st.pyplot(plt)

    # Correlation matrix
    if st.sidebar.checkbox("Show Correlation Matrix"):
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)

    # Suggestions for further processing
    st.sidebar.subheader("Suggestions for Further Processing")

    # Feature engineering
    if st.sidebar.checkbox("Suggest Feature Engineering"):
        st.info("Consider creating new features based on existing ones, such as ratios, differences, or interactions.")

    # Normalization
    if st.sidebar.checkbox("Suggest Normalization"):
        st.info("Normalize numerical features to bring them to a similar scale.")

    # Encoding categorical variables
    if st.sidebar.checkbox("Suggest Encoding Categorical Variables"):
        st.info("Encode categorical variables using one-hot encoding or label encoding for better model performance.")

else:
    st.info("Please upload a CSV file to get started.")
