import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('columns.pkl', 'rb') as f:
            columns = pickle.load(f)
        return model, columns
    except:
        return None, None

model, model_columns = load_model()


st.title("🏠 House Price Prediction System")
st.markdown("### Upload your dataset and get instant predictions with insights 📊")

st.sidebar.header("📌 Instructions")
st.sidebar.info("""
• Upload CSV file  
• Must match training dataset  
• Supports categorical data  
• Download predictions  
""")


st.subheader("📥 Download Sample Format")

sample_df = pd.DataFrame({
    "area": [1000],
    "bedrooms": [2],
    "bathrooms": [1],
    "stories": [1],
    "mainroad": ["yes"],
    "guestroom": ["no"],
    "basement": ["no"],
    "hotwaterheating": ["no"],
    "airconditioning": ["yes"],
    "parking": [1],
    "prefarea": ["yes"],
    "furnishingstatus": ["furnished"]
})

st.download_button(
    "Download Sample CSV",
    sample_df.to_csv(index=False),
    "sample.csv",
    "text/csv"
)


uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if model is None:
    st.error("❌ Model not found. Run train.py first.")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        
        if df.isnull().sum().sum() > 0:
            st.warning("⚠️ Missing values detected → filling with 0")
            df = df.fillna(0)

        
        df_encoded = pd.get_dummies(df)

        # Align columns
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

       
        predictions = model.predict(df_encoded)
        df['PredictedPrice'] = predictions

        st.subheader("✅ Prediction Results")
        st.dataframe(df.head())

       
        st.subheader("📊 Key Insights")

        col1, col2, col3 = st.columns(3)

        col1.metric("💰 Avg Price", f"{df['PredictedPrice'].mean():,.0f}")
        col2.metric("📈 Max Price", f"{df['PredictedPrice'].max():,.0f}")
        col3.metric("📉 Min Price", f"{df['PredictedPrice'].min():,.0f}")

        
        st.subheader("📊 Visual Insights")

        c1, c2 = st.columns(2)

        with c1:
            fig1, ax1 = plt.subplots()
            ax1.hist(df['PredictedPrice'])
            ax1.set_title("Price Distribution")
            st.pyplot(fig1)

        with c2:
            fig2, ax2 = plt.subplots()
            ax2.plot(df['PredictedPrice'])
            ax2.set_title("Prediction Trend")
            st.pyplot(fig2)

        
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

        st.success("🎉 Predictions generated successfully!")

    except Exception as e:
        st.error(f"❌ Error: {e}")