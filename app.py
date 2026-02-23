import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

# Title
st.title("🧠 AI Procrastination Pattern Analyzer")

# Load dataset
df = pd.read_csv("dataset.csv")

# Clean data
df = df.fillna("Stress")

# Create delay column
df["Delay"] = df["Actual_Time"] - df["Planned_Time"]

# Train AI model
features = df[["Planned_Time", "Actual_Time", "Delay"]]
kmeans = KMeans(n_clusters=3, random_state=42)
df["Procrastination_Type"] = kmeans.fit_predict(features)

# Show dataset preview
st.subheader("📊 Dataset Preview")
st.write(df.head())

# Show average delay
st.metric("⏱️ Average Delay (minutes)", round(df["Delay"].mean(), 2))

# Chart 1 — Delay distribution
st.subheader("📉 Delay Distribution")
st.bar_chart(df["Delay"])

# Chart 2 — Distraction frequency
st.subheader("📱 Top Distractions")
st.bar_chart(df["Distraction"].value_counts())

# Show AI clusters
st.subheader("🤖 AI Clustering Result")
st.write(df[["Planned_Time", "Actual_Time", "Delay", "Procrastination_Type"]].head())

st.subheader("🧪 Test Your Own Procrastination Pattern")

planned = st.number_input("Enter Planned Time (minutes)", min_value=0, value=60)
actual = st.number_input("Enter Actual Time (minutes)", min_value=0, value=90)

if st.button("🔍 Analyze My Behavior"):
    delay = actual - planned

    # Prepare input for AI
    user_data = [[planned, actual, delay]]
    prediction = kmeans.predict(user_data)[0]

    # Convert AI label to human text
    if prediction == 0:
        label = "🔥 High Procrastinator"
        explanation = "You frequently postpone important tasks and struggle to start on time. Try using small goals, deadlines, and distraction-free environments to improve productivity."

    elif prediction == 1:
        label = "⚖️ Moderate Procrastinator"
        explanation = "You occasionally delay tasks, especially when motivation is low. Building better routines and time management habits can help you stay consistent."

    else:
        label = "🚀 Low Procrastinator"
        explanation = "You usually manage tasks efficiently and avoid unnecessary delays. Maintaining your current planning and focus strategies will help sustain this productivity."

    st.markdown("### 🤖 AI Result")
    st.success(label)

    st.markdown(f"**⏱️ Delay:** {delay} minutes")
    st.info(explanation)