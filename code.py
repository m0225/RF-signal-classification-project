import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

# Streamlit Page Configurations
st.set_page_config(page_title="RF Signal Dashboard", layout="wide")

# Title
st.title("📡 RF Signal Data Dashboard")

# Load Precomputed Pipeline
@st.cache_data
def load_pipeline():
    with open("/home/kartik/Downloads/anomaly_detection_pipeline2.pkl", "rb") as f:
        return pickle.load(f)

pipeline = load_pipeline()
df = pipeline["df"]
features = pipeline["features"]
rf_if = pipeline["rf_if"]
rf_dbscan = pipeline["rf_dbscan"]
rf_ocsvm = pipeline["rf_ocsvm"]

# Ensure Timestamp is in datetime format
df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

# Sidebar Filters
st.sidebar.header("🔍 Filter Data")
modulation_types = st.sidebar.multiselect(
    "Select Modulation Type:",
    df["Modulation"].dropna().unique(),
    default=df["Modulation"].dropna().unique()
)

device_types = st.sidebar.multiselect(
    "Select Device Type:",
    df["Device Type"].dropna().unique(),
    default=df["Device Type"].dropna().unique()
)
st.sidebar.markdown("---")
st.sidebar.header("📌 Select Use Case")
use_case = st.sidebar.radio("Choose a Use Case:",
                                ["Modulation Type Prediction", "Anomaly Detection", "Monitoring System Failure", "Go back to main dashboard"], index=None)
# Filtering Data
df_filtered = df[(df["Modulation"].isin(modulation_types)) | (df["Device Type"].isin(device_types))]
if df_filtered.empty:
    st.warning("⚠️ No data available. Please select at least one Modulation Type or one Device Type from the sidebar.")
else:
    if use_case == "Modulation Type Prediction":
        # --- Display "Model Performance" When Use Case 1 is Selected ---
        st.header("📡 Use Case 1: Modulation Type Prediction")
        st.caption("This module aims to classify the modulation type of RF signals using machine learning.")

        with st.expander("📘 About this Use Case", expanded=False):
            st.markdown("""
            ### 🔧 What is Modulation?
            In wireless communication systems, **modulation** is the process of modifying a carrier signal to encode information for transmission.
             This modification allows the signal to travel efficiently across varying channels and distances. Modulation is fundamental to enabling
              reliable and efficient communication in diverse conditions.

            Different modulation schemes alter the carrier in unique ways to suit specific transmission requirements, environments, and applications.
             Common types of modulation include:
            - **Amplitude Modulation (AM)**  
            - **Frequency Modulation (FM)**  
            - **Phase Shift Keying (PSK)** 
            - **Quadrature Amplitude Modulation (QAM)**
            - **8 Phase Shift Keying (8PSK)**  
            - **Quadrature Phase Shift Keying(PSK)** 

            Accurately identifying the modulation type of a received signal is essential across many fields such as **cognitive radio networks**
             **spectrum monitoring**, **defense communication systems**, and **intelligent signal processing in IoT environments**.

            ### 🤖 Purpose of this Use Case
            This use case aims to build a robust **machine learning system** capable of classifying the modulation type of an RF signal using a
             variety of real-world features. These features are derived from measurements that include **WiFi signal strength**, **system load**,
              **frequency**, and external **environmental conditions** like weather. Together, these reflect the dynamic and noisy nature of real-world 
              communication systems and serve as critical indicators for machine learning models.

            ### 🛠️ Models Used for Classification
            To tackle this classification task, multiple **supervised learning algorithms** have been implemented and compared. The models include:
            - Logistic Regression 📊  
            - Support Vector Machine (SVM) 📐  
            - Decision Tree 🌲  
            - Random Forest 🌳  

            Each model is trained using a labeled dataset where the modulation schemes act as the target labels. After training, the models are
             evaluated using a test set and standard performance metrics such as **accuracy**, **confusion matrix**, **classification report** 
             (covering **precision**, **recall**, and **F1-score**), and the **ROC curve** with **AUC values**.

            ### 🎯 What This Module Offers
            The primary goal of this module is not only to correctly predict the modulation scheme of a signal but also to provide a clear,
             comparative view of each model’s performance. This enables users to gain insights into which model works best under certain conditions
              and why, based on both the dataset characteristics and model evaluation metrics.
            """)

        # Model Selection Dropdown
        model_choice = st.selectbox("Select a Model", ["Logistic Regression", "SVM", "Decision Tree", "Random Forest"])

        # File Paths for Models
        model_files = {
            "Logistic Regression": "/home/kartik/Downloads/loggistic_regression_model.pkl",
            "SVM": "/home/kartik/Downloads/svm_model1.pkl",
            "Decision Tree": "/home/kartik/Downloads/decision_tree_model.pkl",
            "Random Forest": "/home/kartik/Downloads/random_forest_model.pkl",
        }

        # File Paths for Precomputed Predictions (Only for DT & RF)
        prediction_files = {
            "Decision Tree": "/home/kartik/Downloads/dt_predictions.pkl",
            "Random Forest": "/home/kartik/Downloads/rf_predictions.pkl",
        }


        # Function to Load a Model
        def load_model(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)


        # Function to Load Precomputed Predictions
        def load_predictions(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)


        # Load Test Data
        with open("/home/kartik/Downloads/test_data.pkl", "rb") as f:
            X_test, y_test = pickle.load(f)

        # Load Predictions for DT & RF, Predict Directly for SVM & LogReg
        if model_choice in prediction_files:
            y_test, y_pred = load_predictions(prediction_files[model_choice])
        else:
            model = load_model(model_files[model_choice])
            y_pred = model.predict(X_test)

        # Calculate Metrics
        accuracy = accuracy_score(y_test, y_pred) * 100  # Convert to percentage
        class_report_dict = classification_report(y_test, y_pred, output_dict=True)  # Dict format for DataFrame
        conf_matrix = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        class_report_df = pd.DataFrame(class_report_dict).transpose()

        # Center-align all text and numbers using .style and custom CSS
        styled_class_report_html = class_report_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]},
             {'selector': 'td', 'props': [('text-align', 'center')]}]
        ).format(precision=2).to_html()

        # Display Accuracy and Data Split Info (subtle and dashboard-friendly)
        st.subheader("📊 Model Performance Metrics")
        st.write(f"**Accuracy:** {accuracy:.2f}%")
        st.caption("Training Data: 80% | Testing Data: 20%")
        st.markdown("---")
        # Plot ROC Curve using Plotly
        st.subheader("ROC Curve")
        roc_curve_fig = go.Figure()
        roc_curve_fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC Curve (AUC = {roc_auc:.2f})", line=dict(color="blue")))
        roc_curve_fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(color="gray", dash="dash")))

        roc_curve_fig.update_layout(
            xaxis_title="False Positive Rate (FPR)",
            yaxis_title="True Positive Rate (TPR)",
            title=f"ROC Curve for {model_choice}",
            legend=dict(x=0.4, y=0.1),
        )
        st.plotly_chart(roc_curve_fig)

        # Display Classification Report and Explanation Side by Side
        # Classification Report Section
        st.markdown("### 🧪 Model Evaluation Metrics")
        st.markdown("---")

        col1, col2 = st.columns([2, 1.4])
        with col1:
            st.subheader("📋 Classification Report")
            st.markdown(styled_class_report_html, unsafe_allow_html=True)

        with col2:
            st.subheader("🔍 Metric Descriptions")
            st.markdown("""
            **🎯 Precision**  
            Measures how many of the predicted positive instances were actually correct.
            
            **📈 Recall**  
            Indicates how many actual positive instances were captured by the model.
            
            **⚖️ F1-Score**  
            A balanced measure that considers both precision and recall.
            
            **📊 Support**  
            The number of true instances for each class in the test data.
            """)

        # Confusion Matrix Section
        st.markdown("---")

        labels = ['Class 0', 'Class 1']  # Replace with your actual class names
        conf_df = pd.DataFrame(conf_matrix, index=[f'Predicted {label}' for label in labels],
                               columns=[f'Actual {label}' for label in labels])
        styled_html = conf_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]},
             {'selector': 'td', 'props': [('text-align', 'center')]}]
        ).to_html()

        col3, col4 = st.columns([2, 1.4])
        with col3:
            st.subheader("🧮 Confusion Matrix")
            st.markdown(styled_html, unsafe_allow_html=True)

        with col4:
            st.subheader("📘 How to Interpret")
            st.markdown("""
            **Class 0**  
            Represents unmodulated signals (e.g., noise or idle signal).  
            **Class 1**  
            Represents modulated signals (e.g., BPSK, QPSK, etc.).  

            **Matrix Layout**  
            - Top-left: Correct predictions for Class 0  
            - Bottom-right: Correct predictions for Class 1  
            - Top-right / Bottom-left: Misclassifications
            """)
        # ========================== 🔍 Predict Using User Input ==========================
        # ========================== 🔍 Predict Using User Input ==========================
        st.markdown("---")
        st.subheader("🧠 Predict Modulation Type from Custom Input")
        st.caption("Enter signal and environmental features to classify the modulation type using the selected model.")

        # Define input features (based on training)
        input_features = [
            "WiFi Strength", "System Load", "Frequency", "Temperature", "Humidity", "Weather Condition"
        ]

        # Create user input fields
        user_input = {}
        col1, col2 = st.columns(2)

        with col1:
            user_input["WiFi Strength"] = st.number_input("📶 WiFi Strength", value=-40.0, step=1.0)
            user_input["System Load"] = st.number_input("🖥️ System Load", value=55.0, step=1.0)
            user_input["Frequency"] = st.number_input("📡 Frequency (MHz)", value=2400.0, step=1.0)

        with col2:
            user_input["Temperature"] = st.number_input("🌡️ Temperature (°C)", value=25.0, step=1.0)
            user_input["Humidity"] = st.number_input("💧 Humidity (%)", value=50.0, step=1.0)
            user_input["Weather Condition"] = st.selectbox("☁️ Weather Condition", ["Sunny", "Cloudy", "Rainy"])

        # Encode Weather Condition (ensure it matches training)
        weather_map = {"Sunny": 0, "Cloudy": 1, "Rainy": 2}
        user_input["Weather Condition"] = weather_map[user_input["Weather Condition"]]

        # Prepare input for prediction
        input_df = pd.DataFrame([user_input])

        # Predict when the user clicks the button
        if st.button("🔍 Predict Modulation Type"):
            try:
                # Load selected model
                model = load_model(model_files[model_choice])

                # Predict and decode output
                prediction = model.predict(input_df.values)[0]  # .values avoids feature name warning

                # Map prediction to signal type
                signal_type_map = {
                    0: "Analog Signal (AM/FM)",
                    1: "Digital Signal (QPSK, 8PSK, QAM, FSK)"
                }
                signal_type = signal_type_map.get(prediction, f"Unknown Class ({prediction})")

                st.success(f"✅ Predicted Modulation Type: **{signal_type}**")

            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")




    # --- Display "Anomaly Detection" Page When Use Case 2 is Selected ---
    elif use_case == "Anomaly Detection":
        st.markdown("---")
        st.header("🚀 Anomaly Detection - Use Case 2")
        st.caption(
            "🧠 This module detects unusual patterns in system behavior using unsupervised learning, helping preempt system failures or performance drops.")

        with st.expander("📘 About this Use Case", expanded=False):
            st.markdown("""
            ### 🧠 Theory: Anomaly Detection in Wireless Systems

            In dynamic RF environments, system behavior often deviates due to **interference**, **hardware faults**, or **external disruptions**.  
            This use case applies anomaly detection to key RF metrics:

            📡 **Signal Strength**  
            📶 **Frequency**  
            📊 **Bandwidth**  
            ⚙️ **System Load**  
            📡 **WiFi Strength**

            These deviations can reveal **network issues**, **performance bottlenecks**, or **potential security threats**.  
            To capture these anomalies, we use **unsupervised machine learning models** on resampled daily averages:

            🌲 **Isolation Forest** – Separates anomalies using random trees.  
            🔍 **DBSCAN** – Detects clusters and flags ungrouped data as outliers.  
            🤖 **One-Class SVM** – Models the boundary of normal data and highlights unusual behavior.

            ---

            ### 🎯 Purpose of This Use Case

            This module is designed to **proactively detect and visualize abnormal conditions** in RF communication systems.  
            By identifying anomalies in system performance, it supports:

            - ⚠️ Early fault detection  
            - 📉 Reduced downtime  
            - 🔧 Preventive maintenance  
            - 📈 Improved network reliability and operational resilience

            The comparative visualizations and feature importance insights help users understand **why** anomalies occur and how to 
            **respond effectively**.
            """)

        df_copy = df.set_index('Timestamp')
        df_copy_resample = df_copy.resample('D')[
            "Frequency", "Signal Strength", "Bandwidth", "System Load", "WiFi Strength"].mean()
        # Select relevant features
        features = ["Frequency", "Signal Strength", "Bandwidth", "System Load", "WiFi Strength"]
        X = df_copy_resample[features]

        # Normalize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Isolation Forest
        # ============================
        iso_forest = IsolationForest(n_estimators=100, contamination=0.3, random_state=42)
        df_copy_resample["Anomaly_iso_forest"] = iso_forest.fit_predict(X_scaled)
        df_normal_iso = df_copy_resample[df_copy_resample["Anomaly_iso_forest"] == 1]
        df_anomaly_iso = df_copy_resample[df_copy_resample["Anomaly_iso_forest"] == -1]

        fig_iso = go.Figure()
        fig_iso.add_trace(go.Scatter(x=df_normal_iso.index, y=df_normal_iso["System Load"],
                                     mode='lines', name="Normal",
                                     line=dict(color="#00CC96", width=2), opacity=0.8))
        fig_iso.add_trace(go.Scatter(x=df_anomaly_iso.index, y=df_anomaly_iso["System Load"],
                                     mode='markers', name="Anomaly",
                                     marker=dict(color="#FF4B4B", size=6), opacity=1))
        fig_iso.update_layout(title="🌲 Isolation Forest", xaxis_title="Time", yaxis_title="System Load",
                              legend_title="", template="plotly_dark")

        # ============================
        # DBSCAN
        # ============================
        dbscan = DBSCAN(eps=1.5, min_samples=2)
        df_copy_resample["Cluster"] = dbscan.fit_predict(X_scaled)
        df_copy_resample["Anomaly_dbscan"] = df_copy_resample["Cluster"].apply(lambda x: -1 if x == -1 else 1)
        df_normal_db = df_copy_resample[df_copy_resample["Anomaly_dbscan"] == 1]
        df_anomaly_db = df_copy_resample[df_copy_resample["Anomaly_dbscan"] == -1]

        fig_db = go.Figure()
        fig_db.add_trace(go.Scatter(x=df_normal_db.index, y=df_normal_db["System Load"],
                                    mode='lines', name="Normal",
                                    line=dict(color="#636EFA", width=2), opacity=0.8))
        fig_db.add_trace(go.Scatter(x=df_anomaly_db.index, y=df_anomaly_db["System Load"],
                                    mode="markers", name="Anomaly",
                                    marker=dict(size=6, color="#FF4B4B"), opacity=1))
        fig_db.update_layout(title="🔍 DBSCAN", xaxis_title="Time", yaxis_title="System Load",
                             legend_title="", template="plotly_dark")

        # ============================
        # One-Class SVM
        # ============================
        svm = OneClassSVM(kernel='rbf', nu=0.3, gamma='scale')
        df_copy_resample["Anomaly_svm"] = svm.fit_predict(X_scaled)
        df_normal_svm = df_copy_resample[df_copy_resample["Anomaly_svm"] == 1]
        df_anomaly_svm = df_copy_resample[df_copy_resample["Anomaly_svm"] == -1]

        fig_svm = go.Figure()
        fig_svm.add_trace(go.Scatter(x=df_normal_svm.index, y=df_normal_svm["System Load"],
                                     mode='lines', name="Normal",
                                     line=dict(color="#FFA15A", width=2), opacity=0.8))
        fig_svm.add_trace(go.Scatter(x=df_anomaly_svm.index, y=df_anomaly_svm["System Load"],
                                     mode="markers", name="Anomaly",
                                     marker=dict(size=6, color="#FF4B4B"), opacity=1))
        fig_svm.update_layout(title="🤖 One-Class SVM", xaxis_title="Time", yaxis_title="System Load",
                              legend_title="", template="plotly_dark")

        # ============================
        # Display in Streamlit
        # ============================

        st.markdown("""
        <div style='background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center;'>
            <h3>📈 Anomaly Detection Model Comparison</h3>
        </div>
        """, unsafe_allow_html=True)

        # Row 1 - Two Columns
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_iso, use_container_width=True)
        with col2:
            st.plotly_chart(fig_db, use_container_width=True)

        # Row 2 - Centered One
        st.markdown(" ")
        col_center = st.columns([1, 2, 1])  # Centered layout
        with col_center[1]:
            st.plotly_chart(fig_svm, use_container_width=True)

        # Dropdown Above Graphs (Instead of Sidebar)
        selected_model = st.selectbox("📊 Select Model:", ["Isolation Forest", "DBSCAN", "One-Class SVM"], index=None)

        if selected_model:
            # Determine Feature Importance Based on Model Selection
            if selected_model == "Isolation Forest" and hasattr(rf_if, "feature_importances_"):
                feature_importance = rf_if.feature_importances_
            elif selected_model == "DBSCAN" and hasattr(rf_dbscan, "feature_importances_"):
                feature_importance = rf_dbscan.feature_importances_
            elif selected_model == "One-Class SVM" and hasattr(rf_ocsvm, "feature_importances_"):
                feature_importance = rf_ocsvm.feature_importances_
            else:
                feature_importance = np.zeros(len(features))  # Default if not available

            # Create DataFrame for Feature Importance
            feature_importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance}).sort_values(
                by="Importance", ascending=False)

            # Feature Importance Graph
            st.subheader(f"📊 Feature Importance ({selected_model})")
            fig = px.bar(feature_importance_df, x="Importance", y="Feature", orientation="h",
                         text_auto=".2f", color="Importance", color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)



    elif use_case == "Monitoring System Failure":

        st.subheader("Use Case 3: Monitoring System Failure")
        st.caption(
            "📡 This module classifies system failure for improving communication efficiency.")

        # --- Theoretical & Contextual Information ---
        with st.expander("ℹ️ About this Use Case"):
            st.markdown("""
            ### 🧠 Theory Behind Monitoring System Failures

            In real-world IoT or RF systems, **high system load** and **poor WiFi strength** are key indicators of potential or impending system failures.
            - **System Load** reflects the computational burden on a device or system. When this value exceeds typical limits (90th percentile here), the risk of performance degradation or crash increases.
            - **WiFi Strength** indicates network connectivity. A drop below the 10th percentile suggests weak signal or connectivity issues, which can disrupt real-time processing or data transfer.

            By **resampling** data to daily averages and applying **threshold-based anomaly detection**, we can proactively identify critical conditions without complex models.

            ### 📈 3D Plot Interpretation
            - The 3D plot shows time progression (X-axis), WiFi strength (Y-axis), and system load (Z-axis).
            - Points above the system load threshold or below the WiFi threshold are flagged.
            - This allows operational teams to **visually spot problematic trends or days** where the system was stressed.


            ### 🎯 Why This Matters

            - **Preventive Maintenance**: Allows engineers to fix issues before full system failure.
            - **Cost Savings**: Reduces downtime and maintenance costs.
            - **Real-time Monitoring**: Helps in designing alert systems or dashboards that notify operators in advance.


            """, unsafe_allow_html=True)

        # Convert 'Timestamp' to datetime

        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        # Resample to get daily averages

        daily_data = df.resample('D', on='Timestamp')[["WiFi Strength", "System Load"]].mean().reset_index()

        # Calculate thresholds using quantiles

        thresholds = {

            'System Load': daily_data['System Load'].quantile(0.90),  # High Load Threshold

            'WiFi Strength': daily_data['WiFi Strength'].quantile(0.10)  # Low WiFi Strength Threshold

        }

        # Identify exceeding and normal points based on thresholds

        exceeding_points = daily_data[

            (daily_data["System Load"] > thresholds["System Load"]) |

            (daily_data["WiFi Strength"] < thresholds["WiFi Strength"])

            ]

        normal_points = daily_data.drop(exceeding_points.index)

        # Display calculated threshold values

        st.info(f"""

        **Thresholds Applied :**

        - High System Load Threshold (90th percentile): `{thresholds["System Load"]:.2f}`

        - Low WiFi Strength Threshold (10th percentile): `{thresholds["WiFi Strength"]:.2f}`

        - Assumed Critical Weather Condition: `Rainy`

        """)

        # 3D Plot using Plotly

        fig = go.Figure()

        # Normal points

        fig.add_trace(go.Scatter3d(

            x=normal_points["Timestamp"],

            y=normal_points["WiFi Strength"],

            z=normal_points["System Load"],

            mode='markers',

            marker=dict(

                size=6,

                color=normal_points["System Load"],

                colorscale="Viridis",

                opacity=0.8

            ),

            name="Normal Conditions"

        ))

        # Exceeding points

        fig.add_trace(go.Scatter3d(

            x=exceeding_points["Timestamp"],

            y=exceeding_points["WiFi Strength"],

            z=exceeding_points["System Load"],

            mode='markers',

            marker=dict(

                size=6,

                color='red',

                opacity=0.9,

                symbol="x"

            ),

            name="Exceeding Thresholds"

        ))

        fig.update_layout(

            title="3D Visualization of Daily Avg System Load, WiFi Strength, and Time",

            scene=dict(

                xaxis_title="Date",

                yaxis_title="Avg WiFi Strength",

                zaxis_title="Avg System Load",

                xaxis=dict(showgrid=True, zeroline=False),

                yaxis=dict(showgrid=True, zeroline=False),

                zaxis=dict(showgrid=True, zeroline=False),

            ),

            margin=dict(l=0, r=0, b=0, t=40)

        )

        st.plotly_chart(fig, use_container_width=True)


    # --- Display Regular Dashboard If Use Case 2 or 1 or 3 is Not Selected ---
    else:
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.markdown("### 🌦️ Weather Patterns Observed")
            st.write(df_filtered["Weather Condition"].unique())

        with info_col2:
            st.markdown("### 📱 Connected Device Types")
            st.write(df_filtered["Device Type"].unique())

        # KPI Metrics
        st.markdown("---")
        st.markdown("## 📊 System Snapshot: Key Metrics")

        first_row_col1, first_row_col2, first_row_col3 = st.columns(3)
        first_row_col1.metric("🔢 Total Logged Records", f"{len(df_filtered):,}")
        first_row_col2.metric("🧩 Unique Devices", df_filtered["Device Type"].nunique())
        first_row_col3.metric("🎛️ Modulation Types", df_filtered["Modulation"].nunique())

        signal_col1, signal_col2, signal_col3 = st.columns(3)
        avg_signal = np.nan_to_num(df_filtered["Signal Strength"].mean()) if not df_filtered.empty else 0
        median_signal = np.nan_to_num(df_filtered["Signal Strength"].median()) if not df_filtered.empty else 0
        std_signal = np.nan_to_num(df_filtered["Signal Strength"].std()) if not df_filtered.empty else 0

        signal_col1.metric("📡 Avg Signal Strength", round(avg_signal, 2))
        signal_col2.metric("📉 Median Signal", round(median_signal, 2))
        signal_col3.metric("📈 Signal Strength Std Dev", round(std_signal, 2))

        third_row_col1, third_row_col2 = st.columns(2)
        # Safely get most used modulation
        if not df_filtered["Modulation"].mode().empty:
            most_used_modulation = df_filtered["Modulation"].mode()[0]
        else:
            most_used_modulation = "N/A"
        st.markdown("---")
        # Layout for Graphs
        graph_col1, graph_col2 = st.columns(2)

        # Pie Chart - Modulation Distribution
        with graph_col1:
            st.markdown("### 🧬 Modulation Distribution")
            modulation_counts = df_filtered["Modulation"].value_counts().reset_index()
            modulation_counts.columns = ["Modulation", "Count"]
            fig_pie = px.pie(modulation_counts, values="Count", names="Modulation", title="")
            st.plotly_chart(fig_pie, use_container_width=True)

        # Correlation Heatmap
        with graph_col2:
            st.markdown("### 🤝 Feature Relationship Map")
            numeric_df = df_filtered.select_dtypes(include=[np.number]).drop(columns=["Air Pressure"], errors="ignore")
            correlation_matrix = numeric_df.corr()
            fig_heatmap = ff.create_annotated_heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns.tolist(),
                y=correlation_matrix.index.tolist(),
                colorscale="Viridis",
                annotation_text=np.round(correlation_matrix.values, 2)
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

        # Signal Strength & WiFi Strength Over Time
        st.markdown("---")
        st.markdown("## 📶 Signal & WiFi Strength Over Time")

        time_col1, time_col2 = st.columns(2)

        if not df_filtered.empty:
            df_filtered["Date"] = df_filtered["Timestamp"].dt.date

            with time_col1:
                df_grouped_signal = df_filtered.groupby("Date", as_index=False)["Signal Strength"].mean()
                fig_signal = px.line(
                    df_grouped_signal,
                    x="Date",
                    y="Signal Strength",
                    markers=True,
                    title="📱 Signal Strength Trend",
                    labels={"Date": "Date", "Signal Strength": "Signal Strength (dBm)"},
                    line_shape="spline",
                    template="plotly_white",
                )
                st.plotly_chart(fig_signal, use_container_width=True)

            with time_col2:
                if "WiFi Strength" in df_filtered.columns:
                    df_grouped_wifi = df_filtered.groupby("Date", as_index=False)["WiFi Strength"].mean()
                    fig_wifi = px.line(
                        df_grouped_wifi,
                        x="Date",
                        y="WiFi Strength",
                        markers=True,
                        title="📶 WiFi Strength Trend",
                        labels={"Date": "Date", "WiFi Strength": "WiFi Strength (dBm)"},
                        line_shape="spline",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_wifi, use_container_width=True)
                else:
                    st.warning("⚠️ 'WiFi Strength' column not found")
#CHATBOT CODE:
import groq

# Set Groq API Key
os.environ['GROQ_API_KEY'] = "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"  # Replace with your actual key
groq_client = groq.Client(api_key=os.environ['GROQ_API_KEY'])

# Load Dataset Summary
with open("/home/kartik/Documents/Dataset_Summary.txt", "r", encoding="utf-8") as file:
    dataset_info = file.read()

# System Prompt for Groq
system_prompt = f"""
You are an AI assistant specializing in RF signal analysis and system performance evaluation.
Below is the dataset information extracted from a text file. 
Use this information to answer user queries accurately.

DATASET INFORMATION:
{dataset_info}

When a user asks about signal parameters, modulation type, environmental conditions, or system performance, refer to the above dataset and provide relevant details.
Ensure responses are concise and directly answer the user's query.Total no. of records are 164160. Signal Strength Std Dev is 29.19.
Give answers only if related topic is available; otherwise say: "Not relevant to my dataset."
"""

# Ask Groq Function
def ask_groq(question, chat_history):
    messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": question}]
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=messages,
        temperature=0.2,
        max_tokens=300
    )
    return response.choices[0].message.content

show_chatbot = st.sidebar.toggle("💬 Open AI Chatbot")

if show_chatbot:
    st.markdown("---")
    st.subheader("🤖 AI Chatbot")

    # Chat History Init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if user_prompt := st.chat_input("Ask about RF signal dataset..."):
        st.chat_message("user").markdown(user_prompt)
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        with st.spinner("🤖 Thinking..."):
            reply = ask_groq(user_prompt, st.session_state.chat_history)

        st.chat_message("assistant").markdown(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

        