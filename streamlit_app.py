"""
Streamlit web application for Titanic survival prediction
Python translation of Shiny app.R
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

@st.cache_data
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('models/titanic_model.pkl')
        feature_names = joblib.load('models/titanic_model_features.pkl')
        return model, feature_names
    except FileNotFoundError:
        st.error("Model not found. Please run model training first.")
        return None, None

def preprocess_single_input(pclass, sex, age, sibsp, parch, fare, embarked, title):
    """
    Preprocess single input for prediction
    """
    # Create dataframe
    data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex], 
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked],
        'title': [title]
    })
    
    # Apply same encoding as training
    # Convert categorical to numeric (simplified version)
    pclass_map = {'1st': 0, '2nd': 1, '3rd': 2}
    sex_map = {'male': 1, 'female': 0}
    embarked_map = {'S': 2, 'C': 0, 'Q': 1}
    
    # Common titles mapping
    title_map = {
        'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3,
        'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Major': 8,
        'Other': 9
    }
    
    data['pclass'] = data['pclass'].map(pclass_map)
    data['sex'] = data['sex'].map(sex_map)
    data['embarked'] = data['embarked'].map(embarked_map)
    data['title'] = data['title'].map(title_map)
    
    return data

def main():
    st.title("🚢 Titanic Survival Predictor")
    st.markdown("---")
    
    # Load model
    model, feature_names = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar for input method selection
    st.sidebar.header("Input Method")
    input_method = st.sidebar.radio(
        "Choose input method:",
        ["Manual Input", "Upload CSV File"]
    )
    
    if input_method == "Manual Input":
        st.header("Single Passenger Prediction")
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Passenger Information")
            
            pclass = st.selectbox(
                "Passenger Class:",
                options=['1st', '2nd', '3rd'],
                index=0
            )
            
            sex = st.selectbox(
                "Sex:",
                options=['male', 'female'],
                index=0
            )
            
            age = st.number_input(
                "Age:",
                min_value=0,
                max_value=100,
                value=30,
                step=1
            )
            
            sibsp = st.number_input(
                "Number of Siblings/Spouses:",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )
        
        with col2:
            st.subheader("Additional Details")
            
            parch = st.number_input(
                "Number of Parents/Children:",
                min_value=0,
                max_value=10,
                value=0,
                step=1
            )
            
            fare = st.number_input(
                "Fare:",
                min_value=0.0,
                value=32.0,
                step=1.0
            )
            
            embarked = st.selectbox(
                "Port of Embarkation:",
                options=['S', 'C', 'Q'],
                format_func=lambda x: {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[x],
                index=0
            )
            
            title = st.selectbox(
                "Title:",
                options=['Mr', 'Mrs', 'Miss', 'Master', 'Dr', 'Rev', 'Col', 'Other'],
                index=0
            )
        
        # Prediction button
        if st.button("🔮 Predict Survival", type="primary"):
            # Preprocess input
            input_data = preprocess_single_input(pclass, sex, age, sibsp, parch, fare, embarked, title)
            
            # Make prediction
            try:
                prediction_proba = model.predict_proba(input_data)[0]
                prediction_class = model.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.header("Prediction Results")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    # Survival prediction
                    if prediction_class == 1:
                        st.success("🎉 **SURVIVED**")
                        survival_text = "Survived"
                        color = "green"
                    else:
                        st.error("💀 **DID NOT SURVIVE**")
                        survival_text = "Did not survive"
                        color = "red"
                    
                    # Probability
                    survival_prob = prediction_proba[1] * 100
                    st.metric("Survival Probability", f"{survival_prob:.1f}%")
                    
                    # Probability bar chart
                    fig = go.Figure(go.Bar(
                        x=['Did not survive', 'Survived'],
                        y=[prediction_proba[0], prediction_proba[1]],
                        marker_color=['red', 'green']
                    ))
                    fig.update_layout(
                        title="Survival Probability",
                        yaxis_title="Probability",
                        showlegend=False,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Passenger details summary
                st.subheader("Passenger Details Summary")
                details_df = pd.DataFrame({
                    'Attribute': ['Class', 'Sex', 'Age', 'Title', 'Siblings/Spouses', 'Parents/Children', 'Fare', 'Embarked'],
                    'Value': [pclass, sex.title(), age, title, sibsp, parch, f'${fare:.2f}', 
                             {'S': 'Southampton', 'C': 'Cherbourg', 'Q': 'Queenstown'}[embarked]]
                })
                st.dataframe(details_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    else:  # Upload CSV File
        st.header("Batch Predictions")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV should contain columns: pclass, sex, age, sibsp, parch, fare, embarked, title"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                data = pd.read_csv(uploaded_file)
                
                # Display data preview
                st.subheader(f"Data Preview ({len(data)} rows)")
                st.dataframe(data.head(10), use_container_width=True)
                
                # Validate required columns
                required_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked', 'title']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                else:
                    if st.button("🚀 Predict All", type="primary"):
                        # Process batch predictions
                        with st.spinner("Making predictions..."):
                            # Preprocess data (simplified for demo)
                            processed_data = data.copy()
                            
                            # Apply mappings
                            pclass_map = {'1st': 0, '2nd': 1, '3rd': 2}
                            sex_map = {'male': 1, 'female': 0}
                            embarked_map = {'S': 2, 'C': 0, 'Q': 1}
                            title_map = {
                                'Mr': 0, 'Mrs': 1, 'Miss': 2, 'Master': 3,
                                'Dr': 4, 'Rev': 5, 'Col': 6, 'Mlle': 7, 'Major': 8,
                                'Other': 9
                            }
                            
                            processed_data['pclass'] = processed_data['pclass'].map(pclass_map)
                            processed_data['sex'] = processed_data['sex'].map(sex_map)
                            processed_data['embarked'] = processed_data['embarked'].map(embarked_map)
                            processed_data['title'] = processed_data['title'].map(title_map)
                            
                            # Fill missing values
                            processed_data = processed_data.fillna(0)
                            
                            # Make predictions
                            predictions_proba = model.predict_proba(processed_data)
                            predictions_class = model.predict(processed_data)
                            
                            # Add predictions to original data
                            results = data.copy()
                            results['Predicted_Survival'] = ['Survived' if p == 1 else 'Did not survive' for p in predictions_class]
                            results['Survival_Probability'] = [f"{p[1]*100:.1f}%" for p in predictions_proba]
                            
                            # Display results
                            st.subheader("Batch Prediction Results")
                            st.dataframe(results, use_container_width=True)
                            
                            # Summary statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Passengers", len(results))
                            with col2:
                                survived_count = sum(predictions_class)
                                st.metric("Predicted Survivors", survived_count)
                            with col3:
                                survival_rate = survived_count / len(results) * 100
                                st.metric("Survival Rate", f"{survival_rate:.1f}%")
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name="titanic_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()