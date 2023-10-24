import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the models
with open('./model/UTUC_PFS.pkl', 'rb') as f:
    model_pfs = pickle.load(f)

with open('./model/UTUC_CSS.pkl', 'rb') as f:
    model_css = pickle.load(f)

# Streamlit app title
st.title('UTUC Survival Prediction Model')

# Sidebar input fields
st.sidebar.header('Patient Details')
input_params = {
    'Age': st.sidebar.slider('Age', min_value=0, max_value=100, value=50, step=1),
    'Female': st.sidebar.radio('Sex', ['Male', 'Female']) == 'Female',
    'BMI': st.sidebar.slider('BMI', min_value=10.0, max_value=60.0, value=25.0, step=0.5),
    'Pre_GFR': st.sidebar.slider('Pre_GFR', min_value=0.0, max_value=150.0, value=90.0, step=0.5)
}
st.sidebar.header('Disease Details')
input_params.update({
    'Location_Pelvis': st.sidebar.checkbox('Location in Pelvis?'),
    'Location_multiple': st.sidebar.checkbox('Multiple Locations?'),
    'HUN': st.sidebar.checkbox('HUN?'),
    'path_T': st.sidebar.selectbox(
        'Path T stage', 
        ['T0', 'Tis', 'Ta', 'T1', 'T2', 'T3', 'T4'], 
        index=3  
    ),
    'Path_N': st.sidebar.selectbox(
        'Path N stage', 
        ['N0', 'N1', 'Nx'], 
        index=2  
    ),
    'Path_Grade': st.sidebar.radio('Path_Grade', ['Low Grade', 'High Grade']) == 'Path_Grade',
    'Path_CIS': st.sidebar.checkbox('Path CIS?')
})

# Update 'Female' to 'Sex' column and map path_T and Path_N to their numeric values
input_params['Female'] = 1 if input_params.pop('Female') else 0
input_params['path_T'] = {'T0': 0, 'Tis': 1, 'Ta': 2, 'T1': 3, 'T2': 4, 'T3': 5, 'T4': 6}[input_params['path_T']]
input_params['Path_N'] = {'N0': 0, 'N1': 1, 'Nx': 0}[input_params.pop('Path_N')]

ordered_columns = ['Age', 'Female', 'BMI', 'Pre_GFR', 'Location_Pelvis', 'Location_multiple', 'HUN', 'path_T', 'Path_N', 'Path_Grade', 'Path_CIS']
features_df = pd.DataFrame([input_params])[ordered_columns]


# Model prediction and plotting
if st.sidebar.button('Predict'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Years after Surgery')
    ax.set_ylabel('Survival Probability')
    ax.set_xlim([-0.01, 131])
    ax.set_ylim([0.0, 1.01])
    ax.grid(True)
    
    survival_data_pfs, upper_ci_pfs, lower_ci_pfs = model_pfs.predict(features_df, return_ci=True)
    ax.plot(survival_data_pfs.columns, survival_data_pfs.mean(), label="AI predicted PFS survival")
    ax.fill_between(survival_data_pfs.columns, lower_ci_pfs.mean(), upper_ci_pfs.mean(), alpha=0.2)
    
    survival_data_css, upper_ci_css, lower_ci_css = model_css.predict(features_df, return_ci=True)
    ax.plot(survival_data_css.columns, survival_data_css.mean(), label="AI predicted CSS survival")
    ax.fill_between(survival_data_css.columns, lower_ci_css.mean(), upper_ci_css.mean(), alpha=0.2, color='orange')
    
    ax.legend()
    st.pyplot(fig)

 
    # Landmark survival probabilities
    landmarks = [12, 36, 60, 120]
    pfs_probs = survival_data_pfs.mean().loc[landmarks].values
    css_probs = survival_data_css.mean().loc[landmarks].values

    # Display in Streamlit using two columns
    st.subheader('Landmark Survival Probabilities')
    col1, col2 = st.columns(2)  # Change this line

    with col1:
        st.write('**PFS Survival Probability**')
        st.write(f"At 1 year: {pfs_probs[0]:.2%}")
        st.write(f"At 3 years: {pfs_probs[1]:.2%}")
        st.write(f"At 5 years: {pfs_probs[2]:.2%}")
        st.write(f"At 10 years: {pfs_probs[3]:.2%}")

    with col2:
        st.write('**CSS Survival Probability**')
        st.write(f"At 1 year: {css_probs[0]:.2%}")
        st.write(f"At 3 years: {css_probs[1]:.2%}")
        st.write(f"At 5 years: {css_probs[2]:.2%}")
        st.write(f"At 10 years: {css_probs[3]:.2%}")