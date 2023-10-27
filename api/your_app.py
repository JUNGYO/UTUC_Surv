import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Check if models are already loaded into session state
if 'models' not in st.session_state:
    st.session_state.models = {
        'pfs': load_model('./model/UTUC_PFS.pkl'),
        'css': load_model('./model/UTUC_CSS.pkl')
    }

models = st.session_state.models
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
    'HUN': st.sidebar.checkbox('Presence of Hydronephrosis?'),
    'path_T': st.sidebar.selectbox('Path T stage', ['T0', 'Tis', 'Ta', 'T1', 'T2', 'T3', 'T4'], index=3),
    'Path_N': st.sidebar.selectbox('Path N stage', ['N0', 'N1', 'Nx'], index=2),
    'Path_Grade': st.sidebar.radio('Path_Grade', ['Low Grade', 'High Grade']) == 'High Grade',
    'Path_CIS': st.sidebar.checkbox('CIS on your pathologic specimen?')
})

# Transform inputs
input_params['Female'] = 1 if input_params.pop('Female') else 0
input_params['path_T'] = {'T0': 0, 'Tis': 1, 'Ta': 2, 'T1': 3, 'T2': 4, 'T3': 5, 'T4': 6}[input_params['path_T']]
input_params['Path_N'] = {'N0': 0, 'N1': 1, 'Nx': 0}[input_params.pop('Path_N')]

ordered_columns = ['Age', 'Female', 'BMI', 'Pre_GFR', 'Location_Pelvis', 'Location_multiple', 'HUN', 'path_T', 'Path_N', 'Path_Grade', 'Path_CIS']
features_df = pd.DataFrame([input_params])[ordered_columns]


def make_predictions_and_plot(features_df, years=5):
    pfs_color = '#1f77b4'  # A soft blue color
    css_color = '#d62728'  # A soft red color
    pfs_ci_color = 'rgba(31, 119, 180, 0.001)'  # Light blue with transparency
    css_ci_color = 'rgba(214, 39, 40, 0.001)'  # Light red with transparency


    # Create a new Plotly figure
    fig = go.Figure()

    survival_data_pfs, upper_ci_pfs, lower_ci_pfs = models['pfs'].predict(features_df, return_ci=True)
    fig.add_trace(go.Scatter(x=survival_data_pfs.columns, 
                             y=survival_data_pfs.mean(),
                             mode='lines',
                             name="AI predicted your PFS",
                             line=dict(color=pfs_color),
                             fill=None))
    fig.add_trace(go.Scatter(x=survival_data_pfs.columns, 
                             y=upper_ci_pfs.mean(),
                             mode='lines',
                             showlegend=False,
                             line=dict(color=pfs_ci_color, width=0.5)))
    fig.add_trace(go.Scatter(x=survival_data_pfs.columns,
                             y=lower_ci_pfs.mean(),
                             mode='lines',
                             showlegend=False,
                             fill='tonexty',
                             line=dict(color=pfs_ci_color, width=0.5)))

    survival_data_css, upper_ci_css, lower_ci_css = models['css'].predict(features_df, return_ci=True)
    fig.add_trace(go.Scatter(x=survival_data_css.columns, 
                             y=survival_data_css.mean(),
                             mode='lines',
                             name="AI predicted your OS",
                             line=dict(color=css_color),
                             fill=None))
    fig.add_trace(go.Scatter(x=survival_data_css.columns, 
                             y=upper_ci_css.mean(),
                             mode='lines',
                             showlegend=False,
                             line=dict(color=css_ci_color, width=0.5)))
    fig.add_trace(go.Scatter(x=survival_data_css.columns,
                             y=lower_ci_css.mean(),
                             mode='lines',
                             showlegend=False,
                             fill='tonexty',
                             line=dict(color=css_ci_color, width=0.5)))

    # Set layout for the plot
    fig.update_layout(title='Survival Prediction Plot',
                      xaxis_title='Months after Surgery',
                      yaxis_title='Survival Probability',
                      yaxis=dict(range=[0, 1.05]))

    landmarks = [12, 36, 60, 120]
    pfs_probs = survival_data_pfs.mean().loc[landmarks].values
    css_probs = survival_data_css.mean().loc[landmarks].values

    return fig, pfs_probs, css_probs




# If 'Predict' button is clicked, display the generated plot and probabilities
if st.sidebar.button('Predict'):
    plot_fig, pfs_probs, css_probs = make_predictions_and_plot(features_df)
    st.plotly_chart(plot_fig)  # Use this to display Plotly figure in Streamlit

    # Landmark survival probabilities
    st.subheader('Landmark Survival Probabilities')
    col1, col2 = st.columns(2)

    with col1:
        st.write('**Progression free Survival Probability**')
        st.write(f"At 1 year: {pfs_probs[0]:.2%}")
        st.write(f"At 3 years: {pfs_probs[1]:.2%}")
        st.write(f"At 5 years: {pfs_probs[2]:.2%}")
        st.write(f"At 10 years: {pfs_probs[3]:.2%}")

    with col2:
        st.write('**Overall Survival Probability**')
        st.write(f"At 1 year: {css_probs[0]:.2%}")
        st.write(f"At 3 years: {css_probs[1]:.2%}")
        st.write(f"At 5 years: {css_probs[2]:.2%}")
        st.write(f"At 10 years: {css_probs[3]:.2%}")
