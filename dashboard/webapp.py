
def streamlit_main():
    """
    streamlit main 함수

    :return: None
    """
    st.title('CCPP Power Output Predictor')

		...

    # sidebar input 값 선택 UI 생성
    st.sidebar.header('User Menu')
    user_input_data = get_user_input_features()

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()

    submit = st.sidebar.button('Get predictions')
    if submit:
        results = requests.post(url + predict_endpoint, json=raw_input_data)
        results = json.loads(results.text)

        # 예측 결과 표시
        st.subheader('Results')
        prediction = results["prediction"]
        st.write("Prediction: ", round(prediction, 2))

        # expander 형식으로 model input 표시
        st.subheader('Input Features')
        features_selected = ['AT', 'V', 'AP', 'RH']

        model_input_expander = st.beta_expander('Model Input')
        model_input_expander.write('Input Features: ')
        model_input_expander.text(", ".join(list(raw_input_data[0].keys())))
        model_input_expander.json(raw_input_data[0])
        model_input_expander.write('Selected Features: ')
        model_input_expander.text(", ".join(features_selected))
        selected_features_values = OrderedDict((k, results[k]) for k in features_selected)
        model_input_expander.json(selected_features_values)

        # shap 값 계산
        shap_results = requests.post(url + shap_endpoint, json=raw_input_data)
        shap_results = json.loads(shap_results.text)

        base_value = shap_results['base_value']
        shap_values = np.array(shap_results['shap_values'])

        # shap force plot 표시
        st.subheader('Interpretation Plot')
        draw_shap_plot(base_value, shap_values, pd.DataFrame(raw_input_data)[features_selected])

        ...

        # expander 형식으로 shap detail 값 표시
        shap_detail_expander = st.beta_expander('Shap Detail')
        for key, item in zip(features_selected, shap_values):
            shap_detail_expander.text('%s: %s' % (key, item))

if __name__ == '__main__':
    streamlit_main()