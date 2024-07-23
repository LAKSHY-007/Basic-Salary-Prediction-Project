import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
import numpy as np



# Title and introduction
st.title('Salary Predictor App')
st.markdown("""
Welcome to the Salary Predictor App. Navigate through the sidebar to use the app's features:
- **Home**: Overview and instructions.
- **Prediction**: Predict salaries based on input features.
- **Contribute**: Add data to improve our model.
""")



# Sidebar navigation
nav = st.sidebar.radio('Navigation', ['Home', 'Prediction', 'Contribute'])

# Home section
if nav == 'Home':
    st.header('Home')
    st.image('Data//pngtree.jpg', use_column_width=True)
    st.write("""
    This app uses machine learning to predict salaries based on various factors. You can:
    - Predict salaries using the Prediction section.
    - Contribute your data to help improve the model in the Contribute section.
    """)
    
    data = pd.read_csv('Data//salary_data.csv')
    x = np.array(data['Years of Experience']).reshape(-1, 1)
    lr = LinearRegression()

    lr.fit(x, np.array(data['Salary']))
    if st.checkbox('Show Table', True):
         st.table(data)

    graph = st.selectbox('What kind of graph', ['Non-interactive', 'interactive'])

    val = st.slider('Filter by Years of Experience', min_value=0, max_value=25)
    data = data.loc[data['Years of Experience'] >= val]


    if graph == 'Non-interactive':

        
        plt.figure(figsize=(10, 5))
        plt.scatter(data['Years of Experience'], data['Salary'], color='blue')
        plt.ylim(0)
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.tight_layout()
        st.pyplot()

    if graph == 'interactive':
        layout =go.Layout(xaxis=dict(range=[0,15]), yaxis=dict(range=[0,220000]))

        fig = go.Figure(data=[go.Scatter(x=data['Years of Experience'], y=data['Salary'], mode='markers', marker=dict(color='blue'))], layout=layout)
        st.plotly_chart(fig)




# Model prediction
elif nav == 'Prediction':
    st.header('Salary Prediction')
    st.write('Please enter the details to predict the salary.')

    # Load the dataset
    file_path = 'Data//salary_data.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f'Error: The file {file_path} was not found.')
        data = pd.DataFrame(columns=['Years of Experience', 'Salary'])
    except Exception as e:
        st.error(f'An error occurred while loading the data: {str(e)}')
        data = pd.DataFrame(columns=['Years of Experience', 'Salary'])

    # Input fields for prediction
    experience = st.number_input(
        'Years of Experience:',
        min_value=0.0,
        max_value=50.0,
        step=0.1,
        format="%.1f",
        help="Specify the number of years of experience for which you want to predict the salary."
    )
    education_level = st.selectbox('Education Level', ['High School', 'Bachelor', 'Master', 'PhD'])
    location = st.selectbox('Location', ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Others'])

    # Button to trigger salary prediction
    if st.button('Predict Salary'):
        try:
            # Check if data has at least one entry
            if data.empty:
                st.error('The dataset is empty. Cannot train the model.')
            else:
                # Prepare the input data for prediction
                x = np.array(data['Years of Experience']).reshape(-1, 1)
                y = np.array(data['Salary'])
                lr = LinearRegression()
                lr.fit(x, y)

                # Predict the salary
                experience_reshaped = np.array(experience).reshape(-1, 1)
                predicted_salary = lr.predict(experience_reshaped)[0]

                # Display the prediction result
                st.success(f'The predicted salary for {experience:.1f} years of experience is ${predicted_salary:,.2f}.')
        except Exception as e:
            st.error(f'An error occurred while predicting the salary: {str(e)}')

    # Placeholder for an alternative prediction example
    st.write('Alternatively, use this static prediction model for demonstration:')
    if st.button('Predict Salary (Static Model)'):
        try:
            # Example static prediction logic
            predicted_salary_static = 50000 + (experience * 2000)
            st.success(f'The predicted salary using the static model is ${predicted_salary_static:,.2f}.')
        except Exception as e:
            st.error(f'An error occurred with the static model prediction: {str(e)}')


# Contribute section
elif nav == 'Contribute':
    st.header('Contribute Data')
    st.write('Help us improve the model by contributing your data.')

    # Example input fields for contribution (customize as needed)
    name = st.text_input('Name')
    email = st.text_input('Email')
    experience_contrib = st.number_input('Years of Experience (Contribute)', min_value=0, max_value=50, step=1)
    education_level_contrib = st.selectbox('Education Level (Contribute)', ['High School', 'Bachelor', 'Master', 'PhD'])
    location_contrib = st.selectbox('Location (Contribute)', ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Others'])
    salary = st.number_input('Current Salary', min_value=0, step=1000)

    if st.button('Submit Data'):
         new_data = pd.DataFrame({
              'Name': [name],
              'Email': [email],
              'Years of Experience': [experience_contrib],
              'Education Level': [education_level_contrib],
              'Location': [location_contrib],
              'Salary': [salary]
         })
         file_path = 'Data//salary_data.csv'

         try:
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
            updated_data.to_csv(file_path, index=False)
            st.success('Thank you for your contribution!')
         except Exception as e:
            st.error(f'Error occurred: {e}')

# Footer
st.markdown("""
---
*Powered by L.P.*
""")




