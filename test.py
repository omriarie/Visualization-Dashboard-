import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Load data
data = pd.read_csv(r"C:\Users\97254\OneDrive\תקיית העבודות של ליאל ועמרי\סימסטר ו\וויזואליזציה\liel\out_modified.csv")
columns_to_remove = ['district', 'Municipal_status', 'Distance_from_Tel Aviv _District border_Km',
                     'Residence_Percentage', 'Health_Percentage_of_Area', 'Education_Percentage', 'Public_utilities_Percentage',
                     'Culture_percentage', 'Commercial_Offices_Percentage', 'Industry_Percentage', 'Infrastructure_transportation_percentage',
                     'Agricultural_structures_percentage', 'Gardening_park_Percentage', 'Forest_percent',
                     'Plant_Percentage', 'crops_Percentage', 'Open_space_percentage']
data = data.drop(columns=columns_to_remove, errors='ignore')

# Convert 'Total_men' and 'Total_woman' columns to numeric, forcing errors to NaN
data['Total_men'] = pd.to_numeric(data['Total_men'], errors='coerce')
data['Total_woman'] = pd.to_numeric(data['Total_woman'], errors='coerce')

# Sidebar for filters
st.sidebar.title("Filters")

# Get the columns dynamically
columns = data.columns.tolist()
non_feature_columns = ['year_of_data', 'name']
columns = [col for col in columns if col not in non_feature_columns]

# Select attribute and year
attribute = st.sidebar.selectbox("Select Attribute", columns)
year = st.sidebar.slider("Select Year", int(data['year_of_data'].min()), int(data['year_of_data'].max()))

# Filter data
filtered_data = data[data['year_of_data'] == year]

# Create two columns
col1, col2 = st.columns(2)

# First graph: Bar chart
with col1:
    if attribute in ['Average_monthly_income_of_the_self-employed', 'Average_monthly_salary_of_employee',
                     'Number_of_employees', 'Number_of_self-employed', 'Number_of_families_with_child']:
        # Bar chart with city names on x-axis and sorted by value
        st.subheader(f"Bar Chart for {attribute} in {year}")
        bar_data = filtered_data.groupby('name')[attribute].sum().reset_index()
        bar_data = bar_data.sort_values(by=attribute, ascending=False)  # Sort by attribute value in descending order
        fig = px.bar(bar_data, x='name', y=attribute, title=f"{attribute} by City in {year}")
        fig.update_layout(xaxis_title='City Name', yaxis_title=attribute, xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    elif pd.api.types.is_numeric_dtype(filtered_data[attribute]):
        # Bar chart for numeric attribute
        st.subheader(f"Bar Chart for {attribute} in {year}")
        bar_data = filtered_data.groupby('name')[attribute].sum().reset_index()
        bar_data = bar_data.sort_values(by=attribute, ascending=False)  # Sort by attribute value in descending order
        fig = px.bar(bar_data, x='name', y=attribute, title=f"{attribute} by City in {year}")
        fig.update_layout(xaxis_title='City Name', yaxis_title=attribute, xaxis={'categoryorder': 'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Bar chart for categorical attribute
        st.subheader(f"Bar Chart for {attribute} in {year}")
        bar_data = filtered_data.groupby('name')[attribute].value_counts().unstack().fillna(0).reset_index()
        fig = go.Figure()
        for column in bar_data.columns[1:]:
            fig.add_trace(go.Bar(
                x=bar_data['name'],
                y=bar_data[column],
                name=column
            ))
        fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'}, xaxis_title='City Name', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

# Second graph: Line chart for distribution over years
with col2:
    st.subheader(f"Distribution of {attribute} Over Years for Specific City")
    name = st.selectbox("Select City Name", data['name'].unique())
    specific_data = data[data['name'] == name]

    # Ensure chronological order by sorting years
    line_data = specific_data.groupby('year_of_data')[attribute].sum().reset_index()
    line_data = line_data.sort_values(by='year_of_data')  # Sort by year in ascending order

    fig = px.line(line_data, x='year_of_data', y=attribute, title=f"{attribute} Over Years for {name}")
    fig.update_layout(xaxis_title='Year', yaxis_title=attribute)
    st.plotly_chart(fig, use_container_width=True)














