import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import st_folium
from matplotlib import cm, colors, colorbar
import matplotlib.pyplot as plt
from io import BytesIO

# def coff_reg_making():
#     data = pd.read_csv("out.csv")
#     # Print initial data info
#     print("Initial data info:")
#     print(data.info())
#     # Remove specified columns
#     columns_to_remove = ['year_of_data','name', 'district', 'Municipal_status', 'Average_monthly_salary_of_employee', 'Average_monthly_income_of_the_self-employed']
#     data = data.drop(columns=columns_to_remove)
#     # Convert all columns to numeric, replacing ',' and forcing errors to NaN
#     data = data.apply(lambda x: pd.to_numeric(x.astype(str).str.replace(',', ''), errors='coerce'))
#     # Print data info after conversion
#     data = data.fillna(data.median())
#     # Define features and target
#     features = data.drop(columns=['Weighted_Average_Salary'])
#     target = data['Weighted_Average_Salary']
#     # Split the data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
#     # Train a Linear Regression model
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     # Make predictions
#     predictions = model.predict(X_test)
#     # Evaluate the model
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
#     print("Mean Absolute Error:", mae)
#     print("R2 Score:", r2)
#     # Get the coefficients
#     coefficients = model.coef_
#     intercept = model.intercept_
#     feature_names = features.columns
#     # Create a DataFrame for the coefficients
#     coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
#     coefficients_df.loc[len(coefficients_df)] = ['Intercept', intercept]
#     # Save coefficients to CSV
#     coefficients_df.to_csv('regression_coefficients.csv', index=False)
#     # Plot the coefficients
#     plt.figure(figsize=(12, 8))
#     sns.barplot(x='Coefficient', y='Feature', data=coefficients_df)
#     plt.title('Regression Coefficients')
#     plt.show()
#     print("Regression coefficients saved to 'regression_coefficients.csv'")


######################################


# Define feature importance sections


important_features = [
    'Socioeconomic_cluster', 'Distance_from_Tel Aviv _District border_Km', 'Commercial_Offices_Percentage',
    'Percentage_degree_35-55', 'Age_65+', 'Residence_Percentage', 'Number_of_self-employed',
    'Cluster_Most_peripheral', 'crops Percentage', 'Age_0-4'
]
less_important_features = [
    'Jews_percent_of_Jews_and_others', 'Age_75+', 'Age_30-44', 'Infrastructure_transportation_percentage',
    'Age_45-59', 'Population_density', 'Health_Percentage_of_Area', 'Agricultural_structures_percentage',
    'Area', 'Age_20-29', 'Public_utilities_Percentage', 'Age_60-64', 'Number_of_families_with_child',
    'Percentage_of_students_20-25', 'Total population', 'Age_10-14'
]

not_important_features = [
    'Age_5-9', 'Gardening_park_Percentage', 'Total_men', 'Total_woman', 'Age_15-19', 'Number_of_employees',
    'Plant_Percentage', 'Percentage_of_total_students', 'Percentage_of_population_growth_compared_to_the_previous_year',
    'Culture_percentage', 'Industry_Percentage', 'migration_balance', 'Start_of_construction_apartments',
    'Number_of_councillors', 'Age_0-17', 'Arab_percentages', 'Jews_and_others', 'Open_space_percentage',
    'Muslims_percent_of_the_Arab_population', 'Education_Percentage',
    'Christians_percent_of_the_Arab_population', 'Forest_percent', 'Druze_percentage_of_the_Arab_population',
    'Percentage_of_higher_education_within_8_years_out_of_school'
]

st.set_page_config(layout="wide")
# Load the coefficients and intercept from the CSV file
coefficients_df = pd.read_csv('regression_coefficients.csv')
coefficients_dict = dict(zip(coefficients_df['Feature'], coefficients_df['Coefficient']))

intercept = coefficients_dict['Intercept']
del coefficients_dict['Intercept']
feature_names = important_features + less_important_features + not_important_features

# Get min and max values for features
min_max_df = pd.read_csv('min_max_values_adjusted.csv')

data = pd.read_csv('out.csv')

def calculate_salary(values):
    salary = intercept
    for feature, value in zip(feature_names, values):
        salary += coefficients_dict[feature] * value
    return salary

def income_analysis():
    st.header("Section 1: Income Analysis")
    st.markdown("""
    This section allows users to explore how different socio-economic and demographic factors impact the predicted average monthly salary in various cities. The predictions are based on a linear regression model with an R² value of 0.85, indicating a strong relationship between the selected features and the target variable. The model uses the most important features, sorted by feature importance, to calculate the predicted salary.

    **How to Use:**

    1. **Select City and Year**: Choose a city and year from the dropdown menus to analyze specific data.
    2. **Select Features**: Use the dropdown menu to select the features you want to display sliders for. The features are sorted by their importance in the model, allowing you to focus on the most impactful factors.
    3. **Adjust Sliders**: Modify the sliders to see how changes in various factors (e.g., socio-economic status, population) impact the predicted average monthly salary.
    4. **View Predicted Salary**: The displayed salary updates dynamically as you adjust the sliders, showing the potential effects of each change.
    """)

    # Add dropdowns to select city and year
    city = st.selectbox('Select City', data['name'].unique())
    year = st.selectbox('Select Year', data['year_of_data'].unique())

    selected_row = data[(data['name'] == city) & (data['year_of_data'] == year)]

    if selected_row.empty:
        st.error("No data available for the selected city and year.")
        return

    # Get all feature values for the selected city and year
    selected_values = selected_row.iloc[0][feature_names].values

    # Ensure selected values are numeric
    selected_values = pd.to_numeric(selected_values, errors='coerce')
    selected_values = np.nan_to_num(selected_values, nan=0.0)

    # Create a copy of selected_values to modify with sliders
    updated_values = selected_values.copy()

    salary_placeholder = st.empty()

    # Multi-select to choose which features to display sliders for
    selected_features = st.multiselect(
        "Select Features to Display",
        feature_names,
        default=feature_names[:5]  # Default selection
    )

    num_cols = 4  # Number of columns to spread the sliders across
    columns = st.columns(num_cols)

    # Iterate over selected features to display sliders in the order they are selected
    # Iterate over selected features to display sliders in the order they are selected
    for i, feature in enumerate(selected_features):
        col = columns[i % num_cols]
        with col:
            min_value = min_max_df[min_max_df['Feature'] == feature]['Min'].values[0]
            max_value = min_max_df[min_max_df['Feature'] == feature]['Max'].values[0]

            # Get the index of the current feature from the original list
            feature_index = feature_names.index(feature)
            slider_value = st.slider(f"{feature}", min_value=float(min_value), max_value=float(max_value),
                                     value=float(selected_values[feature_index]))
            updated_values[feature_index] = slider_value  # Update with slider value

    # Calculate the predicted salary using all feature values
    predicted_salary = calculate_salary(updated_values)

    # Update the predicted salary placeholder
    salary_placeholder.markdown(
        f"<div class='fixed-header'><p style='font-size: 36px;'>Predicted Average Monthly Salary: {predicted_salary :.2f}</p></div>",
        unsafe_allow_html=True
    )


#######################################
def demographic_analysis():
    st.header("Section 2: Demographic Analysis")

    st.markdown("""
        This section provides an interactive way to explore demographic attributes across different cities and years. It helps visualize trends and distributions of various demographic metrics such as age groups, education levels, and employment statistics. The analysis is based on historical data, allowing users to identify patterns and correlations that may influence socio-economic outcomes.

        **How to Use:**

        1. **Select Attribute and Year**: Use the dropdown menus to choose a specific demographic attribute and year you want to analyze.
        2. **Bar Chart and Line Plot**:
           - The **bar chart** shows the distribution of the selected attribute across different cities for the chosen year.
           - The **line plot** displays the changes in the selected attribute over the years for a specific city.
        3. **Interact with the Plots**: Hover over the plots for detailed data points, and use the zoom and pan tools to explore specific areas of interest.

        """)

    data = pd.read_csv(r"out_modified.csv")
    columns_to_remove = ['district', 'Municipal_status', 'Distance_from_Tel Aviv _District border_Km',
                         'Residence_Percentage', 'Health_Percentage_of_Area', 'Education_Percentage',
                         'Public_utilities_Percentage',
                         'Culture_percentage', 'Commercial_Offices_Percentage', 'Industry_Percentage',
                         'Infrastructure_transportation_percentage',
                         'Agricultural_structures_percentage', 'Gardening_park_Percentage', 'Forest_percent',
                         'Plant_Percentage', 'crops_Percentage', 'Open_space_percentage']
    data = data.drop(columns=columns_to_remove, errors='ignore')

    # Convert 'Total_men' and 'Total_woman' columns to numeric, forcing errors to NaN
    data['Total_men'] = pd.to_numeric(data['Total_men'], errors='coerce')
    data['Total_woman'] = pd.to_numeric(data['Total_woman'], errors='coerce')

    # Get the columns dynamically
    columns = data.columns.tolist()
    non_feature_columns = ['year_of_data', 'name']
    columns = [col for col in columns if col not in non_feature_columns]

    # Select attribute and year (now on the main page above the plots)
    attribute = st.selectbox("Select Attribute", columns)
    year = st.selectbox("Select Year", sorted(data['year_of_data'].unique()))

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
            bar_data = bar_data.sort_values(by=attribute,
                                            ascending=False)  # Sort by attribute value in descending order
            fig = px.bar(bar_data, x='name', y=attribute, title=f"{attribute} by City in {year}")
            fig.update_layout(xaxis_title='City Name', yaxis_title=attribute,
                              xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, use_container_width=True)
        elif pd.api.types.is_numeric_dtype(filtered_data[attribute]):
            # Bar chart for numeric attribute
            st.subheader(f"Bar Chart for {attribute} in {year}")
            bar_data = filtered_data.groupby('name')[attribute].sum().reset_index()
            bar_data = bar_data.sort_values(by=attribute,
                                            ascending=False)  # Sort by attribute value in descending order
            fig = px.bar(bar_data, x='name', y=attribute, title=f"{attribute} by City in {year}")
            fig.update_layout(xaxis_title='City Name', yaxis_title=attribute,
                              xaxis={'categoryorder': 'total descending'})
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
            fig.update_layout(barmode='stack', xaxis={'categoryorder': 'total descending'}, xaxis_title='City Name',
                              yaxis_title='Count')
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

#######################################
def geographical_analysis():
    st.header("Section 3: Geographical Analysis")

    st.markdown("""
    This section allows you to visually compare the geographical distribution of various area usage percentages across different cities and years. It provides insights into how different types of land use, such as residential, commercial, and agricultural areas, are distributed within selected cities. Additionally, it helps to compare the average weighted salaries alongside the land usage distribution for a more comprehensive understanding of the geographical and economic landscape.

    **How to Use:**

    1. **Select Cities and Years**: Choose two cities and respective years using the dropdown menus below each pie chart. This selection will display the area usage distribution and average weighted salary for each city and year.
    2. **Pie Charts**: The **pie charts** visually represent the proportion of different land usage types within each city for the selected year. Each segment color corresponds to a specific area usage type.
    3. **Bar Plot**: Below the pie charts, select a percentage attribute and year to view a **bar plot** comparing that attribute's distribution across all cities.
    4. **Interact with the Plots**: Hover over the charts to view more detailed information and use the interactive tools to explore the data more deeply.

        """)

    # Load data
    data = pd.read_csv("out.csv")

    # Define area usage columns
    area_usage_columns = [
        'Residence_Percentage', 'Open_space_percentage', 'Forest_percent',
        'Infrastructure_transportation_percentage', 'Education_Percentage',
        'Public_utilities_Percentage', 'Commercial_Offices_Percentage', 'Plant_Percentage',
        'Culture_percentage', 'Industry_Percentage', 'Health_Percentage_of_Area',
        'Agricultural_structures_percentage', 'Gardening_park_Percentage', 'crops Percentage'
    ]

    # Define a specific color palette matching the provided order
    color_palette = {
        'Residence_Percentage': '#1f77b4',  # Blue
        'Open_space_percentage': '#aec7e8',  # Light blue
        'Forest_percent': '#ff7f0e',  # Orange
        'Infrastructure_transportation_percentage': '#ffbb78',  # Light orange
        'Education_Percentage': '#2ca02c',  # Green
        'Public_utilities_Percentage': '#98df8a',  # Light green
        'Commercial_Offices_Percentage': '#d62728',  # Red
        'Plant_Percentage': '#ff9896',  # Light red
        'Culture_percentage': '#9467bd',  # Purple
        'Industry_Percentage': '#c5b0d5',  # Light purple
        'Health_Percentage_of_Area': '#8c564b',  # Brown
        'Agricultural_structures_percentage': '#c49c94',  # Light brown
        'Gardening_park_Percentage': '#e377c2',  # Pink
        'crops Percentage': '#f7b6d2'  # Light pink
    }

    # Display pie charts side by side
    col1, col2 = st.columns(2)

    with col1:
        # City and year selection for the first pie chart
        city1 = st.selectbox("Select City 1", data['name'].unique(), key='city1')
        year1 = st.selectbox("Select Year for City 1", sorted(data['year_of_data'].unique()), key='year1')

        # Filter data for the selected city and year
        city1_data = data[(data['name'] == city1) & (data['year_of_data'] == year1)]

        if city1_data.empty:
            st.error("Data for City 1 is not available.")
        else:
            # Calculate the mean area usage for city 1
            city1_usage = city1_data[area_usage_columns].mean()
            # Calculate the average weighted salary for city 1
            avg_weighted_salary1 = city1_data['Weighted_Average_Salary'].mean()
            # Plot pie chart for area usage of city 1 with salary in the title
            fig1 = px.pie(
                values=city1_usage,
                names=area_usage_columns,
                title=f"Area Usage Distribution for {city1} in {year1} (Avg Salary: {avg_weighted_salary1:.2f})",
                color=area_usage_columns,
                color_discrete_map=color_palette
            )
            fig1.update_layout(
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # City and year selection for the second pie chart
        city2 = st.selectbox("Select City 2", data['name'].unique(), index=1, key='city2')
        year2 = st.selectbox("Select Year for City 2", sorted(data['year_of_data'].unique()), key='year2')

        # Filter data for the selected city and year
        city2_data = data[(data['name'] == city2) & (data['year_of_data'] == year2)]

        if city2_data.empty:
            st.error("Data for City 2 is not available.")
        else:
            # Calculate the mean area usage for city 2
            city2_usage = city2_data[area_usage_columns].mean()
            # Calculate the average weighted salary for city 2
            avg_weighted_salary2 = city2_data['Weighted_Average_Salary'].mean()
            # Plot pie chart for area usage of city 2 with salary in the title
            fig2 = px.pie(
                values=city2_usage,
                names=area_usage_columns,
                title=f"Area Usage Distribution for {city2} in {year2} (Avg Salary: {avg_weighted_salary2:.2f})",
                color=area_usage_columns,
                color_discrete_map=color_palette
            )
            fig2.update_layout(
                margin=dict(t=40, b=20)
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.write("Compare the area usage distribution and average salary between two selected cities and years.")

    # Add bar plot for selected attribute across all cities
    st.subheader("Compare Percentage Attribute Across Cities")

    # Select attribute and year for bar plot
    selected_attribute = st.selectbox("Select Percentage Attribute for Bar Plot", area_usage_columns, key='selected_attribute')
    selected_year = st.selectbox("Select Year for Bar Plot", sorted(data['year_of_data'].unique()), key='selected_year')

    # Filter data for the selected year
    year_data = data[data['year_of_data'] == selected_year]

    # Check if data is available for the selected year
    if year_data.empty:
        st.error("Data for the selected year is not available.")
    else:
        # Create bar plot for the selected attribute across all cities
        fig_bar = px.bar(
            year_data,
            x='name',
            y=selected_attribute,
            title=f"{selected_attribute} Across Cities in {selected_year}",
        )
        fig_bar.update_layout(xaxis_title='City', yaxis_title=selected_attribute)
        fig_bar.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_bar, use_container_width=True)


#######################################
def distance_analysis():
    st.header("Section 4: Distance Analysis")

    st.markdown("""
       This section provides an interactive geographical visualization that focuses on the spatial distribution of cities relative to Tel Aviv, alongside economic metrics such as the weighted average salary. By examining the distance from Tel Aviv and the corresponding average salaries, users can identify spatial economic patterns and understand the impact of geographical location on economic indicators.

       **How to Use:**

       1. **Select Year**: Choose a specific year from the dropdown menu to analyze the spatial distribution and salary data for that year.
       2. **Map Visualization**: The map displays circles for each city, where:
          - **Interactive Features**: Click on a city to view detailed information such as the exact weighted average salary and its distance from Tel Aviv.
       3. **Color Bar**: The vertical color bar on the right provides a legend for interpreting the color coding of the cities based on their weighted average salary.
       """)

    # Load data with coordinates
    data_with_coords = pd.read_csv('out_with_coordinates.csv', encoding='utf-8')

    # Load the GeoJSON file
    with open('cities.geojson', encoding='utf-8') as f:
        geojson_data = json.load(f)

    # Function to filter data by year
    def filter_data_by_year(df, year):
        return df[df['year_of_data'] == year]

    # Function to create the map and color bar
    def create_map_and_colorbar(data, year):
        filtered_data = filter_data_by_year(data, year).copy()

        # Convert 'Weighted_Average_Salary' to numeric, handling strings with commas
        if filtered_data['Weighted_Average_Salary'].dtype == 'object':
            filtered_data['Weighted_Average_Salary'] = pd.to_numeric(
                filtered_data['Weighted_Average_Salary'].str.replace(',', ''), errors='coerce')
        else:
            filtered_data['Weighted_Average_Salary'] = pd.to_numeric(
                filtered_data['Weighted_Average_Salary'], errors='coerce')

        # Drop rows with NaN values in Weighted_Average_Salary
        filtered_data = filtered_data.dropna(subset=['Weighted_Average_Salary'])

        if filtered_data.empty:
            st.error("No data available for the specified year.")
            return None, None

        # Normalize the Weighted_Average_Salary values to the 0-1 range
        norm = colors.Normalize(vmin=filtered_data['Weighted_Average_Salary'].min(),
                                vmax=filtered_data['Weighted_Average_Salary'].max())

        # Create a colormap with 5 colors
        cmap = cm.get_cmap('YlOrRd', 5)

        # Initialize the map centered on Tel Aviv
        m = folium.Map(location=[32.0853, 34.7818], zoom_start=8)

        # Add city markers with color based on the Weighted_Average_Salary
        for _, row in filtered_data.iterrows():
            if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
                salary_norm = norm(row['Weighted_Average_Salary'])
                color = colors.rgb2hex(cmap(salary_norm)[:3])  # Convert to hex color

                folium.CircleMarker(
                    location=[row['Latitude'], row['Longitude']],
                    radius=7,  # Adjust size as needed
                    popup=folium.Popup(f"City: {row['name']}<br>"
                                       f"Weighted Average Salary: {row['Weighted_Average_Salary']}<br>"
                                       f"Distance from Tel Aviv: {row['Distance_from_Tel Aviv _District border_Km']} km",
                                       max_width=300),
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)

        # Create the vertical color bar
        fig, ax = plt.subplots(figsize=(1.5, 6))  # Adjust size to be narrower and taller
        cbar = colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
        cbar.set_label('Weighted Average Salary')
        plt.tight_layout(pad=0.1)  # Adjust padding around color bar

        # Save color bar to BytesIO object
        img = BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', pad_inches=0.1)  # Small padding to align the color bar
        plt.close()
        img.seek(0)

        return m, img

    # Streamlit UI for year selection
    year_to_display = st.selectbox('Select Year', data_with_coords['year_of_data'].unique())

    # Create and display the map and color bar
    map_obj, colorbar_img = create_map_and_colorbar(data_with_coords, year_to_display)
    if map_obj:
        # Use st_folium to embed the map
        col1, col2 = st.columns([3, 1])  # 3/4 map, 1/4 color bar

        with col1:
            st_folium(map_obj, width=800, height=600)

        with col2:
            st.image(colorbar_img, use_column_width=False, width=100)  # Adjust width as needed
            st.markdown("### Color Bar")



#######################################
def main():
    st.title("Urban Income Explorer")

    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Section 1: Income Analysis", "Section 2: Demographic Analysis",
                                         "Section 3: Geographical Analysis", "Section 4: Distance Analysis"])

    if section == "Section 1: Income Analysis":
        income_analysis()
    elif section == "Section 2: Demographic Analysis":
        demographic_analysis()
    elif section == "Section 3: Geographical Analysis":
        geographical_analysis()
    elif section == "Section 4: Distance Analysis":
        distance_analysis()



if __name__ == '__main__':
    main()
