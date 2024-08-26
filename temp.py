import pandas as pd


def find_min_max_values(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Drop non-numeric columns
    data = data.drop(columns=['name', 'district', 'Municipal_status', 'Average_monthly_salary_of_employee', 'Average_monthly_income_of_the_self-employed'])

    # Ensure all remaining columns are numeric
    data = data.apply(pd.to_numeric, errors='coerce')

    # Define percentage columns
    percentage_columns = [
        'Jews_percent_of_Jews_and_others', 'Arab_percentages', 'Muslims_percent_of_the_Arab_population',
        'Christians_percent_of_the_Arab_population', 'Druze_percentage_of_the_Arab_population',
        'Percentage_of_population_growth_compared_to_the_previous_year', 'Percentage_degree_35-55',
        'Percentage_of_higher_education_within_8_years_out_of_school', 'Percentage_of_students_20-25',
        'Percentage_of_total_students', 'Residence_Percentage', 'Education_Percentage',
        'Health_Percentage_of_Area', 'Public_utilities_Percentage', 'Culture_percentage',
        'Commercial_Offices_Percentage', 'Industry_Percentage', 'Infrastructure_transportation_percentage',
        'Agricultural_structures_percentage', 'Gardening_park_Percentage', 'Forest_percent',
        'Plant_Percentage', 'crops Percentage', 'Open_space_percentage'
    ]

    # Get the min and max values for each column
    min_values = data.min()
    max_values = data.max()

    # Adjust min and max values for percentage columns
    for col in percentage_columns:
        if col in data.columns:
            min_values[col] = 0
            max_values[col] = 100

    # Create a DataFrame to hold the min and max values
    min_max_df = pd.DataFrame({'Feature': data.columns, 'Min': min_values, 'Max': max_values})

    # Save the min and max values to a CSV file
    min_max_df.to_csv('min_max_values_adjusted.csv', index=False)

    return min_max_df

# Call the function and print the result
min_max_df = find_min_max_values('out.csv')