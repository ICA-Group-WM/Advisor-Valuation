import pandas as pd
from datetime import datetime
from fuzzywuzzy import fuzz
import re

# Define a function to preprocess names
def preprocess_name(name):
    return re.sub(r'[^\w\s]', '', name.lower())

advisor_name = 'Change the name here'
current_date = datetime.now().strftime("%m-%d-%Y")
current_year = datetime.now().year
output_file = f"Informal-Value-Exact-{current_date}-{advisor_name}.csv"

# Load data
client_summary_df = pd.read_csv("change file directory here")
revenue_by_client_df = pd.read_csv("change file directory here")
revenue_by_client_df['Group Browse Client Name'] = revenue_by_client_df['Group Browse Client Name'].astype(str)
revenue_by_client_df.drop(revenue_by_client_df.index[-1], inplace=True)
# Exclude rows where 'Client Since' is NaN or 'Client Type' is 'Prospect'
condition_to_keep = (client_summary_df['Client Since'].notna()) & (client_summary_df['Client Type'] != 'Prospect')
filtered_client_summary_df = client_summary_df[condition_to_keep]
# Create dictionaries for mapping preprocessed names back to original names
filtered_client_summary_df = filtered_client_summary_df.copy()
filtered_client_summary_df['Preprocessed Name'] = filtered_client_summary_df['Client Name'].apply(preprocess_name)
revenue_by_client_df['Preprocessed Name'] = revenue_by_client_df['Group Browse Client Name'].apply(preprocess_name)
# List of columns to be summed
def clean_and_convert(value):
    if isinstance(value, str):
        return float(value.replace(',', '').replace('(', '').replace(')', ''))
    return value

# List of columns to be summed
columns_to_sum = ['Total Assets', 'Net Advisory Revenue', 'Net Trails', 'Net Brokerage Commissions', 'Gross Revenue', 'Net Revenue']

# Convert columns to numeric
for col in columns_to_sum:
    revenue_by_client_df[col] = revenue_by_client_df[col].apply(clean_and_convert)

# Group by
grouped_by_name_ssn = revenue_by_client_df.groupby(['Preprocessed Name', 'Client SSN / TIN'], dropna=False)

# Initialize an empty DataFrame to store the processed results
ProcessedRevenueDF = pd.DataFrame()

# Loop through each group and aggregate the information as needed
for name, group in grouped_by_name_ssn:
    if len(group) > 1:
        # Sum only the specific columns
        numeric_sum = group[columns_to_sum].sum()
        
        # Create a dictionary to store both numeric and non-numeric values
        combined_row = {}
        for col in group.columns:
            if col in columns_to_sum:
                combined_row[col] = numeric_sum[col]
            else:
                combined_row[col] = group[col].iloc[0]
        
        # Append combined row to ProcessedRevenueDF
        ProcessedRevenueDF = pd.concat([ProcessedRevenueDF, pd.DataFrame([combined_row])], ignore_index=True)
    else:
        ProcessedRevenueDF = pd.concat([ProcessedRevenueDF, group.reset_index(drop=True)], ignore_index=True)

# Clear and update revenue_by_client_df with the processed rows
revenue_by_client_df.drop(revenue_by_client_df.index, inplace=True)
revenue_by_client_df = pd.concat([revenue_by_client_df, ProcessedRevenueDF], ignore_index=True)

# Fuzzy matching and candidate link generation
most_similar_links = {}  # Dictionary to store most similar links
for idx_right, row_right in revenue_by_client_df.iterrows():
    best_matches = []
    best_score = 80
    for idx_left, row_left in filtered_client_summary_df.iterrows():
        name_similarity = fuzz.token_sort_ratio(row_left['Preprocessed Name'], row_right['Preprocessed Name'])
        if name_similarity >= best_score:
            best_matches.append((idx_left, name_similarity))
    if best_matches:
        # Sort the matches and pick the best one
        best_match = sorted(best_matches, key=lambda x: x[1], reverse=True)[0]
        most_similar_links[idx_right] = best_match
# Assign best matches to each row in revenue_by_client_df
revenue_by_client_df['Mapped Client Name'] = revenue_by_client_df.index.map(lambda x: filtered_client_summary_df.loc[most_similar_links[x][0], 'Client Name'] if x in most_similar_links else None)
# Merge DataFrames
merged_df = revenue_by_client_df.merge(filtered_client_summary_df, left_on='Mapped Client Name', right_on='Client Name', how='left')
# Add original 'Group Browse Client Name' as 'Client/Household Name'
merged_df['Client/Household Name'] = merged_df['Group Browse Client Name']
# Fill NaN values in 'Age' with default 58
merged_df['Age'].fillna(58, inplace=True)
# Calculate 'Years with Firm'
merged_df['Client Since'] = pd.to_datetime(merged_df['Client Since']).dt.year
merged_df['Years with Firm'] = current_year - merged_df['Client Since']

# Calculate the "Recurring Revenue" column directly
merged_df['Recurring Revenue'] = 0
recurring_condition_1 = (merged_df['Net Advisory Revenue'].notnull()) & (merged_df['Net Advisory Revenue'] != 0)
recurring_condition_2 = (merged_df['Net Advisory Revenue'] == 0) & (merged_df['Net Trails'].notnull()) & (merged_df['Net Trails'] != 0)
recurring_condition_3 = (merged_df['Net Advisory Revenue'].notnull()) & (merged_df['Net Advisory Revenue'] != 0) & (merged_df['Net Brokerage Commissions'] != 0) & (merged_df['Net Brokerage Commissions'].notnull())
merged_df.loc[recurring_condition_1 | recurring_condition_2 | recurring_condition_3, 'Recurring Revenue'] = merged_df['Gross Revenue'] - merged_df['Net Brokerage Commissions']
merged_df['Recurring Revenue'].fillna(0, inplace=True)

# Calculate the "Non-Recurring Revenue" column directly
merged_df['Non-Recurring Revenue'] = 0
non_recurring_condition_1 = (merged_df['Net Advisory Revenue'].notnull()) & (merged_df['Net Advisory Revenue'] != 0) & (merged_df['Net Brokerage Commissions'] != 0) & (merged_df['Net Brokerage Commissions'].notnull())
non_recurring_condition_2 = (merged_df['Net Advisory Revenue'] == 0) & (merged_df['Net Trails'] == 0)
non_recurring_condition_3 = (merged_df['Net Advisory Revenue'].notnull()) & (merged_df['Net Advisory Revenue'] != 0) & (merged_df['Net Brokerage Commissions'] == 0)
non_recurring_condition_4 = (merged_df['Net Advisory Revenue'] == 0) & (merged_df['Net Trails'].notnull()) & (merged_df['Net Trails'] != 0)

merged_df.loc[non_recurring_condition_1, 'Non-Recurring Revenue'] = merged_df['Net Brokerage Commissions']
merged_df.loc[non_recurring_condition_2, 'Non-Recurring Revenue'] = merged_df['Gross Revenue']
merged_df.loc[non_recurring_condition_3, 'Non-Recurring Revenue'] = merged_df['Net Brokerage Commissions']
merged_df.loc[non_recurring_condition_4, 'Non-Recurring Revenue'] = merged_df['Net Brokerage Commissions']
# Collect user input for the % of Advisor's Ownership from 0-100
while True:
    try:
        advisor_ownership_percent = float(input("Enter the % of Advisor's Ownership (0-100): "))
        if 0 <= advisor_ownership_percent <= 100:
            break
        else:
            print("Invalid input. Please enter a value between 0 and 100.")
    except ValueError:
        print("Invalid input. Please enter a numerical value.")
merged_df['Advisor Ownership %'] = advisor_ownership_percent
merged_df['Average Age'] = merged_df['Age']

Recurring_Rev_Multiple = 2.8
Percent_Recurring_Rev = 0.01
Non_Recurring_Multiple = 1.5
Percent_Non_Reccuring = 0.99
merged_df['HH AUM'] = merged_df['Total Assets']
# Save the original order
merged_df['original_order'] = range(len(merged_df))
# Sort by 'Client/Household Name', 'HH AUM', and 'Recurring Revenue'
merged_df.sort_values(by=['Client/Household Name', 'HH AUM', 'Recurring Revenue'], ascending=[True, True, True], inplace=True)
# Drop duplicates based on 'Client/Household Name' and 'HH AUM', keep the first 
merged_df.drop_duplicates(subset=['Client/Household Name', 'HH AUM', 'Recurring Revenue', 'Client SSN / TIN'], keep='first', inplace=True)
# Sort back by the original order and drop the 'original_order' column
merged_df.sort_values(by='original_order', inplace=True)
merged_df.drop(columns=['original_order'], inplace=True)
Total_AUM = merged_df['HH AUM'].sum()
AUM_Weighted_Age = merged_df.apply(lambda row: row['Average Age'] * (row['HH AUM'] / Total_AUM) if row['HH AUM'] != 0 else 0, axis=1)
merged_df['AUM Weighted average age'] = AUM_Weighted_Age
sum_AUM_Weighted_Age = merged_df['AUM Weighted average age'].sum()
average_Years_with_Firm = merged_df['Years with Firm'].mean()
average_HH_AUM = merged_df['HH AUM'].mean()
def get_yes_no_input(prompt):
    while True:
        user_input = input(prompt).lower().strip()
        if user_input in ['yes', 'no']:
            return user_input
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
proactive_Service_Model = get_yes_no_input("Risk Adjustment: Proactive Service Model (yes/no): ")
reactive_Service_Model = get_yes_no_input("Risk Adjustment: Reactive Service Model (yes/no): ")
growth_Last_3_Years = get_yes_no_input("Risk Adjustment: <5% YOY Growth last 3 years (yes/no): ")
# Calculate Recurring_rev_Adjusted_multiple
def calculate_recurring_adjusted_multiple():
    adjusted_multiple = Recurring_Rev_Multiple
    if sum_AUM_Weighted_Age >= 65:
        adjusted_multiple -= 0.1
    if sum_AUM_Weighted_Age < 50:
        adjusted_multiple += 0.1
    if average_Years_with_Firm <= 5:
        adjusted_multiple -= 0.2
    if average_HH_AUM < 250000:
        adjusted_multiple -= 0.2
    if proactive_Service_Model == 'yes':
        adjusted_multiple += 0.2
    if reactive_Service_Model == 'yes':
        adjusted_multiple -= 0.2
    if growth_Last_3_Years == 'yes':
        adjusted_multiple -= 0.2
    return adjusted_multiple

# Calculate Non_recurring_adjusted_rev_multiple
def calculate_non_recurring_adjusted_multiple():
    adjusted_multiple = Non_Recurring_Multiple
    if sum_AUM_Weighted_Age >= 65:
        adjusted_multiple -= 0.1
    if sum_AUM_Weighted_Age < 50:
        adjusted_multiple += 0.1
    if average_Years_with_Firm <= 5:
        adjusted_multiple -= 0.2
    if average_HH_AUM < 250000:
        adjusted_multiple -= 0.2
    if proactive_Service_Model == 'yes':
        adjusted_multiple += 0.2
    if reactive_Service_Model == 'yes':
        adjusted_multiple -= 0.2
    if growth_Last_3_Years == 'yes':
        adjusted_multiple -= 0.2
    return adjusted_multiple

Recurring_Rev_Adjusted_Multiple = calculate_recurring_adjusted_multiple()
Non_Recurring_Adjusted_Rev_Multiple = calculate_non_recurring_adjusted_multiple()


def calculate_gross_value(row):
    if row['Non-Recurring Revenue'] != '':
        return ((row['Non-Recurring Revenue'] * (advisor_ownership_percent / 100)) * Non_Recurring_Multiple + (row['Recurring Revenue'] * (advisor_ownership_percent / 100) * Recurring_Rev_Multiple))
    else:
        return ''

def calculate_adjusted_value(row):
    if row['Non-Recurring Revenue'] != '':
        return ((row['Non-Recurring Revenue'] * (advisor_ownership_percent / 100)) * (Non_Recurring_Adjusted_Rev_Multiple) + (row['Recurring Revenue'] * (advisor_ownership_percent / 100) * Recurring_Rev_Adjusted_Multiple))
    else:
        return ''
Gross_value = merged_df.apply(calculate_gross_value, axis=1)
Adjusted_Value = merged_df.apply(calculate_adjusted_value, axis=1)

merged_df['Gross Value'] = Gross_value
merged_df['Adjusted Value'] = Adjusted_Value
# Reorder columns to include new ones
desired_columns = ['Client/Household Name', 'Age', 'Average Age', 'AUM Weighted average age', 'Client Since', 'Years with Firm',
                   'HH AUM', 'Recurring Revenue', 'Non-Recurring Revenue', 'Advisor Ownership %', 'Gross Value', 'Adjusted Value']

merged_df['Gross Value'] = pd.to_numeric(merged_df['Gross Value'], errors='coerce')
merged_df['Recurring Revenue'] = pd.to_numeric(merged_df['Recurring Revenue'], errors='coerce')
merged_df['Non-Recurring Revenue'] = pd.to_numeric(merged_df['Non-Recurring Revenue'], errors='coerce')
merged_df['Net Revenue'] = pd.to_numeric(merged_df['Net Revenue'], errors='coerce')
# Calculate the required values
Total_AUM = pd.to_numeric(merged_df['HH AUM'], errors='coerce').sum()
Total_AUM = '${:,.2f}'.format(Total_AUM)
Estimated_Gross_Value = pd.to_numeric(merged_df['Gross Value'], errors='coerce').sum()
Estimated_Gross_Value = '${:,.2f}'.format(Estimated_Gross_Value)
Estimated_Adjusted_Value = pd.to_numeric(merged_df['Adjusted Value'], errors='coerce').sum()
Estimated_Adjusted_Value = '${:,.2f}'.format(Estimated_Adjusted_Value)
average_HH_AUM = '${:,.2f}'.format(average_HH_AUM)
Brokerage = round(merged_df['Net Brokerage Commissions'].sum() / merged_df['Net Revenue'].sum(), 2)
Advisory_Trails = round((merged_df['Net Trails'].sum() + merged_df['Net Advisory Revenue'].sum())/ merged_df['Net Revenue'].sum(), 2)
Brokerage_Times_NetAdRev = Brokerage * merged_df['Gross Revenue'].sum()
Advisory_Trails_Times_NetAdRev = Advisory_Trails * merged_df['Gross Revenue'].sum()
Total_Revenue = Brokerage_Times_NetAdRev + Advisory_Trails_Times_NetAdRev
Total_Revenue = "${:,.2f}".format(Total_Revenue)
def to_money_format(x):
    return '${:,.2f}'.format(x)

money_columns = ['HH AUM', 'Recurring Revenue', 'Non-Recurring Revenue', 'Gross Value', 'Adjusted Value']
merged_df[money_columns] = merged_df[money_columns].applymap(to_money_format)

Percent_Recurring_Rev = Percent_Recurring_Rev * 100
Percent_Non_Recurring = Percent_Non_Reccuring * 100

# Create a new DataFrame for the summarized values
summary_data = {
    'Client/Household Name': ['Summarized Average Values'],
    'AUM Weighted average age': [sum_AUM_Weighted_Age],
    'Years with Firm': [average_Years_with_Firm],
    'HH AUM': [average_HH_AUM]
}
summary_df = pd.DataFrame(summary_data)

# Append the summarized values as new rows to the original DataFrame
merged_df = pd.concat([merged_df, summary_df], ignore_index=True)
# Create a DataFrame for the calculations
calculation_data = {
    'Client/Household Name': ['Total AUM', 'Total Revenue', 'Informal Gross Valuation', 'Informal Adjusted Valuation', 
                    '% of Revenue Recurring', '% of Revenue Non-Recurring'],
    'Age': [Total_AUM, Total_Revenue, Estimated_Gross_Value, Estimated_Adjusted_Value,
              f'{Percent_Recurring_Rev:.2f}%', f'{Percent_Non_Recurring:.2f}%'],
    'Average Age': ['Recurring Revenue Multiple', 'Non-Recurring Revenue Multiple', 'Recurring Revenue Adjusted Multiple', 'Non-Recurring Revenue Adjusted Multiple'],
    'AUM Weighted average age': [Recurring_Rev_Multiple, Non_Recurring_Multiple, Recurring_Rev_Adjusted_Multiple, Non_Recurring_Adjusted_Rev_Multiple],
    'Client Since': ['Risk Adjustment: Proactive Service Model', 'Risk Adjustment: Reactive Service Model', 'Risk Adjustment: <5% YOY Growth last 3'],
    'Years with Firm': [proactive_Service_Model, reactive_Service_Model, growth_Last_3_Years],
    'HH AUM': ['Brokerage', 'Advisory Trails Revenue', 'Brokerage * Gross Revenue', 'Advisory Trails * Gross Revenue'],
    'Recurring Revenue': [Brokerage, Advisory_Trails, Brokerage_Times_NetAdRev, Advisory_Trails_Times_NetAdRev]
}
# Find the max length
max_length = max(len(value) for value in calculation_data.values())

# Pad the shorter lists
for key, value in calculation_data.items():
    if len(value) < max_length:
        calculation_data[key] = value + [None] * (max_length - len(value))

calculation_df = pd.DataFrame(calculation_data)
merged_df = pd.concat([merged_df, calculation_df], ignore_index=True)

# Select only the desired columns and save to CSV
merged_df[desired_columns].to_csv(output_file, index=False)
print("Data exported to:", output_file)