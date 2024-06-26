import numpy as np
import pandas as pd
from math import ceil
from datetime import datetime, timedelta
import calendar
from sqlalchemy import create_engine
from prophet import Prophet
import inspect
from config import base_path
import os
# import pickle
import logging

#Create and configure logger
logging.basicConfig(filename='item_level_gross_amount_training_model.log', format='%(asctime)s %(levelname)s:%(message)s', filemode='w')
#Creating an object
logger=logging.getLogger()
#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

## import Data  
def import_data():

    category_df = pd.read_excel(f"{base_path}/category_with_ID.xlsx",sheet_name='category')
    category_df = category_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    category_df['CategoryID'] = category_df['CategoryID'].astype(str)
    print('Category data shape',category_df.shape)

    property_data = pd.read_csv(f"{base_path}/filtered_with_feb_with_stand_name.csv",encoding = 'latin1' ,
                    usecols=['Booking Date','Gross Amount','Category','CategoryID','Property name','Item name','stand_name','Cost Center Name','Cost center'])
    property_data.rename(columns={'Booking Date':'Booking_Date','Gross Amount':'Gross_Amount','Property name':'Property_name', 'Item name':'Item_Name',
                                  'Category':'Category','CategoryID':'CategoryID','Cost center':'Cost_center','Cost Center Name':'Cost_Center_Name'}, inplace=True)
    property_data = property_data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    property_data['Item_Name'] = property_data['Item_Name'].str.lower()
    print("consumption data shape:",property_data.shape)

    #Correcting one name in the item name list
    wrong_item = 'roissant, plain, butter, Bridor, eclat du terroir, 30gr, frozen'
    correct_item = 'c' + wrong_item

    property_data.loc[property_data['Item_Name'] == wrong_item,'Item_Name'] = correct_item

    # mapping_df = pd.read_excel(f"{base_path}/item_name_mapping.xlsx",sheet_name='mapping_sheet'
    #                            ,usecols=['Item_Name','Item_Name_Standard','Category_Standard','Remove_flag'])
    mapping_df = pd.read_csv(f"{base_path}/item_name_mapping.csv",encoding = 'latin1'
                                ,usecols=['Item_Name','Item_Name_Standard','Category_Standard','Remove_flag','CategoryID_Standard'])

    mapping_df = mapping_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # mapping_df['Category_Standard'] = mapping_df['Category_Standard'].str.lower()
    mapping_df['Item_Name'] = mapping_df['Item_Name'].str.lower()
    mapping_df['Item_Name_Standard'] = mapping_df['Item_Name_Standard'].str.lower()
    mapping_df_ = mapping_df.drop_duplicates().reset_index(drop=True)
    mapping_df_['Remove_flag'] = mapping_df_['Remove_flag'].astype(str)
    mapping_df_ = mapping_df_[mapping_df_['Remove_flag']=='0']
    print('Mapping data shape',mapping_df_.shape)
    
    merged_property_data_df = pd.merge(property_data, mapping_df_[['Item_Name','Category_Standard','Item_Name_Standard','CategoryID_Standard']], on=['Item_Name'], how='inner')
    merged_property_data_df['CategoryID_Standard'] = merged_property_data_df['CategoryID_Standard'].astype(str)
    #### Get only 26 categories data
    final_data_set = merged_property_data_df[merged_property_data_df['CategoryID_Standard'].isin(category_df.CategoryID.tolist())].reset_index(drop=True)
    print('Final data shape',final_data_set.shape)
    

    return category_df,final_data_set

try:
    category_df,final_data_set = import_data()
    logger.debug("Reading interanl consumption data and mapped with mapping file and consider only 26 categories from DB completed. Total records : %d",final_data_set.shape[0])

except OSError as e:
    logging.error("Reading data from shared folder failed.")


def fill_missing_dates(df, unique_id_col, date_col,max_date_):
    # Convert 'Booking_Date' column to datetime if it's not already
    df[date_col] = pd.to_datetime(df[date_col])

    # max_date = df.Booking_Date.max()
    # Generate a date range from the minimum to the maximum date in your specified range
    #date_range = pd.date_range(start=df.Booking_Date.min(), end=df.Booking_Date.max()+pd.offsets.MonthEnd(0), freq='D')
    date_range = pd.date_range(start=df.Booking_Date.min(), end= max_date_+pd.offsets.MonthEnd(0), freq='D')

    # Create a MultiIndex with all unique IDs and the generated date range
    idx = pd.MultiIndex.from_product([df[unique_id_col].unique(), date_range], names=[unique_id_col, date_col])

    # Reindex the DataFrame with the MultiIndex and fill missing values with zeros
    filled_df = df.set_index([unique_id_col, date_col]).reindex(idx, fill_value=0).reset_index()

    return filled_df


def get_days_week_month_count(df):

    # Add a new column 'Year'
    df['Year'] = df['Booking_Date'].dt.isocalendar().year
    # Add a new column 'Week of Year'
    df['Week_of_Year'] = df['Booking_Date'].dt.isocalendar().week

    df['week_year'] = df['Year'].astype(str) + '-' + df['Week_of_Year'].astype(str)

    # Group the original dataframe by 'Unique ID'
    grouped_df = df.groupby(['uniqueID']).agg(
        no_of_days=('Booking_Date', 'nunique'),  # Count the number of unique booking dates
        no_of_weeks=('week_year', 'nunique'),
        no_of_months=('Booking_Date', lambda x: x.dt.to_period('M').nunique()),
        last_Trx_Date = ('Booking_Date', 'max')).reset_index()
    
    grouped_df['lastest_date'] = grouped_df.last_Trx_Date.max()
    grouped_df['months_difference'] = (grouped_df['lastest_date'].dt.to_period('M') - grouped_df['last_Trx_Date'].dt.to_period('M'))
    grouped_df['months_difference'] = grouped_df['months_difference'].apply(lambda x: x.n)
    
    #### Filter uniqueIDs which has more week numbers
    grouped_df_more_week_df = grouped_df[~((grouped_df['no_of_days'] < 11) | (grouped_df['months_difference'] >= 4))]
    more_week_count_list_ = grouped_df_more_week_df.uniqueID.tolist()
    print('Selected Unique IDs count: ',len(more_week_count_list_))
    print('Selected Unique IDs: ',more_week_count_list_)


    #### Filter uniqueIDs which has less week numbers
    grouped_df_less_week_df = grouped_df[(grouped_df['no_of_days']< 21) | (grouped_df['months_difference']>= 4)]  #### filter less than or equal 10 records and last 4 month does not happend any purchase
    less_count_unique_ids_list = grouped_df_less_week_df.uniqueID.tolist()

    print('Unselected Unique IDs count: ',len(less_count_unique_ids_list))
    print('Unselected Unique IDs: ',less_count_unique_ids_list)
    # dfgs = pd.DataFrame(less_count_unique_ids_list, columns=['unique_id'])
    # dfgs.to_csv('test.csv')

    return more_week_count_list_, less_count_unique_ids_list



def weekly_resampling(df):
    sunday_dates = []
    def week_of_month(dt):
        first_day = dt.replace(day=1)
        dom = dt.day
        adjusted_dom = dom + first_day.weekday()
        return int(ceil(adjusted_dom/7.0))
    
    def calculate_sunday_date(year, month, week_number):
        first_day_of_month = pd.Timestamp(year, month, 1)
        first_day_of_month_weekday = first_day_of_month.dayofweek
        days_to_sunday = 6 - first_day_of_month_weekday
        first_sunday_of_month = first_day_of_month + pd.Timedelta(days=days_to_sunday)
        sunday_date = first_sunday_of_month + pd.Timedelta(weeks=week_number - 1)
        return sunday_date
    
    df['Year'] = df['Booking_Date'].dt.year
    df['Month'] = df['Booking_Date'].dt.month
    df['Week_Number'] = df['Booking_Date'].apply(lambda x: week_of_month(x))
    df_weekly = df.groupby(['uniqueID', 'Year', 'Month', 'Week_Number']).agg({'Gross_Amount': 'sum'}).reset_index()
    grouped = df_weekly.groupby(['uniqueID', 'Year', 'Month'])
    for name, group in grouped:
        last_two_weeks = group.tail(2)
        last_day_of_month = calendar.monthrange(name[1], name[2])[1]
        weekday_of_last_day = calendar.weekday(name[1], name[2], last_day_of_month)
 
        if weekday_of_last_day == 6:
            continue
        else:
            sum_last_two_weeks = last_two_weeks['Gross_Amount'].sum()
            min_week_number = min(last_two_weeks['Week_Number'])
            df_weekly.loc[last_two_weeks.index, 'Gross_Amount'] = sum_last_two_weeks
            df_weekly.loc[last_two_weeks.index, 'Week_Number'] = min_week_number
    df_weekly = df_weekly.drop_duplicates()
    
    for index, row in df_weekly.iterrows():
        sunday_date = calculate_sunday_date(row['Year'], row['Month'], row['Week_Number'])
        sunday_dates.append(sunday_date)
 
    df_weekly['Sunday_Date'] = sunday_dates
    columns_to_select = ['uniqueID', 'Sunday_Date', 'Gross_Amount']
    df_weekly = df_weekly[columns_to_select].copy()
    df_weekly = df_weekly.rename(columns={'Sunday_Date':'Booking_Date'}) 

    #### Select wanted columns only 
    selected_colm = ['uniqueID','Booking_Date','Gross_Amount'] 
    df_weekly = df_weekly[selected_colm]

    return df_weekly


# Handle negative values in forecasted data
def replace_negatives_with_weighted_average(forecast_data):
    """
    Replace weighted average of non-neg values for neg values in fcst data 
    """
    replaced_count = 0
    updated_forecast = forecast_data.copy()
    for i, val in enumerate(forecast_data):
        if val < 0:  # Handle negative values
            closest_non_negative_1 = None
            closest_non_negative_2 = None
            distance_1 = None
            distance_2 = None
            # Find the two closest non-negative numbers
            for j in range(1, min(i+1, len(forecast_data)-i)):
                if forecast_data[i-j] >= 0:
                    closest_non_negative_1 = forecast_data[i-j]
                    distance_1 = j
                if forecast_data[i+j] >= 0:
                    closest_non_negative_2 = forecast_data[i+j]
                    distance_2 = j
                if closest_non_negative_1 is not None and closest_non_negative_2 is not None:
                    break
            # Compute the weights based on the inverse of the distances
            if closest_non_negative_1 is not None and closest_non_negative_2 is not None:
                weight_1 = 1 / (distance_1 + 1)  # Adding 1 to avoid division by zero
                weight_2 = 1 / (distance_2 + 1)  # Adding 1 to avoid division by zero
                total_weight = weight_1 + weight_2
                weighted_avg = (closest_non_negative_1 * weight_1 + closest_non_negative_2 * weight_2) / total_weight
                updated_forecast[i] = weighted_avg
            replaced_count+=1 
    return updated_forecast, replaced_count



####### Weekly to monthly prediction - Prophet model
def prophet_1(df,max_date):
        
        def week_of_month(dt):
                first_day = dt.replace(day=1)
                dom = dt.day
                adjusted_dom = dom + first_day.weekday()
                return int(ceil(adjusted_dom/7.0))

        df = df.copy()
        print(df.Booking_Date.max())
        df['Week_Number'] = df['Booking_Date'].apply(lambda x: week_of_month(x))
        prophet_df = df.rename(columns={'Booking_Date': 'ds', 'Gross_Amount': 'y'})

        model = Prophet(interval_width = 0.8)
        model.fit(prophet_df)
        future = model.make_future_dataframe(freq='W',periods=52)
        forecast = model.predict(future)
        forecast_sel_col = ['ds', 'yhat']
        forecast = forecast[forecast_sel_col]
        forecast.rename(columns={'ds': 'Booking_Date', 'yhat':'predicted'}, inplace=True)
        
        ## Handling the negative values with a specific function
        forecast['predicted'], _ = replace_negatives_with_weighted_average(forecast['predicted'])
        forecast['uniqueID'] = df['uniqueID'].iloc[-1]
        merged_df = pd.merge(df, forecast, on=['uniqueID','Booking_Date'], how ='right')
        merged_df_ = merged_df[['uniqueID','Booking_Date', 'Gross_Amount', 'predicted']]

        ##### weekly Predicted value aggregated into monthly
        prophet_agg_prediction_df = merged_df_.groupby(['uniqueID', pd.Grouper(key='Booking_Date', freq='M')]).sum().reset_index()
        #### Predicted value if negative it will be convert it into zero
        prophet_agg_prediction_df['predicted'] = prophet_agg_prediction_df['predicted'].apply(lambda x: max(0,x))

        #### Actual Data frame
        actual_df = prophet_agg_prediction_df[prophet_agg_prediction_df['Booking_Date']<= max_date]
        sel_col = ['uniqueID','Booking_Date','Gross_Amount']
        actual_df = actual_df[sel_col]

        #### Prediction Data frame
        prediction_df = prophet_agg_prediction_df[prophet_agg_prediction_df['Booking_Date']> max_date]
        sel_col_pre = ['uniqueID','Booking_Date','predicted']
        prediction_df = prediction_df[sel_col_pre]

        return prediction_df, actual_df



def data_preprocessing(df):
    df['Booking_Date'] = pd.to_datetime(df['Booking_Date'])
    max_date = df.Booking_Date.max() +  pd.offsets.MonthEnd(0)
    print('Maximum Date: ',max_date)
    
    ### Create the uniqueID
    try:
        separator = '__'
        columns_to_concat = ['Property_name', 'Cost_Center_Name','Category_Standard', 'Item_Name_Standard']
        df['uniqueID'] = df[columns_to_concat].astype(str).apply(lambda x: x.str.cat(sep=separator), axis=1)
        logger.debug("Sucessfully created UniqueID")
    except OSError as e:
        logging.error("Failed create the uniqueID.")

    #### Sort the data frame
    df = df.sort_values(by=['uniqueID','Booking_Date'])

    #### select only specfic column 
    select_columns = ['Booking_Date','uniqueID','Gross_Amount']
    selected_colm_df = df[select_columns]
    
    #### Groping the dataset by 'uniqueID','Booking_Date'
    try:
        selected_grouped_df = selected_colm_df.groupby(['uniqueID','Booking_Date'])['Gross_Amount'].sum().reset_index()
        logger.debug("Successfully aggregated same-day transactions by unique ID")
    except OSError as e:
        logging.error("Failed aggregated same-day transactions by unique ID.")


    #### Filter only have more week numbers
    try:
        selected_grouped_df1 = selected_grouped_df.copy()
        more_week_count_list_, less_week_count_list_ = get_days_week_month_count(selected_grouped_df1)
        selected_filtered_df = selected_grouped_df[selected_grouped_df['uniqueID'].isin(more_week_count_list_)].reset_index(drop=True)

        sel_unique_ids = selected_filtered_df['uniqueID'].unique()
        print('Selected final uniqueIDs to perform forecast: ',len(sel_unique_ids))
        logger.debug("Successfully filterout if have less week numbers.")
 
        ### Write unselected unique IDs to a text file
        with open('selected_item_level_unique_ids.txt','w')as file:
            for unique_id in sel_unique_ids:
                file.write(f"{unique_id}\n")
        logger.debug("Successfully wrote selected unique IDs to text file.")
    except OSError as e:
        logging.error("Failed filterout if have less week numbers.")

    #### Filter have less week numbers
    try:
        unselected_filtered_df = selected_grouped_df[selected_grouped_df['uniqueID'].isin(less_week_count_list_)].reset_index(drop=True)
        unique_ids = unselected_filtered_df['uniqueID'].unique()
        print('Unselected uniqueIDs counts: ',len(unique_ids))
        logger.debug("Successfully filtered only those with fewer days.")

        ### Write unselected unique IDs to a text file
        with open('unselected_item_level_unique_ids.txt','w')as file:
            for unique_id in unique_ids:
                file.write(f"{unique_id}\n")
        logger.debug("Successfully wrote unique IDs to text file.")
    except OSError as e:
        logging.error("Failed to filter only those with fewer days.")
        logging.error(str(e))
    

    def sub_pre(dff):
        ### Fill missing sequence with zero
        try:
            filled_data = []
            for unique_id in dff['uniqueID'].unique():
                subset = dff[dff['uniqueID'] == unique_id]  ### Get each uniqueIDs in loop
                filled_subset = fill_missing_dates(subset, unique_id_col='uniqueID', date_col='Booking_Date', max_date_ = max_date)
                filled_data.append(filled_subset)
            
            filled_missing_df = pd.concat(filled_data) # Concatenate the filled sequence subsets
            logger.debug("Successfully filled the missing sequence date.")
        except OSError as e:
            logging.error("Failed filled the missing sequence date.")

        try:
                #### Resampled by monthly
            month_resampled_data = []
            for unique_id in filled_missing_df['uniqueID'].unique():
                mon_subset = filled_missing_df[filled_missing_df['uniqueID'] == unique_id]
                monthly_resampled_ = mon_subset.groupby(['uniqueID', pd.Grouper(key='Booking_Date', freq='M')]).sum().reset_index()
                month_resampled_data.append(monthly_resampled_)
            monthly_resampling_df = pd.concat(month_resampled_data) # Concatenate the filled subsets
            logger.debug("Successfully daily dataset has been resampled into monthly.")
        except OSError as e:
            logging.error("Failed daily dataset has been resampled into monthly.")

        ##### Resampled by weekly
        try:
            weekly_filled_data = []
            for unique_id_ in filled_missing_df['uniqueID'].unique():
                week_subset = filled_missing_df[filled_missing_df['uniqueID'] == unique_id_]  ### Get each uniqueIDs in loop
                filled_week_subset = weekly_resampling(week_subset)
                weekly_filled_data.append(filled_week_subset)
            
            weekly_resampling_df = pd.concat(weekly_filled_data) # Concatenate the filled subsets
            logger.debug("Successfully resampled each unique ids in weekly.")
        except OSError as e:
            logging.error("Failed resampled each unique ids in weekly.")

        return filled_missing_df, weekly_resampling_df, monthly_resampling_df

        
    try:
        selected_df_daily,selected_df_weekly,selected_df_monthly = sub_pre(selected_filtered_df)
        logger.debug("Successfully saved selected daily, week and monthly resampled dataset into selected_df_daily,selected_df_weekly,selected_df_monthly variabels.")
    except OSError as e:
            logging.error("Failed saved selected daily, week and monthly resampled dataset into selected_df_daily,selected_df_weekly,selected_df_monthly variabels.")
    # try:        
    #     unselected_df_daily,unselected_df_weekly,unselected_df_monthly = sub_pre(unselected_filtered_df)
    #     logger.debug("Successfully saved unselected daily, week and monthly resampled dataset into selected_df_daily,unselected_df_weekly,selected_df_monthly variabels.")
    # except OSError as e:
    #         logging.error("Failed saved unselected daily, week and monthly resampled dataset into selected_df_daily,unselected_df_weekly,selected_df_monthly variabels.")
    
    #### Resampled by monthly
    # month_resampled_data = []
    # for unique_id in filled_missing_df['uniqueID'].unique():
    #     mon_subset = filled_missing_df[filled_missing_df['uniqueID'] == unique_id]
    #     monthly_resampled_df = mon_subset['Gross_Amount'].resample('M').sum()
    #     # monthly_resampled_df_ = monthly_resampled_df.reset_index(drop=False)
    #     month_resampled_data.append(mon_subset)

    return selected_df_daily,selected_df_weekly,selected_df_monthly, max_date



def run_model_for_all_ids(model, filtered_df, max_date=None,model_name=""):
    try:
        # Initialize empty DataFrames to store the results
        all_prediction_df = pd.DataFrame()
        all_actual_df = pd.DataFrame()

        # List to store unique IDs that fail
        failed_unique_ids = []

        # Check if the model function requires max_date as an argument
        model_args = inspect.signature(model).parameters

        # Iterate over unique IDs for Prophet model
        for unique_id in filtered_df['uniqueID'].unique():
            try:
                logger.debug(f"Processing unique ID: {unique_id}")
                # Filter dataframe for each unique ID
                df_subset = filtered_df[filtered_df['uniqueID'] == unique_id].reset_index(drop=True)
                logger.debug(df_subset.head())
                
                # Apply model function to the subset of data
                if 'max_date' in model_args:
                    prediction_df, actual_df = model(df_subset, max_date)
                else:
                    prediction_df, actual_df = model(df_subset)
                
                logger.debug(f"Prediction DataFrame for {unique_id}: {prediction_df.head()}")
                logger.debug(f"Actual DataFrame for {unique_id}: {actual_df.head()}")
                
                # Append the results to the main result DataFrames
                # all_prediction_df = all_prediction_df.append(prediction_df, ignore_index=True)
                all_prediction_df = pd.concat([all_prediction_df,prediction_df], ignore_index=True)
                # all_actual_df = all_actual_df.append(actual_df, ignore_index=True)
                all_actual_df = pd.concat([all_actual_df,actual_df], ignore_index=True)

            except Exception as e:
                logging.error(f"Failed to process unique ID {unique_id}: {e}")
                failed_unique_ids.append(unique_id)

        # Write failing unique IDs to a text file
        if failed_unique_ids:
            logger.debug(f"Writing failed unique IDs to failed_unique_ids_item_level.txt")
            # Check if file exists, and open in append mode if it does
            with open('failed_unique_ids_supplier_category.txt', 'a' if os.path.exists('failed_unique_ids_item_level.txt') else 'w') as f:
                for unique_id in failed_unique_ids:
                    f.write(f"{unique_id}, {model_name}\n")
            logger.debug(f"Failed unique IDs written to failed_unique_ids_item_level.txt")

        logger.debug("Successfully saved both actual data and prophet prediction results as two separate dataframes.")
        return all_prediction_df, all_actual_df

    except Exception as e:
        logging.error(f"Failed to save both actual data and prophet prediction results as two separate dataframes: {e}")
        return pd.DataFrame(), pd.DataFrame()
    

def main():
    selected_df_daily,selected_df_weekly,selected_df_monthly, max_date  = data_preprocessing(final_data_set)
    try:
        prop1_all_prediction_df, prop1_all_actual_df = run_model_for_all_ids(prophet_1, selected_df_weekly,max_date,model_name="prophet_model_1")
        logger.debug("Successfully saved both actual data and prophet 1 prediction results as two separate dataframes.")
        print("prophet model 1: ",prop1_all_prediction_df.uniqueID.nunique())

    except OSError as e:
        logging.error("Failed to save both actual data and prophet 1 prediction results as two separate dataframes.")


    # List of dataframes to be concatenated
    actual_dataframes = [prop1_all_actual_df]
    predict_dataframes = [prop1_all_prediction_df]
    # Concatenating the dataframes
    final_actual_dataframe = pd.concat(actual_dataframes, ignore_index=True)
    final_actual_dataframe[['Property_name', 'Cost_Center_Name','Category_Standard', 'Item_Name_Standard']] = final_actual_dataframe['uniqueID'].str.split('__', expand=True)

    final_predict_dataframe = pd.concat(predict_dataframes, ignore_index=True)
    final_predict_dataframe[['Property_name', 'Cost_Center_Name','Category_Standard', 'Item_Name_Standard']] = final_predict_dataframe['uniqueID'].str.split('__', expand=True)




if __name__ =='__main__':
    main()





