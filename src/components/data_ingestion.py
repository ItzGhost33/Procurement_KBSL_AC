import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# from src.componenets.data_transformation import DataTransformation
# from src.componenets.data_transformation import DataTransformationConfig
# from src.componenets.model_trainer import ModelTrainerConfig
# from src.componenets.model_trainer import ModelTrainer




@dataclass  # if we only need to define variables in a class we can use this else we need to use __init__
class DataIngestionConfig:
    final_data_path:str=os.path.join('artifacts','final_data_set.csv')  # Difining the path where outputs needs to be stored
    # mapping_sheet_path:str=os.path.join('artifacts','item_name_mapping.csv')
    # category_path:str=os.path.join('artifacts','category_with_ID.csv')



class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv(f'Level_4/data/filtered_with_feb_with_stand_name.csv',encoding = 'latin1' ,
                    usecols=['Booking Date','Gross Amount','Category','CategoryID','Property name','Item name','stand_name','Cost Center Name','Cost center'])
            df.rename(columns={'Booking Date':'Booking_Date','Gross Amount':'Gross_Amount','Property name':'Property_name', 'Item name':'Item_Name',
                                  'Category':'Category','CategoryID':'CategoryID','Cost center':'Cost_center','Cost Center Name':'Cost_Center_Name'}, inplace=True)
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            df['Item_Name'] = df['Item_Name'].str.lower()
            logging.info("Read the dataset as dataframe")

            cat_df = pd.read_csv(f'Level_4/data/category_with_ID.csv')
            cat_df = cat_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            cat_df['CategoryID'] = cat_df['CategoryID'].astype(str)
            logging.info("Read the category dataset")

            mapping_df = pd.read_csv(f'Level_4/data/item_name_mapping.csv',encoding = 'latin1'
                                ,usecols=['Item_Name','Item_Name_Standard','Category_Standard','Remove_flag','CategoryID_Standard'])
            mapping_df = mapping_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            mapping_df['Item_Name'] = mapping_df['Item_Name'].str.lower()
            mapping_df['Item_Name_Standard'] = mapping_df['Item_Name_Standard'].str.lower()
            mapping_df_ = mapping_df.drop_duplicates().reset_index(drop=True)
            mapping_df_['Remove_flag'] = mapping_df_['Remove_flag'].astype(str)
            mapping_df_ = mapping_df_[mapping_df_['Remove_flag']=='0']
            logging.info("Read the mapping sheet")

            merged_property_data_df = pd.merge(df, mapping_df_[['Item_Name','Category_Standard','Item_Name_Standard','CategoryID_Standard']], on=['Item_Name'], how='inner')
            merged_property_data_df['CategoryID_Standard'] = merged_property_data_df['CategoryID_Standard'].astype(str)
            final_data_set = merged_property_data_df[merged_property_data_df['CategoryID_Standard'].isin(cat_df.CategoryID.tolist())].reset_index(drop=True)

            os.makedirs(os.path.dirname(self.ingestion_config.final_data_path),exist_ok=True)
            final_data_set.to_csv(self.ingestion_config.final_data_path,index=False,header=True)
            # cat_df.to_csv(self.ingestion_config.mapping_sheet_path,index=False,header=True)
            # mapping_df.to_csv(self.ingestion_config.category_path,index=False,header=True)
            logging.info('data ingestion is completed')

            return(
                self.ingestion_config.final_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=="__main__":
        obj = DataIngestion()
        final_data_path=obj.initiate_data_ingestion()

        # data_tranformation = DataTransformation()
        # train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

        # modeltrainer = ModelTrainer()
        # print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
        



