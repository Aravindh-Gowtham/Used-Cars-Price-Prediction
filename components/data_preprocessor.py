from .data_modeler import DataModeler

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import List

class DataPreprocessor(DataModeler):
    "Wrap the operations of data preprocessing."
    
    def __init__(self):
        super(DataPreprocessor, self).__init__()
        self.label_encoder_dict = {}

    @staticmethod
    def create_folds(data:pd.DataFrame, num_splits:int) -> pd.DataFrame:
        """
        Use stratified kfold to split data according to target distribution.
        @param data: datafram
        @param num_splits: number of splits
        return: splitted data, use a new column named 'fold' to separate
        """
        data["fold"] = -1
        num_bins = int(np.floor(1 + np.log2(len(data))))
        data.loc[:, "bins"] = pd.cut(data["price"], bins=num_bins, labels=False)
        kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=2022)
        
        for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
            data.loc[v_, 'fold'] = f
        
        data = data.drop("bins", axis=1)

        return data

    #* =======================================================================================
    @DataModeler.logger("Dropping useless data")
    def _drop_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows and columns
        @param df: Dataframe
        return: Processed dataframe
        """
        #* Drop rows with duplicate vin
        df = df.groupby('vin', as_index=False).first()        
        #* Drop rows with VIN length not equal to 17
        df = df[df['vin'].str.len()==17]             
        #* Drop rows with duplicate listing_id 
        df = df.groupby('listing_id', as_index=False).first()

        #* Drop invalid columns
        cols_to_drop = [
            'vehicle_damage_category', 'combine_fuel_economy', 'is_certified', 'bed', 'cabin',
            'is_oemcpo', 'is_cpo', 'bed_length', 'bed_height', 'engine_type', 
            'wheel_system_display', 'trim_name', 'trimId', 'sp_id', 'sp_name',
            'latitude', 'longitude', 'city', 'dealer_zip', 'daysonmarket', 'listed_date',
            'main_picture_url', 'savings_amount', 'franchise_dealer', 'seller_rating', 'franchise_make',
            'description', 'major_options', 'listing_id', 'interior_color', 'exterior_color'
        ]
        df.drop(columns=cols_to_drop, inplace=True)

        return df
    
    #* =======================================================================================
    @DataModeler.logger("Cleaning data")
    def _clean_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Clean data
        @param df: Dataframe
        return: cleaned dataframe
        """
        df = df[pd.array(df['price'] < 1e5)]
        df = df[~df['mileage'].isna()]

        dic_trans_dis = {"Automatic": "6", "Continuously Variable Transmission": "6", "Manual": "6"}
        df["transmission_display"].replace(dic_trans_dis, inplace=True)

        dic_engine = {"I4 Hybrid": "I4", "I4 Diesel": "I4", "I4 Flex Fuel Vehicle": "I4", "V6 Hybrid": "V6", "V6 Biodiesel": "V6" ,
              "V6 Flex Fuel Vehicle": "V6", "V8 Flex Fuel Vehicle": "V8", "I6 Diesel": "I6", "I6 Biodiesel": "I6", "V6 Diesel": "V6", 
              "H4 Hybrid": "H4", "W12 Flex Fuel Vehicle": "W12", "V8 Hybrid": "V8", "I4 Compressed Natural Gas": "I4", "I6 Hybrid": "I6",
              "V6 Compressed Natural Gas": "V6", "V8 Biodiesel": "V8", "V8 Diesel": "V8", "I5 Biodiesel": "I5", "V8 Compressed Natural Gas": "V8",
              "I5 Diesel": "I5", "V8 Propane": "V8", "I3 Hybrid": "I3", "V10 Diesel": "V10", "V12 Hybrid": "V12"}
        df["engine_cylinders"].replace(dic_engine, inplace=True)
        dic_engine_type = {"I2": "Inline 2 cylinder", "I3": "Inline 3 cylinder", "I4": "Inline 4 cylinder", 
                   "I5": "Inline 5 cylinder", "I6": "Inline 6 cylinder", "R2": "Rotary Engine", "Electric_Motor": "Electric Motor",
                   "H4": "Boxer 4 cylinder", "H6": "Boxer 6 cylinder"}
        df["engine_cylinders"].replace(dic_engine_type, inplace=True)

        dic_transmission = {"A": "Automatic Transmission (A)", "CVT": "Continuously Variable Transmission (CVT)", 
                    "Dual Clutch": "Dual Clutch Transmission (DCT)", "M": "Manual Transmission (M)"}
        df["transmission"].replace(dic_transmission, inplace=True)

        dic_wheel_system = {"FWD": "Front Wheel Drive (FWD)", "AWD": "All Wheel Drive (AWD)", "4WD": "Four Wheel Drive (4WD)", "RWD": "Rear Wheel Drive (RWD)",
                    "4X2": "Two Wheel Drive (4X2)"}
        df["wheel_system"].replace(dic_wheel_system, inplace=True)

        return df
    
    #* =======================================================================================
    def __trans_feat_type(self, df:pd.DataFrame, cols:List[str], unit:str) -> pd.DataFrame:
        """
        Transform str type cols in df into specified data type.
        Treat '--' as nan.
        @param df: Dataframe
        @params cols: Columns to be transformed
        @param unit: Unit of measurement, needed to be cropped from the original data
        return: Processed dataframe
        """
        for col in cols:
            mask = df[col].str.contains("--") | df[col].isna() #* nan rows
            df[col][mask] = np.nan                             #* replace '--' as nan
            df[col][~mask] = df[col][~mask].str.replace(" "+unit, "")
            df[col] = df[col].astype(float)
        
        return df

    @DataModeler.logger("Transforming feature type")
    def _trans_feat_type(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature dtype.
        @param df: Dataframe
        return: Processed dataframe
        """
        df = self.__trans_feat_type(df, ['width', 'wheelbase', 'length', 'height', 'front_legroom', 'back_legroom'], "in")
        df = self.__trans_feat_type(df, ['maximum_seating'], "seats")
        df = self.__trans_feat_type(df, ['fuel_tank_volume'], "gal")

        df['torque'] = df['torque'].astype(str)
        df['torque'][~df['torque'].str.contains('RPM')] = np.nan #* there is a row with value '174 lb-ft', replace it as nan
        df['torque_power'] = np.nan
        df['torque_power'][~df['torque'].isna()] = df['torque'][~df['torque'].isna()].apply(lambda x:x.split('@')[0][:-6].replace(',',''))
        df['torque_power'] = df['torque_power'].astype(float)
        df['torque_rpm'] = np.nan
        df['torque_rpm'][~df['torque'].isna()] = df['torque'][~df['torque'].isna()].apply(lambda x:x.split('@')[1][:-3].replace(',',''))
        df['torque_rpm'] = df['torque_rpm'].astype(float)
        del df['torque']

        df['power'] = df['power'].astype(str)
        df['power'][~df['power'].str.contains('RPM')] = np.nan
        df['power_rpm'] = np.nan
        df['power_rpm'][~df['power'].isna()] = df['power'][~df['power'].isna()].apply(lambda x:x.split('@')[1][:-3].replace(',',''))
        df['power_rpm'] = df['power_rpm'].astype(float)
        del df['power']

        df['transmission_display'] = df["transmission_display"].str.extract('(\d+)', expand=False).astype("float16")

        return df
  
    #* =======================================================================================
    def __impute_with_mean(self, df:pd.DataFrame, col:str, groupby_cols:List[str]) -> pd.DataFrame:
        """
        Impute categorical feature with mean.
        @param df: Dataframe
        @param col: Target column
        @param groupby_cols: Columns used for groupby
        return: Imputed dataframe
        """
        df["avg_"+col] = df.groupby(groupby_cols)[col].transform("mean")
        df[col] = np.where(df[col].isna(), df["avg_"+col], df[col])
        del df["avg_"+col]

        return df

    def __impute_with_mode(self, df:pd.DataFrame, col:str, groupby_cols:List[str]) -> pd.DataFrame:
        """
        Impute categorical feature with mode.
        @param df: Dataframe
        @param col: Target column
        @param groupby_cols: Columns used for groupby
        return: Imputed dataframe
        """
        tmp = df.groupby(groupby_cols)[col].apply(lambda x: x.mode()).to_frame(name="mode").reset_index()
        tmp = tmp.drop(["level_2"], axis=1)
        df = df.merge(tmp, on=groupby_cols)
        df[col] = np.where(df[col].isna(), df['mode'], df[col])
        del df['mode']
        del tmp

        return df

    @DataModeler.logger("Imputing data")
    def _impute_data(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Impute nan values
        @param df: Dataframe
        return: Imputed dataframe
        """
        cols = ['back_legroom']
        for col in cols:
            df[col] = df[col].fillna(df[col].mean())
        
        df.loc[df["fuel_type"] == "Electric", "engine_cylinders"] = "Electric_Motor"
        df.loc[df["fuel_type"] == "Electric", "transmission_display"] = 6
        df.loc[df["fuel_type"] == "Electric", "fuel_tank_volume"] = df.fuel_tank_volume.mean()

        df['owner_count'] = df['owner_count'].fillna(1)

        #* Use mode to impute categorical feats
        df = self.__impute_with_mode(df, 'body_type', ['make_name','model_name'])
        df = self.__impute_with_mode(df, 'transmission', ['make_name','model_name'])
        df = self.__impute_with_mode(df, 'fuel_type', ['make_name','model_name'])
        df = self.__impute_with_mode(df, 'engine_cylinders', ['make_name','model_name'])
        df = self.__impute_with_mode(df, 'wheel_system', ['make_name','model_name'])

        #* Use mean to impute numerical feats
        df = self.__impute_with_mean(df, 'maximum_seating', ["make_name", "model_name", "body_type"])
        df = self.__impute_with_mean(df, 'horsepower', ["engine_cylinders", "fuel_type", "make_name", "model_name",  "year"])
        df = self.__impute_with_mean(df, 'engine_displacement', ["engine_cylinders", "fuel_type", "make_name", "model_name",  "year"])
        df = self.__impute_with_mean(df, 'city_fuel_economy', ["body_type", "engine_displacement", "engine_cylinders", "fuel_type", 
                                                                "horsepower", "make_name", "model_name",  "year"])
        df = self.__impute_with_mean(df, 'highway_fuel_economy', ["body_type", "engine_displacement", "engine_cylinders", "fuel_type", 
                                                                "horsepower", "make_name", "model_name",  "year"])
        df = self.__impute_with_mean(df, 'torque_power', ["engine_cylinders", "engine_displacement", "horsepower", "fuel_type", "make_name"])
        df = self.__impute_with_mean(df, 'torque_rpm', ["engine_cylinders", "engine_displacement", "horsepower", "fuel_type", "make_name"])
        df = self.__impute_with_mean(df, 'power_rpm', ["engine_cylinders", "engine_displacement", "horsepower", "fuel_type", "make_name"])
        df = self.__impute_with_mean(df, 'fuel_tank_volume', ["make_name", "model_name",  "year"])


        df['fleet'] = df['fleet'].fillna('Unkown').astype(str)
        df['frame_damaged'] = df['frame_damaged'].fillna('Unkown').astype(str)
        df['has_accidents'] = df['has_accidents'].fillna('Unkown').astype(str)
        df['isCab'] = df['isCab'].fillna('Unkown').astype(str)
        df['salvage'] = df['salvage'].fillna('Unkown').astype(str)
        df['theft_title'] = df['theft_title'].fillna('Unkown').astype(str)

        return df

    #* =======================================================================================
    @DataModeler.logger("Reducing memory usage")
    def _reduce_mem_usage(self, df:pd.DataFrame):
        """
        Rudce memory usage by changing feature dtype.
        Credit to https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        """
        start_mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage of dataframe is :",start_mem_usg," MB")
        NAlist = [] # Keeps track of columns that have missing values filled in. 
        for col in df.columns:
            if df[col].dtype != object:  # Exclude strings
                
                # Print current column type
                print("******************************")
                print("Column: ",col)
                print("dtype before: ",df[col].dtype)
                
                # make variables for Int, max and min
                IsInt = False
                mx = df[col].max()
                mn = df[col].min()
                
                # integer does not support NA, therefore, NA needs to be filled
                if not np.isfinite(df[col]).all(): 
                    NAlist.append(col)
                    df[col].fillna(mn-1,inplace=True)  
                    
                # test if column can be converted to an integer
                asint = df[col].fillna(0).astype(np.int64)
                result = (df[col] - asint)
                result = result.sum()
                if result > -0.01 and result < 0.01:
                    IsInt = True

                
                # Make Integer/unsigned Integer datatypes
                if IsInt:
                    if mn >= 0:
                        if mx < 255:
                            df[col] = df[col].astype(np.uint8)
                        elif mx < 65535:
                            df[col] = df[col].astype(np.uint16)
                        elif mx < 4294967295:
                            df[col] = df[col].astype(np.uint32)
                        else:
                            df[col] = df[col].astype(np.uint64)
                    else:
                        if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                        elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                            df[col] = df[col].astype(np.int64)    
                
                # Make float datatypes 32 bit
                else:
                    df[col] = df[col].astype(np.float32)
                
                # Print new column type
                print("dtype after: ",df[col].dtype)
                print("******************************")
        
        # Print final result
        print("___MEMORY USAGE AFTER COMPLETION:___")
        mem_usg = df.memory_usage().sum() / 1024**2 
        print("Memory usage is: ",mem_usg," MB")
        print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
        return df, NAlist

    #* =======================================================================================
    @DataModeler.logger("Label encoding features")
    def label_encode(self, df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
        """
        Label encode specified columns so they can be fed into the Lightgbm model
        @param df: Datafram
        @param cols: Categorical feature names
        return: Transformed dataframe
        """
        for col in cols:
            if col not in self.label_encoder_dict.keys():
                label_encoder = LabelEncoder()
            else:
                label_encoder = self.label_encoder_dict[col]
            
            df[col] = label_encoder.fit_transform(df[col])
            df[col] = df[col].astype('int')
            self.label_encoder_dict[col] = label_encoder

        return df
    
    @staticmethod
    def onehot_encode(df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
        """
        One-hot encode specified columns so they can be fed into normal models
        @param df: Datafram
        @param cols: Categorical feature names
        return: Transformed dataframe
        """
        for col in cols:
            res = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df,res],axis=1)

        return df
    
    #* =======================================================================================
    def preprocess(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess original data.
        @param df: original data
        return: preprocessed data
        """
        df = self._drop_data(df)
        df = self._clean_data(df)
        df = self._trans_feat_type(df)
        df = self._impute_data(df)

        former_len = df.shape[0]
        df = df.dropna()
        curr_len = df.shape[0]
        print(f"{former_len-curr_len} rows with na are dropped")

        df, _ = self._reduce_mem_usage(df)

        df = df.reset_index(drop=True)
        return df