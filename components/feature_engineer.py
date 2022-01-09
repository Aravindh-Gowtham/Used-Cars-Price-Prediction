from .data_modeler import DataModeler

import pandas as pd
from typing import List

class FeatureEngineer(DataModeler):
    "Wrap the operations of feature engineering."
    
    def __init__(self):
        super(FeatureEngineer, self).__init__()
    
    def _groupby_feats(self, df:pd.DataFrame, col:str, groupby_cols:List[str], method:str) -> pd.DataFrame:
        """
        Generate groupby features
        @param df: Dataframe
        @param col: target column
        @param groupby_cols: Columns used to groupby
        @param method: aggregate method, e.g. 'mean'
        """
        df_tmp = df[groupby_cols+[col]]
        df_tmp = df_tmp.groupby(groupby_cols, as_index=False)[col].agg([method]).reset_index()
        df_tmp = df_tmp.rename(columns={method: '_'.join(groupby_cols)+'__'+ col +'__'+method})
        df = pd.merge(df, df_tmp, on=groupby_cols, how='left')

        return df
    
    def _target_encode(self, df:pd.DataFrame, groupby_cols:List[str], method:str) -> pd.DataFrame:
        """
        Target encode categorical features
        @param df: Dataframe
        @param groupby_cols: Columns used to groupby
        @param method: aggregate method, e.g. 'mean'
        """
        df_tmp = df[df['is_train']==1][groupby_cols+['price']]
        df_tmp = df_tmp.groupby(groupby_cols,as_index=False)['price'].agg([method]).reset_index()
        df_tmp = df_tmp.rename(columns={method: '_'.join(groupby_cols)+'__price__'+method})
        df = pd.merge(df, df_tmp, on=groupby_cols, how='left')

        return df

    def generate_feats(self, df:pd.DataFrame) -> pd.DataFrame:
        """
        Generate new features.
        @param df: dataframe
        return: dataframe with new features
        """
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year"], 'mean')
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year", "engine_cylinders", "fuel_type"], 'mean')
        df = self._groupby_feats(df, 'mileage', ['make_name','model_name'], 'std')
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year", "engine_cylinders", "fuel_type"], 'std')
        df = self._groupby_feats(df, 'mileage', ["make_name", "engine_cylinders", "engine_displacement", "horsepower", "fuel_type"], 'mean')
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year", "body_type", "engine_cylinders", "engine_displacement", "horsepower", "fuel_type"], 'mean')

        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year"], 'count')
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year", "engine_cylinders", "fuel_type"], 'count')
        df = self._groupby_feats(df, 'mileage', ["make_name", "engine_cylinders", "engine_displacement", "horsepower", "fuel_type"], 'count')
        df = self._groupby_feats(df, 'mileage', ["make_name", "model_name",  "year", "body_type", "engine_cylinders", "engine_displacement", "horsepower", "fuel_type"], 'count')
        
        df = self._target_encode(df, ['make_name','model_name'], 'mean')
        df = self._target_encode(df, ["make_name", "model_name",  "year"], 'mean')

        cat_feats = ['body_type', 'engine_cylinders', 'fleet', 'frame_damaged', 'fuel_type',
        'has_accidents', 'isCab', 'listing_color', 'make_name', 'model_name',
        'salvage', 'theft_title', 'transmission', 'wheel_system']
        for f in cat_feats:
            df = self._target_encode(df, [f], 'mean')
            df = self._groupby_feats(df, 'mileage', [f], 'mean')
        
        df['mile_per_year'] = df['mileage'] / (2022 - df['year'])
        df['city_fuel_spent'] = df['mileage'] * df['city_fuel_economy']
        df['highway_fuel_spent'] = df['mileage'] * df['highway_fuel_economy']


        return df