# Здесь все функции для модели фрода
import pandas as pd
import re
import numpy as np
import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats.mstats import winsorize
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, EShapCalcType, EFeaturesSelectionAlgorithm

from itertools import cycle, combinations
import json
from copy import copy
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
import phik
import plotly.express as px
from catboost import EFstrType

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix

class MVPError(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = None
            
    def __str__(self):
        if self.message:
            return self.message
        else:
            return 'Неизвестная ошибка'

        
class LabelEncoderPro():
    def __init__(self):
        self.map_level_dict = dict()
        self.classes_ = []
        pass
    
    def fit(self, Serie):
        self.map_level_dict = {i:j for j,i in dict(enumerate(Serie.astype('category').cat.categories)).items()}
        self.classes_ = self.map_level_dict.keys()
        
    def transform(self, Serie):
        Serie = Serie.map(self.map_level_dict).fillna(-1).astype('int')
        return Serie
    
    def fit_transform(self, Serie):
        self.fit(Serie)
        return self.transform(Serie)
    
    def inverse_transform(self, Serie):
        inv_dict = {v: k for k, v in self.map_level_dict.items()}
        Serie = Serie.map(inv_dict).replace(-1, None)
        return Serie

    
class Preprocessor():
    """ Комбайн для препроцессинга """
    
    def __init__(self):
        #from sklearn.preprocessing import LabelEncoder

        # Словарь для хранения кодировщиков
        self.dict_of_encoders = dict()
        # Словарь для хранения параметров
        self.dict_params = dict()

        # Словарь методов для применения
        self.dict_methods = dict({"std": StandardScaler, "minmax": MinMaxScaler, "label": LabelEncoderPro, None:None})
        pass
                

    def fit_Encoder(self, Serie, method, column):       
        """Функция нижнего уровня, выполняет преобразование и записывает енкодер"""
        if method == None:
            return Serie
        if method not in self.dict_methods:
            return
        if method != "label":
            Serie = np.array(Serie).reshape(len(Serie),1)
        E = self.dict_methods[method]()
        E.fit(Serie)
        self.dict_of_encoders[column] = E

    def transform_Encoder(self, Serie, method, column):
        """Функция нижнего уровня, выполняет преобразование"""
        if method == None:
            return Serie
        if method not in self.dict_methods:
            Serie.map(method)
        if method != "label":
            Serie = np.array(Serie).reshape(len(Serie),1)
        return self.dict_of_encoders[column].transform(Serie)
        
    def fit_transform_Encoder(self, Serie, method, column):
        """Функция нижнего уровня, выполняет преобразование и записывает енкодер"""
        self.fit_Encoder(Serie, method, column)
        return self.transform_Encoder(Serie, method, column)
    
        
    def fit_transform_Serie(self, Serie, method, depth=None, 
                            q1=None, q2=None, cut_outliers=False, name_of_feature = None):
        """ Функция среднего уровня, работает с серией из датафрейма, обучает и применяет энкодер

        Parameters
        ----------
        Serie : pd.Series
            Серия для применения к ней преобразований
        method : str or None
            "std" for StandardScaler
            "minmax" for MinMaxScaler
            "label" for LabelEncoderPro
            None for None
        depth: float
            [0-1] для отсечения по долям. Если какого-то значения меньше 0.01 (1%), то его строки
            попадут в отдельную объединённую категорию
            > 1: int, для отсечения по количеству в value_counts
        q1, q2 : float
            границы для отсечения выбросов
        cut_outliers : bool
            Если True, значения выбросов будут отправлены в None для дальнейшей работы
            Если False, значения выбросов будут заменены на границы отсечения (винсоризация)
        name_of_feature: str
            Имя серии (ключ в словаре кодировщиков)

        Returns
        -------
        self.fit_transform_Encoder(Serie, method, name_of_feature): pd.Series
            Изменённая серия        
        """
        self.dict_params[name_of_feature] = {"depth":depth, "q1": q1, "q2" :q2,
                                             "cut_outliers": cut_outliers, "VC": None,
                                            "min": None, "max": None}
        if depth:
            if 0<depth<1:
                VC = Serie.value_counts(dropna=False, normalize=True).reset_index()
                try:
#                     VC = VC[VC[Serie.name] > depth]["index"]
                    VC = VC[VC['proportion'] > depth][Serie.name]
                except:
#                     VC = VC[VC[0] > depth]["index"]
                    VC = VC[VC['proportion'] > depth][Serie.name]
                
            else:
                VC = Serie.value_counts(dropna=False).reset_index()[:int(depth)]["index"]
            self.dict_params[name_of_feature]["VC"] = VC
            return self.fit_transform_Encoder(Serie.apply(lambda x: x if (x in set(VC)) or (pd.isnull(x)) else "N/A"),
                                method,  name_of_feature)

        if q1 or q2:
            if cut_outliers:
                Serie = self.outliers_mask(Serie, q1, q2)
            else:
                Serie = winsorize(Serie, limits=[q1, q2], nan_policy="omit").data
            self.dict_params[name_of_feature]["min"] = Serie.min()
            self.dict_params[name_of_feature]["max"] = Serie.max()
            return self.fit_transform_Encoder(Serie, method, name_of_feature)
        if cut_outliers:
            Serie = self.outliers_mask(Serie, 0.001, 0.999)
            self.dict_params[name_of_feature]["min"] = Serie.min()
            self.dict_params[name_of_feature]["max"] = Serie.max()
        return self.fit_transform_Encoder(Serie, method, name_of_feature)
        
    def transform_Serie(self, Serie, method, name_of_feature = None):
        """ Функция среднего уровня, работает с серией из датафрейма, обучает и применяет энкодер

        Parameters
        ----------
        Serie : pd.Series
            Серия для применения к ней преобразований
        name_of_feature: str
            Имя серии (ключ в словаре кодировщиков)
            
        Returns
        -------
        self.fit_transform_Encoder(Serie, method, name_of_feature): pd.Series
            Изменённая серия        
        """
        depth = self.dict_params[name_of_feature]["depth"]
        q1 = self.dict_params[name_of_feature]["q1"]
        q2 = self.dict_params[name_of_feature]["q2"]
        cut_outliers = self.dict_params[name_of_feature]["cut_outliers"]
        VC = self.dict_params[name_of_feature]["VC"]
        minimum = self.dict_params[name_of_feature]["min"]
        maximum = self.dict_params[name_of_feature]["max"]

        if depth:
            return self.transform_Encoder(Serie.apply(lambda x: x if (x in set(VC)) or (pd.isnull(x)) else "N/A"),
                                    method,  name_of_feature)
        if q1 or q2 or cut_outliers:
            Serie.loc[Serie.between(minimum, maximum) == False] = None
        return self.transform_Encoder(
            Serie.apply(lambda x: x if (x in set(VC)) or (pd.isnull(x)) else "N/A"),
            method,
            name_of_feature,
        )
        
    def fit_transform(self, df:pd.DataFrame, features, method, depth=None, 
                      q1=None, q2=None, cut_outliers=False, inplace=False):
        """ Функция высшего уровня, работает с датафреймом, обучает и применяет энкодер ко всем указанным колонкам

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для применения к нему преобразований
        features: list of str
            Список колонок для преобразования
        method : str or None
            "std" for StandardScaler
            "minmax" for MinMaxScaler
            "label" for LabelEncoderPro
            None for None
        depth: float
            [0-1] для отсечения по долям. Если какого-то значения меньше 0.01 (1%), то его строки
            попадут в отдельную объединённую категорию
            > 1: int, для отсечения по количеству в value_counts
        q1, q2 : float
            границы для отсечения выбросов
        cut_outliers : bool
            Если True, значения выбросов будут отправлены в None для дальнейшей работы
            Если False, значения выбросов будут заменены на границы отсечения (винсоризация)
        inplace: bool
            Если True, то изменится исходный датафрейм

        Returns
        -------
        res: pd.DataFrame
            Изменённый датафрейм       
        """
        if type(features) ==str:
            features = [features]
        res = df.copy()
        for colum in tqdm(features):
            res[colum] = self.fit_transform_Serie(df[colum], method,depth, q1, q2, cut_outliers, colum)
            if inplace:
                df[colum] = res[colum]
        return res
    
    def transform(self, df:pd.DataFrame, features, method, inplace=False):
        """Применение уже обученного кодировщика"""
        if type(features) ==str:
            features = [features]
        res = df.copy()
        for colum in tqdm(features):
            res[colum] = self.transform_Serie(df[colum], method, colum)
            if inplace:
                df[colum] = res[colum]
        return res
    
    def outliers_mask(self, Serie, q1, q2):
        """Наложение маски, все выбросы отправляются в None"""
        if q1==None:
            q1=0
        if q2 == None:
            q2=1
        Serie.loc[Serie.between(Serie.quantile(q1), Serie.quantile(q2))==False] = None
        return Serie

    def inverse_transform(self, df:pd.DataFrame, features=None, inplace=True):
        """Функция высшего уровня, работает с датафреймом, применяет энкодер 
        ко всем указанным колонкам в обратную сторону

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм для применения к нему преобразований
        features: list of str
            Список колонок для обратного преобразования. Если пустой, то будет работа со всеми колонками
        inplace: bool
            Если True, то изменится исходный датафрейм

        Returns
        -------
        res: pd.DataFrame
            Изменённый датафрейм        
        """
        res = df.copy()
        
        if not features:
            features = df.columns
            
        if type(features) == str:
            features = list(features)
        for col in tqdm(features):
            if col not in df.columns:
                print(col, " не найден")
                continue
            if col in self.dict_of_encoders.keys():
                try:
                    res[col] = self.dict_of_encoders[col].inverse_transform(df[col])
                except:
                    res[col] = self.dict_of_encoders[col].inverse_transform(np.array(df[col]).reshape(len(df[col]) ,1))
            else:
                res[col] = df[col]
            if inplace:
                df[col] = res[col]

        return res

    def inverse_transform_Serie(self, Serie, name_of_feature = None):
        """Отдельная функция для одиночного применения к серии"""
        if name_of_feature:
            res = self.dict_of_encoders[name_of_feature].inverse_transform(Serie)
        else:
            res = self.dict_of_encoders[Serie.name].inverse_transform(Serie)
        return res
         
    def transform_by_group(self, df:pd.DataFrame, group_features, agg_func, column, log, multiplier=1):
        """Функция для квантилизации"""
        GB = df.groupby(group_features)[column].agg(agg_func).reset_index()
        if log:
            GB[column] = np.log(GB[column]+1)
        GB[column+"_cut"] = pd.qcut(GB[column], q= int((GB[column].max() - GB[column].min())*multiplier))
        LE = LabelEncoder()
        GB[column+"_cut"] = LE.fit_transform(GB[column+"_cut"])
        self.dict_of_encoders[column+"_cut"]  = LE
        res = pd.merge(df, GB.drop(column, axis=1), on=group_features, how="left")
        return res
    
    def fillna_by_group(self, df, group_features, target_feature):
        """Справляется в случаях, когда более одной групповой переменной"""
        df.groupby(group_features, sort=False)[target_feature].apply(lambda x: x.fillna(x.median()))
        
    """
    Работа с численными признаками
    """

    ##винсоризация
    def robust_cut(s,a=0,b=1):
        return s.apply(lambda x:a if x<a else b if x>b else x)
    
    ##винсоризация по квантилям, но не предусмотрена отрезка выбросов в None
    def quant_cut(s, q1=0.01, q2=0.99):
        return robust_cut(s, s.quantile(q1), s.quantile(q2))
    




class MVP():
    """
    df - исходный датафрейм
    count_category - число уровней, меньше которых фичу считать категорией, больше - объектовой
    cutoff_1_category - если один уровень занимает 99% датасета, выкидываем такую фичу
    cutoff_nan - порог по количеству пропусков. Если превышен - избавляемся от фичи
    isprint - для отладки прохода по фичам
    ftrs - набор признаков перед отбором
    last - набор признаков после отбора
    dicts - словарь предобработок Preprocessor
    types_dict - словарь типов фичей
    config_framework - DataModelConfig для сохранения в файл конфига фреймворка
    model - обученная модель (можно вытащить значимости фичей)
    """
    def __init__(self,
                 df: pd.DataFrame,
                 print_col_type: bool = False,
                 count_category: int = 100,
                 cutoff_1_category: float = 0.99,
                 cutoff_nan: float = 0.7,
#                  model_name: str,
#                  config: AllModelsConfig,
#                  model: Any,
#                  datasubset: DataSubset,
#                  model_config: ModelConfig
                 ):
        self.count_category = count_category
        self.cutoff_1_category = cutoff_1_category
        self.cutoff_nan = cutoff_nan
        
        self.df = df
        self.isprint = print_col_type
        self.ftrs = None
        self.last = None
        self.dicts = {}
        
        self.types_dict = {}
        self.config_framework = None
        self.model = None
        
        self.target_col = None
        self.train_X, self.test_X, self.train_y, self.test_y = None, None, None, None
        self.cat_ftrs = None
        self.phik_matrix = None
        self.to_drop = None
        self.depth = 0.01
        pass
    
    
    def value_type(self):
        """
        Функция для разделения признаков по количеству значений данных в них

        Parameters
        ----------
        df : pd.DataFrame
            Датафрейм, из которого будут получены данные
        isprint : bool
            Флаг, отвечающий за то, будет ли выводиться строка после распределения по каждому признаку

        Returns
        -------
        (bin_list, cat_list, num_list, drop_list, obj_list): Cortage of 5 [list of str]
            (
Список бинарных
признаков (2 значения),
            Список категориальных признаков (от 3 до 20 уникальных значений в столбцах),
            Список числовых признаков (всё, что не object с большим количеством значений),
            Список признаков на удаление (1 значение), 
            Список признаков типа object (обязательны к рассмотрению),
            ) 

        Examples
        --------
        >>> (bin_list, cat_list, num_list, drop_list, obj_list) = value_type(df, isprint=False)
            BINARY: ['EventCreatedByGIBDDFlag', 'E-Garant', <...> ]
            CATEGORIAL: ['CustomerImportance', 'DTPOSAGOType', <...>]
            NUMERIC: ['LossNumber', 'InsuredSum', 'LossDateTime', <...>]
            TO_DROP: ['EventTypeDescription', 'InsuranceTypeName', <...>]
            OBJECT: ['ContractNumber', 'VictimContractNumber', <...>]
        """
        # Инициализация списков
        bin_list, cat_list, num_list, drop_list, date_list, obj_list = [], [], [], [], [], []
        
        # Цикл по колонкам датафрейма
        for col in tqdm(self.df.columns):
            try:
                VC = self.df[col].nunique(dropna=False)
            except:
                print(col, ' не хэшируемый тип')
                continue
            # Если только 1 значение
            if VC ==1  or \
                self.df[col].value_counts(normalize=True, dropna=False).values[0] > self.cutoff_1_category or \
                self.df[col].isna().mean() > self.cutoff_nan:
                if self.isprint: print('DROP:', col )
                drop_list.append(col)
            # Если только 2 значения
            elif VC ==2:
                if self.isprint: print('BINARY:', col )
                bin_list.append(col)
            # Если значений в столбце от 3 до count_category
            elif 2 < VC <= self.count_category and self.df[col].dtype == object:
                if self.isprint: print('CATEGORIAL:', col )
                cat_list.append(col)
            elif self.df[col].dtype == object:
                if self.isprint: print('object:', col )
                obj_list.append(col)
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                if self.isprint: print('date:', col )
                date_list.append(col)
            else:
                if self.isprint: print('numeric:', col )
                num_list.append(col)
            
        self.types_dict = {"BINARY": bin_list,
                 "CATEGORIAL": cat_list,             
                "NUMERIC": num_list,
                 "TO_DROP": drop_list,
                 "OBJECT": obj_list, 
                 "DATE": date_list
                }
        self.ftrs = self.types_dict['NUMERIC'] + self.types_dict['CATEGORIAL']  + self.types_dict['BINARY']
        for key, value in self.types_dict.items():
            print(f"{key}:", value)
        return self.types_dict
#         return self.types_dict
    
    def correct_types(self, 
                      input_types_dict:dict,
                     other_cols: list):
        self.input_types_dict = input_types_dict
        for key in input_types_dict.keys():
            for col in input_types_dict[key]:
                for key_in in self.types_dict.keys():
                    if key == key_in:
                        if col not in self.types_dict[key_in]:
                            self.types_dict[key_in].append(col)
                    else:
                        if col in self.types_dict[key_in]:
                            self.types_dict[key_in].remove(col)

        for key in other_cols:
            for key_in in self.types_dict.keys():
                if key in self.types_dict[key_in]:
                    self.types_dict[key_in].remove(key)
        if self.to_drop:
            self.types_dict = {i:[k for k in j if k not in self.to_drop] for i, j in self.types_dict.items()}
            self.types_dict['TO_DROP'] += self.to_drop

        self.ftrs = self.types_dict['NUMERIC'] + self.types_dict['CATEGORIAL']  + self.types_dict['BINARY']

        
    def prepare_data(self, 
                    target_col: str = None,
                    ftrs: list[str] = None,
                    depth: float = None,
                    test_size: float = 0.2,
                    random_state:int = 42):
        
        if not self.target_col:
            if not target_col:
                raise MVPError("Параметр 'target_col' в функции не определён")
            else:
                self.target_col = target_col
            
        if depth:
            self.depth = depth
            
        if ftrs:
            self.ftrs = ftrs
        else:
            self.ftrs = self.types_dict['NUMERIC'] + self.types_dict['CATEGORIAL']  + self.types_dict['BINARY']
        X, y = self.df[self.ftrs].copy(), self.df[target_col]
        for col in self.types_dict['BINARY']:
            try:
                X[col] = X[col].astype(int)
            except:
                pass
        
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, 
                                                                                test_size=test_size, 
                                                                                random_state=random_state)
        
        binary_str = [i for i in self.types_dict['BINARY'] if isinstance(self.df[i].value_counts().index[0], str)]
        
        P_CB = Preprocessor()
        P_CB.fit_transform(self.train_X, [i for i in (self.types_dict['CATEGORIAL'] + binary_str) if i in self.ftrs],
                                     "label", depth = self.depth, inplace=True)
        self.test_X = P_CB.transform(self.test_X, [i for i in (self.types_dict['CATEGORIAL'] + binary_str) if i in self.ftrs], "label")

        self.dicts['dict_params'] = P_CB.dict_params
        self.dicts['map_dict'] = {}
        for col in P_CB.dict_of_encoders.keys():
            if col in self.ftrs:
                self.dicts['map_dict'][col] = P_CB.dict_of_encoders[col].map_level_dict

        
    def single_fit(self,  
                   params: dict,
                   ftrs: list[str] = None,
                   catboost_classification: bool = True):

        if not ftrs:
            if not self.ftrs:
                raise MVPError("Параметр 'ftrs' в функции не определён")
            else:
                ftrs = self.ftrs

        cat_ftrs = [i for i in self.types_dict['CATEGORIAL'] + self.types_dict['BINARY'] if i in ftrs]

        if catboost_classification:
            self.model = CatBoostClassifier(**params)
        else:
            self.model = CatBoostRegressor(**params)

        self.model.fit(
            self.train_X[ftrs], self.train_y,
            eval_set=(self.test_X[ftrs], self.test_y),
            cat_features=cat_ftrs
        )

        # ======= МЕТРИКИ ДЛЯ КЛАССИФИКАЦИИ =======
        if catboost_classification:
            y_true = self.test_y
            y_pred = self.model.predict(self.test_X[ftrs])
            # для бинарной классификации: predict_proba, для multi - может понадобиться один из столбцов
            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(self.test_X[ftrs])[:, 1]  
            else:
                y_proba = None

            metrics = {
                'precision': precision_score(y_true, y_pred, average='binary'),
                'recall': recall_score(y_true, y_pred, average='binary'),
                'f1': f1_score(y_true, y_pred, average='binary'),
                'accuracy': accuracy_score(y_true, y_pred),
                'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
            }
            # ROC AUC возможен только если есть вероятности, и для бинарной задачи
            if y_proba is not None and len(set(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

            self.metrics = metrics  # можно сохранить метрики в self, если нужно

            print("Test metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v}")

        
        
        
    
    def show_importances(self, show_first:int = 10, leak_level:float = 50):
        fig, ax = plt.subplots(dpi = 150, figsize = (7,12))
        a = dict(zip(self.model.feature_names_, self.model.feature_importances_))
        a_res = pd.DataFrame(a.items()).sort_values(1, ascending=False).rename({0:'feature', 1:'importance'}, axis=1)
        a = pd.DataFrame(a.items()).sort_values(1)[-show_first:]
        ax.barh(a[0], a[1])
        plt.show()
        
        if a[a[1] > leak_level].shape[0]:
            print("Одна из фичей является утечкой данных и приводит к переобучению")
            return a[a[1] > leak_level]
        return a_res

    def shap_feature_ranking(self, data, shap_values, columns=[]):
        if not columns: columns = data.columns.tolist()     # If columns are not given, take all columns

        c_idxs = []
        # Get column locations for desired columns in given dataframe
        for column in columns: c_idxs.append(data.columns.get_loc(column))  
        # If shap values is a list of arrays (i.e., several classes)
        if isinstance(shap_values, list):   
            # Compute mean shap values per class 
            means = [np.abs(shap_values[class_][:, c_idxs]).mean(axis=0) for class_ in range(len(shap_values))]  
            shap_means = np.sum(np.column_stack(means), 1)  # Sum of shap values over all classes 
        else:                               # Else there is only one 2D array of shap values
            assert len(shap_values.shape) == 2, 'Expected two-dimensional shap values array.'
            shap_means = np.abs(shap_values).mean(axis=0)

        # Put into dataframe along with columns and sort by shap_means, reset index to get ranking
        df_ranking = pd.DataFrame({'feature': columns + ['C'], 'importance': shap_means}).sort_values(
            by='importance', ascending=False).reset_index(drop=True)
        df_ranking.index += 1
        return df_ranking
    
    def show_importances_shap(self, show_first:int = 10):
        cat_ftrs = [i for i in self.types_dict['CATEGORIAL'] + self.types_dict['BINARY'] if i in self.model.feature_names_]
        shap_values = self.model.get_feature_importance(Pool(self.test_X[self.model.feature_names_], self.test_y,
                                                                 cat_features = cat_ftrs),
                                                                    type = EFstrType.ShapValues)
        ranking = self.shap_feature_ranking(self.test_X[self.model.feature_names_], shap_values)
        fig, ax = plt.subplots(dpi = 150, figsize = (7,12))
        a = ranking.sort_values('importance')[-show_first:]
        ax.barh(a['feature'], a['importance'])
        plt.show()
        
        return ranking

    def feature_selection(self, 
                           top_features: int,
                          params: dict,
                          target_col: str,
                          ftrs: list[str] = None,
                          catboost_classification:bool = True,
                         depth: float = 0.01
                         ):
        self.prepare_data(target_col, ftrs, depth)
        
        self.cat_ftrs = self.types_dict['CATEGORIAL'] + self.types_dict['BINARY']
        train_pool = Pool(self.train_X, self.train_y, feature_names=list(self.train_X.columns),
                  cat_features = self.cat_ftrs)
        test_pool = Pool(self.test_X, self.test_y, feature_names=list(self.train_X.columns), 
                         cat_features = self.cat_ftrs)
        
        if catboost_classification:
            self.model = CatBoostClassifier(**params)
            summary = self.model.select_features(
                train_pool,
                eval_set=test_pool,
                features_for_select = f'0-{self.train_X.shape[1] - 1}',
                num_features_to_select= 1,
            #     steps=train_X.shape[1] - 1,
                steps= self.train_X.shape[1] - 1,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model=True,
                logging_level='Silent',
                plot=False
            )
        else:
            self.model = CatBoostRegressor(**params)
            summary = self.model.select_features(
                train_pool,
                eval_set=test_pool,
                features_for_select = f'0-{self.train_X.shape[1] - 1}',
                num_features_to_select= 1,
            #     steps=train_X.shape[1] - 1,
                steps= self.train_X.shape[1] - 1,
                algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
                shap_calc_type=EShapCalcType.Regular,
                train_final_model=True,
                logging_level='Silent',
                plot=False
            )            
        res = pd.DataFrame([summary['eliminated_features_names'] + summary['selected_features_names'],
                    summary['loss_graph']['loss_values']]).T#.plot()
        self.last = list(res[res.index >= (res.index.max() - top_features)][0].values)
        self.single_fit(params= params, 
                        ftrs = self.last,
                        catboost_classification = catboost_classification)
        return self.last

    def get_correlation(self,
                        threshold: float = 0.9):
        if self.train_X is not None:
            X = self.train_X
        else:
            X = df
        if self.last:
            X = X[reversed(self.last)]
        else:
            X = X[self.types_dict['NUMERIC'] + self.types_dict['CATEGORIAL'] + self.types_dict['BINARY']]
            
        self.phik_matrix = X.phik_matrix(interval_cols = self.types_dict['NUMERIC'])
        
        fig = px.imshow(self.phik_matrix)
        fig.show()
        
        upper = self.phik_matrix.where(np.triu(np.ones(self.phik_matrix.shape), k=1).astype(bool))

        # Найти признаки с корреляцией выше порогового значения
        self.to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return self.to_drop
    
    def drop_correlation(self,
                         to_drop:list = None):
        if to_drop:
            self.to_drop = to_drop
        self.types_dict = {i:[k for k in j if k not in to_drop] for i, j in self.types_dict.items()}
        self.types_dict['TO_DROP'] += to_drop
        
        self.prepare_data()
        
    
    def save_data_config(self,
                         path = 'config_framework.json',
                        default_N_A = True,
                         default_cat = -1,
                         default_num = float(np.nan),
                         fillna = None # -100
                        ):
        config_framework = []

        def correct_dict(d):
            def correct_key(key):
                if isinstance(key, np.floating):
                    return float(key)
                if isinstance(key, np.integer):
                    return int(key)
                return key
            return {correct_key(i): j for i, j in d.items() if pd.notna(i)}


        for col in tqdm([col for col in self.types_dict['BINARY'] + self.types_dict['CATEGORIAL'] if col in self.last]):
            if isinstance(self.df[col].value_counts().index[0], str):
    #             dict_NA = {i: 0 for i in 
    #                        list(set(self.df[col].unique()) -
    #                             set(self.dicts['map_dict'][col].keys()))}
                col_dict = copy(self.dicts['map_dict'][col])

            #     col_dict.update(dict_NA)
                config_framework.append({'name': col})

                if default_N_A:
                    config_framework[-1]['default'] = self.dicts['map_dict'][col]['N/A'] \
                                                        if 'N/A' in self.dicts['map_dict'][col] else default_cat
                else:
                    config_framework[-1]['default'] = default_cat

                if fillna:
                    config_framework[-1]['fillna'] = fillna

                config_framework[-1]['replace'] = correct_dict(col_dict)


            else:
                config_framework.append({
                    'name': col,
                    'default': default_cat, 
                    'replace':  {"_TYPE_": "_NUM_"}
                }) 


        for col in tqdm([col for col in self.types_dict['NUMERIC'] if col in self.last]):
            config_framework.append({
                'name': col,
                'default': default_num,  # float(np.nan)
                'replace':  {"_TYPE_": "_NUM_"}
            }) 

#         print(config_framework)

        self.config_framework = config_framework
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(config_framework, f, ensure_ascii=False, indent=4)
        return self.config_framework