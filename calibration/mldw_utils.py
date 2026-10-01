import os
import pickle
import pandas as pd
import numpy as np
import math
from pandas.api.types import is_object_dtype
from mldataworker.core.enums import EncodingNames
from mldataworker.datasets_manager import DataSetsManager
from mldataworker.ensemble import EnsembleResult
from mldataworker.core.predict import one_model_predict, ensemble_predict

__version__ = '1.18.0'


def get_woe_bins(manager):
    """Получение групп WOE преобрахований."""
    woe_bins_cat = {}
    woe_bins_num = {}
    for model in manager.get_result().keys():
        woe_bins_cat[model] = []
        woe_bins_num[model] = []
        for feature in manager.get_result()[model].model_config.features:
            if feature.mapping:
                if feature.encoding == EncodingNames.woe_cat:
                    woe_bins_cat[model].append([feature.name, len(feature.mapping), list(feature.mapping.keys())])
                else:
                    woe_bins_num[model].append([feature.name, len(feature.mapping), list(feature.mapping.keys())])
        woe_bins_cat[model] = pd.DataFrame(woe_bins_cat[model], columns=['Признак', 'Кол-во групп', 'Группы'])
        woe_bins_num[model] = pd.DataFrame(woe_bins_num[model], columns=['Признак', 'Кол-во групп', 'Группы'])
        woe_bins_num[model].sort_values('Кол-во групп', inplace=True)

    return woe_bins_cat, woe_bins_num


def get_pickle_models(
    pickle_name,
    pickle_path='.',
):
    """Получение списка моделей в pickle файле."""
    with open(os.path.join(pickle_path, f'{pickle_name}.pickle'), 'rb') as f:
        pickle_models = pickle.load(f)

    pickle_info = []
    for pickle_model in pickle_models:
        if isinstance(pickle_model, EnsembleResult):
            model_name = pickle_model.model_name
            for model in pickle_model.models:
                pickle_info.append([model_name, model[0]])
        else:
            model_name = pickle_model['model_config']['name']
            pickle_info.append([model_name, None])

    return pickle_info


def load_pickle(
    pickle_name,
    pickle_path='.',
    models_names=None,
    ensemble_filter=None,
):
    """Загрузка модели из pickle файла."""
    with open(os.path.join(pickle_path, f'{pickle_name}.pickle'), 'rb') as f:
        pickle_models = pickle.load(f)

    models_result_dict = {}
    for pickle_model in pickle_models:
        if isinstance(pickle_model, EnsembleResult):
            model_name = pickle_model.model_name
            if not models_names or model_name in models_names:
                if ensemble_filter is not None:
                    for model in pickle_model.models:
                        if model[0] == ensemble_filter:
                            models_result_dict[model_name] = model[2]
                else:
                    models_result_dict[model_name] = pickle_model
        else:
            model_name = pickle_model['model_config']['name']
            if not models_names or model_name in models_names:
                models_result_dict[model_name] = pickle_model
    if not models_result_dict:
        print('Не найдена требуемая модель в pickle файле')
        return

    return models_result_dict


def get_pickle_features(
    pickle_name,
    pickle_path='.',
    models_names=None,
    ensemble_filter=None,
):
    pickles = load_pickle(
        pickle_name=pickle_name,
        pickle_path=pickle_path,
        models_names=models_names,
        ensemble_filter=ensemble_filter,
    )

    features_numerical = set()
    features_categorical = set()
    for model_name in pickles:
        pickle_model = pickles[model_name]
        if isinstance(pickle_model, EnsembleResult):
            for model in pickle_model.models:
                features_numerical = features_numerical | set(model[2]['features_numerical'])
                features_categorical = features_categorical | set(model[2]['features_categorical'])
        else:
            features_numerical = features_numerical | set(pickle_model['features_numerical'])
            features_categorical = features_categorical | set(pickle_model['features_categorical'])
    features_numerical = [str(feature) for feature in np.sort(list(features_numerical))]
    features_categorical = [str(feature) for feature in np.sort(list(features_categorical))]
    features = features_numerical + features_categorical

    return features, features_numerical, features_categorical


def calc_pickle_result(
    data,
    pickle_name,
    config_name,
    models_names=None,
    result_models_names=None,
    pickle_path='.',
    config_path='.',
    ensemble_filter=None,
    add_metric_suffix=False,
):
    """Расчет прогнозов и метрик модели с использованием pickle файла."""
    models_result_dict = load_pickle(
        models_names=models_names,
        pickle_name=pickle_name,
        pickle_path=pickle_path,
        ensemble_filter=ensemble_filter,
    )

    ds_manager = DataSetsManager(config_name=os.path.join(config_path, config_name))
    # noinspection PyProtectedMember
    ds_manager._DataSetsManager__load_all_models_config()

    result = {}
    for model_name, model_result_dict in models_result_dict.items():
        if result_models_names is None:
            result_model_name = model_name
        else:
            result_model_name = result_models_names[model_name]
        result[result_model_name] = ds_manager.model_predict(
            data=data,
            model_name=model_name,
            model_result=model_result_dict,
        )

        if add_metric_suffix:
            for metric_type in ['train', 'test']:
                result[result_model_name].metrics[metric_type]['full'] = {
                    f'{model_name}_model_1': result[model_name].metrics[metric_type]['full']
                }

    return result


async def calc_pickle_predict(
    data,
    pickle_name,
    models_names=None,
    result_models_names=None,
    pickle_path='.',
    ensemble_filter=None,
    get_features=True,
    log=False,
    return_prepared_data=False,
):
    """Расчет прогнозов модели с использованием pickle файла."""
    models_result_dict = load_pickle(
        models_names=models_names,
        pickle_name=pickle_name,
        pickle_path=pickle_path,
        ensemble_filter=ensemble_filter,
    )

    result = {}
    prepared_data = {}
    for model_name, model_result_dict in models_result_dict.items():
        if result_models_names is None:
            result_model_name = model_name
        else:
            result_model_name = result_models_names[model_name]

        if get_features:
            features, _, _ = get_pickle_features(
                pickle_name=pickle_name,
                pickle_path=pickle_path,
                models_names=[model_name],
                ensemble_filter=ensemble_filter,
            )
            data_model = data[features]
        else:
            data_model = data

        if isinstance(model_result_dict, EnsembleResult):
            result[result_model_name] = pd.Series()
            if return_prepared_data:
                prepared_data[result_model_name] = pd.DataFrame()

            for model in model_result_dict.models:
                data_predict_model = data_model.query(model[0]).copy()
                model_predict_model = await ensemble_predict(
                    ensemble_name=model_name,
                    ensemble=model[2],
                    features_values=data_predict_model,
                    log=log,
                )
                result_model = pd.Series(model_predict_model['result'][model_name], index=data_predict_model.index)
                result[result_model_name] = pd.concat([result[result_model_name], result_model])
                if return_prepared_data:
                    prepared_data_model = pd.DataFrame(
                        model_predict_model['df'][model_name],
                        index=data_predict_model.index
                    )
                    prepared_data[result_model_name] = pd.concat(
                        [prepared_data[result_model_name], prepared_data_model]
                    )
            result[result_model_name] = result[result_model_name].loc[data.index]
            if return_prepared_data:
                prepared_data[result_model_name] = prepared_data[result_model_name].loc[data.index]

        else:
            data_predict = data_model.copy()
            model_predict = await ensemble_predict(
                ensemble_name=model_name,
                ensemble=model_result_dict,
                features_values=data_predict,
                log=False,
            )
            result[result_model_name] = pd.Series(model_predict['result'][model_name])
            if return_prepared_data:
                prepared_data[result_model_name] = pd.DataFrame(model_predict['df'][model_name])

    if return_prepared_data:
        return result, prepared_data
    return result


async def calc_pickle_predict_packet(
    data,
    pickle_name,
    models_names=None,
    result_models_names=None,
    pickle_path='.',
    ensemble_filter=None,
    packet_size=10000,
    get_features=True,
    log=False,
    return_prepared_data=False,
):
    data_result = {}
    data_prepared_data = {}

    total_size = len(data)
    packet_count = math.ceil(total_size / packet_size)
    for packet in range(packet_count):
        packet_begin = packet_size * packet
        packet_end = packet_begin + packet_size
        print('Диапазон:', packet_begin, packet_end)
        data_packet = data.iloc[packet_begin:packet_end].copy()

        result = await calc_pickle_predict(
            data=data_packet.copy(),
            models_names=models_names,
            pickle_name=pickle_name,
            result_models_names=result_models_names,
            pickle_path=pickle_path,
            ensemble_filter=ensemble_filter,
            get_features=get_features,
            log=log,
            return_prepared_data=return_prepared_data,
        )
        if return_prepared_data:
            result, prepared_data = result
        else:
            prepared_data = None
        for model_name, prediction in result.items():
            if model_name not in data_result:
                data_result[model_name] = prediction
                if return_prepared_data:
                    data_prepared_data[model_name] = prepared_data[model_name]
            else:
                data_result[model_name] = pd.concat([data_result[model_name], prediction], axis=0)
                if return_prepared_data:
                    data_prepared_data[model_name] = pd.concat(
                        [data_prepared_data[model_name], prepared_data[model_name]],
                        axis=0,
                    )

        del data_packet
        del result
        del prepared_data

    if return_prepared_data:
        return data_result, data_prepared_data
    return data_result


def get_pickle_prepared_data(
    data,
    pickle_name,
    models_names=None,
    result_models_names=None,
    pickle_path='.',
    ensemble_filter=None,
    get_features=True,
):
    models_result_dict = load_pickle(
        models_names=models_names,
        pickle_name=pickle_name,
        pickle_path=pickle_path,
        ensemble_filter=ensemble_filter,
    )

    data_prepared = {}
    for model_name, model_result_dict in models_result_dict.items():
        if result_models_names is None:
            result_model_name = model_name
        else:
            result_model_name = result_models_names[model_name]

        if get_features:
            features, _, _ = get_pickle_features(
                pickle_name=pickle_name,
                pickle_path=pickle_path,
                models_names=[model_name],
                ensemble_filter=ensemble_filter,
            )
            data_model = data[features]
        else:
            data_model = data

        data_predict = data_model.copy()
        model_result = one_model_predict(
            group_name=model_name,
            model_result=model_result_dict,
            features_values=data_predict,
            log=False,
        )
        data_prepared[result_model_name] = pd.DataFrame(model_result['df'][model_name])

    return data_prepared


def generate_feature_categorical_config(
        data,
        features,
        filename='features_categorical_config',
        min_percent=0.5,
        max_count=20,
        set_notchanged=False,
        other_value='ПРОЧЕЕ',
):
    """Генерация шаблона для файла конфигурации по категориальным признакам."""
    features_reduction = []
    with open(f'{filename}.txt', 'w') as file:
        for feature in features:
            distr = (data[feature].astype(str).value_counts(dropna=False, normalize=True) * 100)
            drop_values = [np.nan, None, 'nan', 'None']
            for drop_value in drop_values:
                if drop_value in distr.index:
                    distr.drop(index=[drop_value], inplace=True)

            distr_len = len(distr)
            mode_value = distr.index[0]
            if distr_len > max_count:
                distr = distr[distr > min_percent]
                if len(distr) < distr_len:
                    features_reduction.append(feature)
                    if isinstance(other_value, str):
                        mode_value = other_value
                    else:
                        mode_value = other_value[feature]

            if set_notchanged:
                feature_replace = {feature.upper(): '_NOTCHANGED_' for feature in distr.sort_index().index}
            else:
                feature_replace = {feature.upper(): feature.upper() for feature in distr.sort_index().index}

            # if is_object_dtype(data[feature]):
            #     encoding = 'WoE_cat_to_num'
            # else:
            #     encoding = 'to_int'

            feature_config = str(
                {'name': feature,
                 'default': mode_value.upper(),
                 'replace': feature_replace,
                 #'encoding': encoding
                }
            ).replace("'", '"') + ',\n'
            file.write(feature_config)
    return features_reduction


def generate_feature_logical_config(
    data,
    features,
    filename='features_logical_config'
):
    """Генерация шаблона для файла конфигурации по логическим признакам."""
    with open(f'{filename}.txt', 'w') as file:
        for feature in features:
            mode_value = str(int(data[feature].mode().iloc[0]))
            feature_config = str({
                'name': feature,
                'default': mode_value,
                'replace': {'0': '0', '0.0': '0', '1': '1', '1.0': '1'},
                'encoding': 'to_int'
            }).replace("'", '"') + ',\n'
            file.write(feature_config)


def generate_feature_numerical_config(
    data,
    features,
    default_type='mode',
    features_min_max=None,
    features_replace=None,
    filename='features_numerical_config'
):
    """Генерация шаблона для файла конфигурации по числовым признакам."""
    with open(f'{filename}.txt', 'w') as file:
        for feature in features:
            if default_type == 'mode':
                default_value = float(data[feature].mode().iloc[0])
            else:
                default_value = float(data[feature].mean())
            feature_config = {
                'name': feature,
                'default': default_value,
                'replace': {'_TYPE_': '_NUM_'}
            }
            if features_replace and feature in features_replace:
                feature_config['replace'][str(features_replace[feature][0])] = str(
                    features_replace[feature][1]
                )
            if features_min_max and feature in features_min_max:
                feature_config['clip'] = {
                    'min_value': features_min_max[feature][0],
                    'max_value': features_min_max[feature][1],
                }
            feature_config = str(feature_config).replace("'", '"') + ',\n'
            file.write(feature_config)
