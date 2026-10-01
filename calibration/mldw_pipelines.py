from loguru import logger
import pandas as pd
import numpy as np
import shutil
import os
from typing import Union, Dict
from itertools import chain
from functools import partial
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from mldataworker.core.pydantic_models import AllModelsConfig
from mldataworker.automl_manager import AutoMLManager
from mldataworker.datasets_manager import DataSetsManager, DSManagerResult
from mldataworker.export_results import ResultExport
from mldataworker.core.utils import ResultPickle
from mldataworker.hyperparameter_tuning import HPTuning
from mldataworker.plots import CompareModelsMetrics, DataframeForPlots, CompareModelsPlot
from mldataworker.core.enums import ModelsParams
from mldataworker.metrics.metrics import BaseMetrics, BaseMetric, gini_updated
from mldataworker.plots import PlotlyWrapper
from mldw_utils import get_woe_bins, calc_pickle_result
from eda import calc_features_shap_importance, plot_importance

__version__ = '1.33.1'


def read_config(models_config):
    with open(models_config, 'r', encoding='utf-8') as f:
        all_models_config = f.read()
    all_models_config = json.loads(all_models_config)
    return all_models_config


def convert_config(all_models_config):
    all_models_config = json.dumps(all_models_config, ensure_ascii=False)
    all_models_config = AllModelsConfig.model_validate_json(all_models_config)
    return all_models_config


def fit(
    models_config,
    manager_type='automl',
    auto_ml_config=None,
    external_config=None,
    extractor=None,
    business_metric=None,
    check_woe=True,
    clear_results=False,
):
    """
    Отличие от функции в AutoMLManager/DataSetsManager.fit_models:
    - Вывод кол-ва признаков.
    - Вывод значений гиперпараметров.
    """
    if isinstance(extractor, type):
        with open(models_config, 'r', encoding='utf-8') as f:
            all_models_config = f.read()
        all_models_config = AllModelsConfig.model_validate_json(all_models_config)
        data_config = all_models_config.data_config
        extractor_object = extractor(data_config=data_config)
    else:
        extractor_object = extractor

    if clear_results:
        # Очистка папки results
        shutil.rmtree(external_config.results_path, ignore_errors=True)
        os.makedirs(external_config.results_path)

    if manager_type == 'automl':
        manager = AutoMLManager(
            auto_ml_config=auto_ml_config,
            models_config=models_config,
            external_config=external_config,
            extractor=extractor_object,
            business_metric=business_metric,
            retro=False,
        )
        # noinspection PyProtectedMember
        manager._init_logger()
    else:
        manager = DataSetsManager(
            config_name=models_config,
            external_config=external_config,
            extractor=extractor_object,
            business_metric=business_metric,
        )

    # noinspection PyProtectedMember
    logger.info(f'Features count: {len(manager._models_configs[0].features)}')

    metrics = manager.fit_models()

    if check_woe:
        _, woe_bins_num = get_woe_bins(manager)
    else:
        woe_bins_num = None
    # noinspection PyProtectedMember
    for model in manager._models_configs:
        hp = None
        if model.wrapper == ModelsParams.catboost:
            hp = manager.get_result()[model.name].model.model.get_params()
        elif model.wrapper == ModelsParams.catboost_over_glm:
            hp = manager.get_result()[model.name].model.model_ctb.get_params()
        if hp is not None:
            logger.info(
                f'Model {model.name} || HP: {hp}'
            )

        if check_woe:
            woe_bins_num_model = woe_bins_num[model.name]
            invalid_woe_features = woe_bins_num_model[woe_bins_num_model['Кол-во групп'] == 1]['Признак'].tolist()
            if invalid_woe_features:
                logger.info(
                    f'Model {model.name} || Invalid WOE features: {invalid_woe_features}'
                )

    return manager, metrics


def predict(
    manager: Union[AutoMLManager, DataSetsManager],
    data,
    calibration=None,
):
    # Алтернативная реализация
    # from mldataworker.core.predict import one_model_predict
    # prediction = one_model_predict(None, manager.get_result()[model.name], data, log=False)
    # prediction = pd.Series(prediction['result'][model.name])
    predictions = {}
    # noinspection PyProtectedMember
    for model in manager._models_configs:
        result = manager.model_predict(data, model.name)
        prediction = pd.concat(
            [result.predictions['train'], result.predictions['test']]
        ).loc[data.index]
        if calibration:
            prediction = calibration.transform(result)
        predictions[model.name] = prediction
    return predictions


def score(manager: Union[AutoMLManager, DataSetsManager]=None, metrics=None, percent=False):
    if metrics is not None:
        scores = metrics
    else:
        scores = {}
        # noinspection PyProtectedMember
        for model in manager._models_configs:
            scores[model.name] = {}
            for sample in ['train', 'test']:
                scores[model.name][sample] = {}
                for metric in ['f1_score', 'mae', 'gini', 'shift']:
                    metrics_full = manager.get_result()[model.name].metrics[sample]['full']
                    if f'{model.name}_model_1' in metrics_full.keys():
                        metrics_full = metrics_full[f'{model.name}_model_1']
                    if metric in metrics_full.keys():
                        score_value = metrics_full[metric]
                        scores[model.name][sample][metric] = float(score_value)

    # Заполнение процента занижения качества на тесте
    for model_name, score_model in scores.items():
        scores[model_name]['test/train'] = {}
        if 'full' in score_model['train']:
            score_model = score_model['train']['full']
            contains_full = True
        else:
            score_model = score_model['train']
            contains_full = False
        for metric_name in score_model.keys():
            if metric_name not in ['threshold']:
                if contains_full:
                    train_value = scores[model_name]['train']['full'][metric_name]
                    test_value = scores[model_name]['test']['full'][metric_name]
                else:
                    train_value = scores[model_name]['train'][metric_name]
                    test_value = scores[model_name]['test'][metric_name]
                if isinstance(train_value, int) or isinstance(train_value, float):
                    if train_value != 0:
                        test_rate = float(abs(train_value - test_value) / train_value)
                    else:
                        test_rate = 1
                    if percent:
                        test_rate = test_rate * 100
                        test_rate = round(test_rate, 1)
                    else:
                        test_rate = round(test_rate, 3)
                    scores[model_name]['test/train'][metric_name] = test_rate

    return scores


# Функция заменена доработанной ClassificationMetric
# def score_with_threshold_tune(
#     manager: Union[AutoMLManager, DataSetsManager],
#     score_func,
#     percent=False,
# ):
#     """Подбор оптимального порога для классификации."""
#     # noinspection PyProtectedMember
#     score_thresholds = {}
#     # noinspection PyProtectedMember
#     for model in manager._models_configs:
#         score_thresholds[model.name] = {}
#         result = manager.get_result()[model.name]
#         y_true_train = result.data_subset.y_train
#         y_true_test = result.data_subset.y_test
#         y_prob_pred_train = result.predictions['train']
#         y_prob_pred_test = result.predictions['test']
#
#         thresholds = np.arange(0.1, 0.9, 0.05)
#         scores = []
#         for threshold in thresholds:
#             y_pred_test = pd.Series(y_prob_pred_test > threshold).astype(int)
#             score_test = score_func(y_true_test, y_pred_test)
#             scores.append([threshold, score_test])
#         scores = pd.DataFrame(scores, columns=['threshold', 'score'])
#         scores.set_index('threshold', drop=True, inplace=True)
#         threshold = round(float(scores['score'].idxmax()), 2)
#         score_thresholds[model.name]['threshold'] = threshold
#
#         y_pred_train = (y_prob_pred_train >= threshold).astype(int)
#         y_pred_test = (y_prob_pred_test >= threshold).astype(int)
#         score_train = round(score_func(y_true_train, y_pred_train), 4)
#         score_test = round(score_func(y_true_test, y_pred_test), 4)
#         score_thresholds[model.name]['train'] = score_train
#         score_thresholds[model.name]['test'] = score_test
#         test_rate = (score_train - score_test) / score_train
#         if percent:
#             test_rate = test_rate * 100
#             test_rate = round(test_rate, 1)
#         else:
#             test_rate = round(test_rate, 3)
#         score_thresholds[model.name]['test/train'] = test_rate
#
#     return score_thresholds

def tuning(
    manager: AutoMLManager,
    parameters_for_optuna: dict,
    trials: int = 100,
):
    """
    Отличие от функции AutoMLManager.hp_tuning:
    - Возможность настройки trials.
    - Вывод используемой метрики.
    """
    # noinspection PyProtectedMember
    logger.info(f'HP Tuning|Metric scores: {manager._hp_tuning_config.metric_score}')

    new_hp = {}

    # noinspection PyProtectedMember
    for model in manager._models_configs:
        new_hp[model.name] = {}
        try:

            parameters_for_optuna_func = None
            if parameters_for_optuna is not None:
                try:
                    parameters_for_optuna_func = parameters_for_optuna[model.name]
                except KeyError:
                    logger.warning('HP_Tune||No parameters for model')
            # noinspection PyProtectedMember
            new_hp[model.name] = HPTuning(data_preprocessor=manager._data_preprocessor,
                                          sampler=manager._hp_tuning_config.sampling,
                                          scoring_fun=manager._hp_tuning_config.metric_score[model.name],
                                          folds_num_for_cv=manager._hp_tuning_config.cv_folds_num,
                                          objective=model.objective,
                                          random_state=manager.config.data_config.separation.random_state
                                          ).best_params(model_name=model.name,
                                                        parameters_for_optuna_func=parameters_for_optuna_func,
                                                        timeout=manager.timeout,
                                                        trials=trials)
            logger.info(new_hp[model.name])

        except Exception as exc:
            manager.errors['HP tuning'] = exc
            logger.info('Returning {}')
    return new_hp


def plot_learning_curve(
    manager: Union[AutoMLManager, DataSetsManager],
    model_name=None,
    params=None,
    figsize=(5, 3),
):
    """
    Построение графика обучения классификации для модели CatBoost.

    Параметры:
        manager: менеджер из которого используется подготовленные данные.
        model_name: имя модели. Если не указано, то используется первая модель.
    """
    if model_name is None:
        model_name = list(manager.get_result().keys())[0]
    result = manager.get_result()[model_name]

    features_numerical = [
        str(f)
        for f in np.sort(result.data_subset.features_numerical)
    ] if result.data_subset.features_numerical is not None else None
    features_categorical = [
        str(f)
        for f in np.sort(result.data_subset.features_categorical)
    ] if result.data_subset.features_categorical is not None else None

    # noinspection PyPep8Naming
    X_train = result.data_subset.X_train[chain(features_numerical, features_categorical)].copy()
    # noinspection PyPep8Naming
    X_test = result.data_subset.X_test[chain(features_numerical, features_categorical)].copy()
    y_train = result.data_subset.y_train.copy()
    y_test = result.data_subset.y_test.copy()

    if params is None:
        params = {
            'random_state': 1,
        }

    if result.model_config.wrapper == ModelsParams.catboost and result.model_config.objective == ModelsParams.poisson:
        catboost_model = CatBoostRegressor
        catboost_objective = "Poisson"

    elif result.model_config.wrapper == ModelsParams.catboost and result.model_config.objective == ModelsParams.gamma:
        catboost_model = CatBoostRegressor
        catboost_objective = "Tweedie:variance_power=1.9999999"

    elif result.model_config.wrapper == ModelsParams.catboost and result.model_config.objective == ModelsParams.binary:
        catboost_model = CatBoostClassifier
        catboost_objective = "Logloss"
    elif (
            result.model_config.wrapper == ModelsParams.catboost and
            result.model_config.objective == ModelsParams.rmsewithuncertainty
    ):
        catboost_model = CatBoostRegressor
        catboost_objective = "RMSEWithUncertainty"
    else:
        catboost_model = CatBoostRegressor
        catboost_objective = "RMSE"

    model = catboost_model(objective=catboost_objective, **params)
    pool = Pool(
        data=X_train[chain(features_numerical, features_categorical)],
        label=y_train,
        cat_features=features_categorical,
        has_header=True,
    )
    model = model.fit(
        pool,
        eval_set=(X_test[chain(features_numerical, features_categorical)], y_test),
        plot=True,
    )

    eval_metrics = model.get_evals_result()
    train_loss = eval_metrics['learn'][catboost_objective]
    val_loss = eval_metrics['validation'][catboost_objective]

    plt.figure(figsize=figsize)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.xlabel('Итерации')
    plt.ylabel(catboost_objective)
    plt.title('График обучения')
    plt.legend()


def save_train_test_keys(
    manager: Union[AutoMLManager, DataSetsManager],
    key_cols,
):
    """Сохранение таблицы с перечнем ключей для обучающей и тестовой выборок."""
    if isinstance(key_cols, str):
        key_cols = [key_cols]
    # noinspection PyProtectedMember
    for model in manager._models_configs:
        train_keys = manager.dataset.loc[manager.get_result()[model.name].data_subset.X_train.index][key_cols]
        test_keys = manager.dataset.loc[manager.get_result()[model.name].data_subset.X_test.index][key_cols]
        train_keys['is_test'] = 0
        test_keys['is_test'] = 1
        train_test_keys = pd.concat([train_keys, test_keys], axis=0)
        filename = f'{manager.group_name}_{model.name}_train_test_keys.parquet'
        # noinspection PyProtectedMember
        filename = os.path.join(manager._external_config.results_path, filename)
        train_test_keys.to_parquet(filename, index=False)


def plot_cohorts(
    manager: Union[AutoMLManager, DataSetsManager],
    config,
    only_test=False,
    samples=100,
    cut_min_value=0.01,
    cut_max_value=0.9,
    use_description_func=False,
    cohort_base='model',
    plotly_params=None,
):
    """
    График когорт по прогнозу без сравнения с предыдущей моделью.
    Отличие от функции ResultExport.plots:
    - Возможность настройки samples, cut_min_value, cut_max_value для каждой модели отдельно.
    """
    res_export = ResultExport(
        ds_manager=manager,
        config=config
    )
    for model_name in res_export.result:
        if isinstance(samples, dict):
            if model_name in samples:
                model_samples = samples[model_name]
            else:
                model_samples = 100
        else:
            model_samples = samples
        if isinstance(cut_min_value, dict):
            if model_name in cut_min_value:
                model_cut_min_value = cut_min_value[model_name]
            else:
                model_cut_min_value = 0.01
        else:
            model_cut_min_value = cut_min_value
        if isinstance(cut_max_value, dict):
            if model_name in cut_max_value:
                model_cut_max_value = cut_max_value[model_name]
            else:
                model_cut_max_value = 0.9
        else:
            model_cut_max_value = cut_max_value
        if use_description_func:
            user_plot_func = partial(
                cohort_plot_description,
                cut_min_value=cut_min_value,
                cut_max_value=cut_max_value,
                samples=samples,
                cohort_base=cohort_base,
                plotly_params=plotly_params,
            )
        else:
            user_plot_func = None
        res_export.plots(
            model_name=model_name,
            plot_type=2,
            only_test=only_test,
            samples=model_samples,
            cut_min_value=model_cut_min_value,
            cut_max_value=model_cut_max_value,
            show=True,
            user_plot_func=user_plot_func,
        )


def plot_features(
    manager: Union[AutoMLManager, DataSetsManager],
    model_name,
    features,
    config,
    only_test=False,
    bins_for_numerical_features=5,
    use_description_func=False,
    features_description=None,
    plotly_params=None,
    use_exposure=False,
    features_numerical_as_cat=None,
    sort_type=None,
    annotation_data=None,
    annotation_feature=None,
    annotation_round=None,
    annotation_mean=False,
):
    """
    График когорт по признакам без сравнения с предыдущей моделью.
    Отличие от функции ResultExport.plots:
    - Возможность настройки bins_for_numerical_features для каждого признака отдельно.
    """
    res_export = ResultExport(
        ds_manager=manager,
        config=config
    )
    for feature in features:
        if isinstance(bins_for_numerical_features, dict):
            if feature in bins_for_numerical_features:
                bins_for_numerical_features_value = bins_for_numerical_features[feature]
            else:
                bins_for_numerical_features_value = 5
        else:
            bins_for_numerical_features_value = bins_for_numerical_features
        if use_description_func:
            user_plot_func = partial(
                feature_plot_description,
                features_description=features_description,
                plotly_params=plotly_params,
                use_exposure=use_exposure,
                features_numerical_as_cat=features_numerical_as_cat,
                sort_type=sort_type,
                annotation_data=annotation_data,
                annotation_feature=annotation_feature,
                annotation_round=annotation_round,
                annotation_mean=annotation_mean,
            )
        else:
            user_plot_func = None
        res_export.plots(
            model_name=model_name,
            plot_type=1,
            features=[feature],
            only_test=only_test,
            bins_for_numerical_features=bins_for_numerical_features_value,
            show=True,
            user_plot_func=user_plot_func,
        )


def plot_times(
        manager,
        date_col,
        period='M',
        only_test=False,
        use_exposure=True,
        plotly_params=None,
):
    results = manager.get_result()
    # noinspection PyProtectedMember
    for model in manager._models_configs:
        y_graph, _, _ = ResultExport.df_for_graphs(
            result=results[model.name],
            features=None,
            use_exposure=use_exposure,
            only_test=only_test
        )

        df = y_graph.copy()
        if 'exposure' not in df.columns:
            df['exposure'] = 1
            logger.error('No exposure for cohort plot||Exposure=1 vector added')

        period = manager.dataset.loc[y_graph.index][date_col].dt.to_period(period)
        y_group = df[['y_true', 'y_prediction', 'exposure']].groupby(period, dropna=False, observed=False).sum()
        y_group['target'] = y_group['y_true'] / y_group['exposure']
        y_group['predict'] = y_group['y_prediction'] / y_group['exposure']

        fig = PlotlyWrapper(model_name=model.name, plotly_params=plotly_params)
        fig.add_figure(
            name='fact',
            x=y_group.index.astype(str),
            y=y_group['target'].values,
        )
        fig.add_figure(
            name='model',
            x=y_group.index.astype(str),
            y=y_group['predict'].values,
        )
        fig.add_figure(
            x=y_group.index.astype(str),
            y=y_group['exposure'].values, type='bar',
            name='exposure',
        )
        figure = fig.prepare_plot(x_name='Время')
        figure.show()


def show_importance(
    manager: Union[AutoMLManager, DataSetsManager],
    model_name,
    features_new=None,
    features_description=None,
    title=None,
    plot=True,
    figsize=(5, 10),
    split=None,

    show_directions=False,
    features_categorical_numerical=None,
    features_categorical_replace=None,
    target_col=None,
    exposure_col=None,
):
    model_result = manager.get_result()[model_name]
    ml_model = model_result.model.model
    features_numerical_model = list(np.sort(model_result.data_subset.features_numerical))
    features_categorical_model = list(np.sort(model_result.data_subset.features_categorical))
    features = list(chain(features_numerical_model, features_categorical_model))
    features = [str(feature) for feature in features]
    data_importance = manager.data_subsets[model_name].X_test[features]
    features_importance = calc_features_shap_importance(
        data=data_importance,
        model=ml_model,
        features_description=features_description,
    )

    if plot:
        if show_directions:
            data = manager.dataset[manager.dataset.index.isin(data_importance.index)]
            relative_features = [
                feature.name
                for feature in manager.get_result()[model_name].model_config.relative_features
            ]
            for feature in relative_features:
                feature_values = pd.concat([
                    manager.get_result()[model_name].data_subset.X_train[feature],
                    manager.get_result()[model_name].data_subset.X_test[feature]
                ])
                data[feature] = feature_values
            cols = chain(features, [target_col])
            if exposure_col:
                cols = chain(cols, [exposure_col])
            data = data[cols].copy()
            features_numerical = None
            # noinspection PyProtectedMember
            for model in manager._models_configs:
                if model.name == model_name:
                    features_numerical = [f.name for f in model.features if '_TYPE_' in f.replace]
        else:
            data = None
            features_numerical = None
        plot_importance(
            features_importance=features_importance,
            importance_col='SHAP',
            features_new=features_new,
            features_description=features_description,
            title=title,
            figsize=figsize,
            split=split,

            show_directions=show_directions,
            data=data,
            features_numerical=features_numerical,
            features_categorical_numerical=features_categorical_numerical,
            features_categorical_replace=features_categorical_replace,
            target_col=target_col,
            exposure_col=exposure_col,
            zero_corr_threshold=0,
            verbose=False,
        )

    return features_importance


def compare(
    new_model_name,
    new_manager=None,
    new_pickle_name=None,
    new_pickle_path='.',
    new_config_name=None,
    new_config_path='.',
    new_ensemble_filter=None,
    old_manager=None,
    old_model_name=None,
    old_pickle_name=None,
    old_pickle_path='.',
    old_config_name=None,
    old_config_path='.',
    old_ensemble_filter=None,
    data=None,
    model_title=None,
    samples=20,
):
    if new_manager is None:
        new_result = calc_pickle_result(
            data=data,
            models_names=[new_model_name],
            pickle_name=new_pickle_name,
            config_name=new_config_name,
            pickle_path=new_pickle_path,
            config_path=new_config_path,
            ensemble_filter=new_ensemble_filter,
            add_metric_suffix=True,
            result_models_names=None,
        )
    else:
        new_result = new_manager.get_result()

    if old_model_name is None:
        old_model_name = new_model_name
    if old_manager is None:
        old_result = calc_pickle_result(
            data=data,
            models_names=[old_model_name],
            pickle_name=old_pickle_name,
            config_name=old_config_name,
            pickle_path=old_pickle_path,
            config_path=old_config_path,
            ensemble_filter=old_ensemble_filter,
            result_models_names=None,
        )
    else:
        old_result = old_manager.get_result()
    old_result[new_model_name] = old_result.pop(old_model_name)

    compare_metrics = CompareModelsMetrics(
        result1=new_result,
        result2=old_result,
        show=False,
    ).compare_metrics(model_name=new_model_name, only_main=True)

    y_graph, features_categorical, features_numerical = DataframeForPlots().df_for_plots(
        result=new_result[new_model_name],
        use_exposure=True,
        only_test=True,
    )
    y_graph2, features_categorical2, features_numerical2 = DataframeForPlots().df_for_plots(
        result=old_result[new_model_name],
        use_exposure=True,
        only_test=True,
    )

    if model_title is None:
        model_title = new_model_name
    compare_plots = CompareModelsPlot(
        model_name=model_title,
        df1=y_graph,
        df2=y_graph2,
        features_categorical=features_categorical,
        features_numerical=features_numerical,
        show=False
    ).make(
        plot_type=2,
        samples=samples,
        cohort_base='model1',
    )

    return compare_metrics, compare_plots


def ensemble_score(managers, classification=False):
    scores = {}
    # noinspection PyProtectedMember
    for model in managers[0]._models_configs:
        model_name = model.name
        scores[model_name] = {}
        for sample in ['train', 'test']:
            scores[model_name][sample] = {}
            y_true_managers = []
            y_pred_managers = []
            exposure_managers = []
            for manager in managers:
                if sample == 'train':
                    y_true_managers.append(manager.get_result()[model_name].data_subset.y_train)
                else:
                    y_true_managers.append(manager.get_result()[model_name].data_subset.y_test)
                y_pred_managers.append(manager.get_result()[model_name].predictions[sample])
                if sample == 'train':
                    exposure_train = manager.get_result()[model_name].data_subset.exposure_train
                    if exposure_train is not None:
                        exposure_managers.append(exposure_train)
                else:
                    exposure_test = manager.get_result()[model_name].data_subset.exposure_test
                    if exposure_test is not None:
                        exposure_managers.append(exposure_test)
            y_true = pd.concat(y_true_managers)
            y_pred = pd.concat(y_pred_managers)
            if exposure_managers:
                exposure = pd.concat(exposure_managers)
            else:
                exposure = None

            if classification:
                scores[model_name][sample]['roc_auc'] = round(
                    roc_auc_score(y_true, y_pred, sample_weight=exposure),
                    4
                )
                scores[model_name][sample]['gini'] = round(
                    float(gini_updated(y_true, y_pred, sample_weight=exposure)),
                    4
                )
            else:
                model_metrics = BaseMetrics(y_true, y_pred, exposure).calculate_metric()
                model_metrics = {key: value for key, value in model_metrics.items() if key in ['mae', 'gini', 'shift']}
                scores[model_name][sample] = model_metrics

    return scores


# noinspection PyUnusedLocal
def cohort_plot_description(
    y_graph,
    model_name,
    features_categorical,
    features_numerical,
    bins,
    cut_min_value=0.01,
    cut_max_value=0.9,
    samples=100,
    cohort_base='model',
    plotly_params=None,
):
    samples = samples
    df = y_graph.copy()
    if 'exposure' not in df.columns:
        df['exposure'] = 1
        logger.error('No exposure for cohort plot||Exposure=1 vector added')
    df["predict"] = df['y_prediction'] / df['exposure']
    if cohort_base == 'model':
        column_to_group = "predict"
        column_for_target_count = 'y_true'
        names = ['fact', 'model']
    elif cohort_base == 'fact':
        names = ['model', 'fact']
        column_to_group = "y_true"
        column_for_target_count = 'y_prediction'

    else:
        logger.error('Unknown cohort base!||Chosen Model type ')
        column_to_group = "predict"
        column_for_target_count = 'y_true'
        names = ['fact', 'model']
    df["predict_gr"] = round(df[column_to_group] * samples) / samples
    df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                          upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
    if len(df["predict_gr"].unique()) > 20 and samples == 100.0:
        for i in range(8):
            if len(df["predict_gr"].unique()) < 20:
                break
            samples = samples / 5
            df["predict_gr"] = round(df[column_to_group] * samples) / samples
            df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                  upper=df["predict_gr"].quantile(cut_max_value), inplace=True)
    elif len(df["predict_gr"].unique()) < 5 and cut_max_value == 0.9:
        for i in range(8):
            if len(df["predict_gr"].unique()) >= 5:
                break
            cut_max_value = cut_max_value * 1.1
            if cut_max_value > 1: cut_max_value = 1
            samples = samples * 2
            df["predict_gr"] = round(df[column_to_group] * samples) / samples
            df["predict_gr"].clip(lower=df["predict_gr"].quantile(cut_min_value),
                                  upper=df["predict_gr"].quantile(cut_max_value), inplace=True)

    y_test_freq_gr = (
        df
        [[column_for_target_count, 'exposure', "predict_gr"]]
        .groupby("predict_gr")
        .sum()
        .reset_index()
    )
    y_test_freq_gr["predict_gr_"] = y_test_freq_gr["predict_gr"]
    y_test_freq_gr['target'] = y_test_freq_gr[column_for_target_count] / y_test_freq_gr['exposure']
    cohort_plot = PlotlyWrapper(model_name=model_name, plotly_params=plotly_params)
    cohort_plot.add_figure(x=y_test_freq_gr["predict_gr"].values,
                           y=y_test_freq_gr["target"].values, name=names[0])
    cohort_plot.add_figure(name=names[1],
                           x=y_test_freq_gr["predict_gr"].values,
                           y=y_test_freq_gr["predict_gr_"].values, )
    cohort_plot.add_figure(name="exposure",
                           x=y_test_freq_gr["predict_gr"].values,
                           y=y_test_freq_gr['exposure'].values, type='bar')
    figure = cohort_plot.prepare_plot(x_name=plotly_params['x_name'])
    figure.show()


def feature_plot_description(
        y_graph,
        model_name,
        features_categorical,
        features_numerical,
        bins,
        features_description=None,
        plotly_params=None,
        use_exposure=False,
        features_numerical_as_cat=None,
        sort_type=None,
        annotation_data=None,
        annotation_feature=None,
        annotation_round=None,
        annotation_mean=False,
):
    # noinspection PyPep8Naming
    X_test = y_graph
    column_exposure = 'exposure'
    if not use_exposure: X_test[column_exposure] = 1
    # noinspection PyBroadException
    try:
        X_test[column_exposure]
    except:
        X_test[column_exposure] = 1
        logger.error('No exposure in model||Added exposure = 1')
    column_claims_count = 'y_true'
    bins_test_fact = {}
    bins_test_pred = {}
    features = features_numerical + features_categorical
    group_cols = [column_claims_count, column_exposure]
    if annotation_data is not None:
        annotation_data = annotation_data.copy()
        assert pd.Series(annotation_data.index != X_test.index).sum() == 0
        if isinstance(annotation_data, pd.Series):
            annotation_data = pd.DataFrame(annotation_data)
            annotation_data.columns = ['annotation']
        else:
            annotation_data.rename(columns={annotation_feature: 'annotation'}, inplace=True)
        annotation_data = annotation_data[['annotation']]
        X_test = pd.concat([X_test, annotation_data], axis=1)
        group_cols += ['annotation']
    if features_numerical_as_cat is None:
        features_numerical_as_cat = []
    for feature in features:
        if feature in features_categorical or feature in features_numerical_as_cat:
            feature_vals = X_test[feature]
            bins_test_fact[feature] = X_test[group_cols].groupby(feature_vals,
                                                                 dropna=False,
                                                                 observed=False).sum()
            bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                              bins_test_fact[feature][column_exposure]
            bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(feature_vals, dropna=False,
                                                                                        observed=False).sum()
            bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                column_exposure]
        else:
            feature_vals = X_test[feature]
            n_bins = bins
            if feature_vals.nunique() > n_bins and bins is not None:
                breakpoints = np.arange(0, n_bins + 1) / n_bins * 100
                breakpoints = [np.percentile(X_test[feature], bp) for bp in breakpoints]
                breakpoints[0] = breakpoints[0] - 0.1
            else:
                breakpoints = bins

            bins_test_fact[feature] = X_test[group_cols].groupby(
                pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                observed=False).sum()
            bins_test_fact[feature]["freq"] = bins_test_fact[feature][column_claims_count] / \
                                              bins_test_fact[feature][column_exposure]
            bins_test_pred[feature] = X_test[['y_prediction', column_exposure]].groupby(
                pd.cut(X_test[feature], breakpoints, duplicates="drop"),
                observed=False).sum()
            bins_test_pred[feature]["freq"] = bins_test_pred[feature]['y_prediction'] / bins_test_pred[feature][
                column_exposure]
        if annotation_data is not None and annotation_mean:
            bins_test_fact[feature]['annotation'] = (
                    bins_test_fact[feature]['annotation'] / bins_test_fact[feature][column_exposure]
            )
        if annotation_data is not None and annotation_round is not None:
            bins_test_fact[feature]['annotation'] = bins_test_fact[feature]['annotation'].round(annotation_round)

        if sort_type == 'fact_asc':
            bins_test_fact[feature] = bins_test_fact[feature].sort_values("freq").copy()
            bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()
        elif sort_type == 'fact_desc':
            bins_test_fact[feature] = bins_test_fact[feature].sort_values("freq", ascending=False).copy()
            bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()
        elif sort_type == 'pred_asc':
            bins_test_pred[feature] = bins_test_pred[feature].sort_values("freq").copy()
            bins_test_fact[feature] = bins_test_fact[feature].loc[bins_test_pred[feature].index].copy()
        elif sort_type == 'pred_desc':
            bins_test_pred[feature] = bins_test_pred[feature].sort_values("freq", ascending=False).copy()
            bins_test_fact[feature] = bins_test_fact[feature].loc[bins_test_pred[feature].index].copy()
        elif sort_type == 'exp_asc':
            bins_test_fact[feature] = bins_test_fact[feature].sort_values(column_exposure).copy()
            bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()
        elif sort_type == 'exp_desc':
            bins_test_fact[feature] = bins_test_fact[feature].sort_values(column_exposure, ascending=False).copy()
            bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()
        elif sort_type in ['index_asc', 'index_desc']:
            index = bins_test_fact[feature].index.astype(str)
            if len(index) > 0 and '(' in index[0] and ',' in index[0] and ']' in index[0]:
                bins_test_fact[feature]['sort'] = [
                    float(s.split(',')[0][1:].replace('-inf', '-1e10'))
                    for s in
                    index
                ]
            else:
                bins_test_fact[feature]['sort'] = bins_test_fact[feature].index
            if sort_type == 'index_asc':
                bins_test_fact[feature] = bins_test_fact[feature].sort_values('sort').copy()
                bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()
            else:
                bins_test_fact[feature] = bins_test_fact[feature].sort_values('sort', ascending=False).copy()
                bins_test_pred[feature] = bins_test_pred[feature].loc[bins_test_fact[feature].index].copy()

        fig = PlotlyWrapper(model_name=model_name, plotly_params=plotly_params)
        fig.add_figure(type='bar', name="exposure",
                       x=bins_test_fact[feature].index.astype(str),
                       y=bins_test_fact[feature][column_exposure].values)
        fig.add_figure(name="fact",
                       x=bins_test_fact[feature].index.astype(str),
                       y=bins_test_fact[feature]["freq"].values)
        fig.add_figure(
            name="model",
            x=bins_test_pred[feature].index.astype(str),
            y=bins_test_pred[feature]["freq"].values)
        if annotation_data is not None:
            text = bins_test_fact[feature]['annotation'].values.astype(str)
            fig.figure.update_traces(text=text, selector={'type': 'bar'})

        if features_description:
            x_name = features_description[feature]
        else:
            x_name = feature
        figure = fig.prepare_plot(x_name=x_name)
        if plotly_params and 'x_range' in plotly_params:
            figure.update_xaxes(range=plotly_params['x_range'])
        if plotly_params and 'y_range' in plotly_params:
            figure.update_yaxes(range=plotly_params['y_range'], secondary_y=False)
        figure.show()


class ClassificationCalibration:
    def __init__(self, manager=None):
        self._manager = manager
        self._calibrated_model = None
        self._features_categorical = None
        self._features_numerical = None

    def fit_transform(self):
        # noinspection PyProtectedMember
        for model in self._manager._models_configs:
            result = self._manager.get_result()[model.name]
            self._features_numerical = [
                str(f)
                for f in np.sort(result.data_subset.features_numerical)
            ] if result.data_subset.features_numerical is not None else None
            self._features_categorical = [
                str(f)
                for f in np.sort(result.data_subset.features_categorical)
            ] if result.data_subset.features_categorical is not None else None
            exposure_train = (
                result.data_subset.exposure_train
                if result.data_subset.exposure_train is not None
                else None
            )
            # noinspection PyPep8Naming
            X_train = result.data_subset.X_train[chain(self._features_numerical, self._features_categorical)]
            # noinspection PyPep8Naming
            X_test = result.data_subset.X_test[chain(self._features_numerical, self._features_categorical)]

            self._calibrated_model = CalibratedClassifierCV(
                estimator=result.model.model,
                method='isotonic',
                cv='prefit',
            )
            self._calibrated_model.fit(
                X_train,
                result.data_subset.y_train,
                cat_features=self._features_categorical,
                has_header=True,
                weight=exposure_train,
            )

            predictions_train_clb = pd.Series(self._calibrated_model.predict_proba(X_train)[:, 1], index=X_train.index)
            predictions_test_clb = pd.Series(self._calibrated_model.predict_proba(X_test)[:, 1], index=X_test.index)

            brier_score_train = round(brier_score_loss(result.data_subset.y_train, result.predictions['train']), 5)
            brier_score_test = round(brier_score_loss(result.data_subset.y_test, result.predictions['test']), 5)
            logger.info(f'Model {model.name} || Before brier score train: {brier_score_train}')
            logger.info(f'Model {model.name} || Before brier score test: {brier_score_test}')

            result.load_predictions(predictions_train_clb, 'train')
            result.load_predictions(predictions_test_clb, 'test')

            brier_score_train = round(brier_score_loss(result.data_subset.y_train, result.predictions['train']), 5)
            brier_score_test = round(brier_score_loss(result.data_subset.y_test, result.predictions['test']), 5)
            logger.info(f'Model {model.name} || After brier score train: {brier_score_train}')
            logger.info(f'Model {model.name} || After brier score test: {brier_score_test}')

    def transform(self, result=None, data=None):
        # noinspection PyPep8Naming
        if result:
            # noinspection PyPep8Naming
            X = pd.concat(
                [result.data_subset.X_train, result.data_subset.X_test]
            ).loc[result.data_subset.X.index]
        else:
            # noinspection PyPep8Naming
            X = data
        # noinspection PyPep8Naming
        X = X[chain(self._features_numerical, self._features_categorical)]
        return pd.Series(self._calibrated_model.predict_proba(X)[:, 1], index=X.index)

    def save_results(self):
        saving_start_time = datetime.now()
        result_pickle_name = ResultPickle().generate_name(
            self._manager.group_name + '_cc', saving_start_time
        )
        results = {
            'model': self._calibrated_model,
            'features_categorical': self._features_categorical,
            'features_numerical': self._features_numerical,
        }
        # noinspection PyProtectedMember
        with open(os.path.join(self._manager._external_config.results_path, result_pickle_name), 'wb') as f:
            # noinspection PyTypeChecker
            pickle.dump(results, f)

    def load_result(self, pickle_name, pickle_path='.'):
        if not '.pickle' in pickle_name:
            pickle_name = pickle_name + '.pickle'
        with open(os.path.join(pickle_path, pickle_name), 'rb') as f:
            results = pickle.load(f)
        self._calibrated_model = results['model']
        self._features_categorical = results['features_categorical']
        self._features_numerical = results['features_numerical']


class ClassificationMetric(BaseMetric):
    """Расчет метрик для классификации."""
    def __init__(
        self,
        score_func=f1_score,
        plot_matrix_train=True,
        plot_matrix_test=True,
        cmap=None,
        xticklabels=None,
        yticklabels=None,
    ):
        self._score_func = score_func
        self._plot_matrix_train = plot_matrix_train
        self._plot_matrix_test = plot_matrix_test
        self._cmap = cmap
        self._xticklabels = xticklabels
        self._yticklabels = yticklabels

    def calculate_metric(self, results: Dict[str, DSManagerResult], threshold=None, thresholds=None) -> dict:
        metrics = {}
        for model_name, result in results.items():
            thresholds_model = thresholds
            if thresholds_model is None:
                thresholds_start = result.data_subset.y_train.quantile(0.001)
                thresholds_end = result.data_subset.y_train.quantile(0.999)
                thresholds_step = (thresholds_end - thresholds_start) / 30
                thresholds_model = np.arange(thresholds_start, thresholds_end, thresholds_step)
            threshold_train = threshold
            metrics[model_name] = {}
            for sample in ['train', 'test']:
                # Расчет общих метрик
                metrics[model_name][sample] = {}
                if sample == 'train':
                    y_true = result.data_subset.y_train
                    exposure = result.data_subset.exposure_train
                else:
                    y_true = result.data_subset.y_test
                    exposure = result.data_subset.exposure_test

                y_pred_prob = result.predictions[sample]
                if exposure is not None:
                    y_pred_prob = y_pred_prob * exposure

                metrics[model_name][sample]['roc_auc'] = round(roc_auc_score(y_true, y_pred_prob, sample_weight=exposure), 4)
                metrics[model_name][sample]['gini'] = round(
                    float(gini_updated(y_true, y_pred_prob, sample_weight=exposure)),
                    4
                )
                metrics[model_name][sample]['shift'] = round(float(y_pred_prob.sum() / (y_true.sum())), 4)

                # Расчет метрик с подбором оптимального порога
                if sample == 'train' and threshold_train is None:
                    scores = []
                    for threshold_current in thresholds_model:
                        y_pred = pd.Series(y_pred_prob > threshold_current).astype(int)
                        score_test = self._score_func(y_true, y_pred)
                        scores.append([threshold_current, score_test])
                    scores = pd.DataFrame(scores, columns=['threshold', 'score'])
                    scores.set_index('threshold', drop=True, inplace=True)
                    threshold_train = round(float(scores['score'].idxmax()), 5)
                metrics[model_name][sample]['threshold'] = threshold_train

                y_pred = pd.Series(y_pred_prob > threshold_train).astype(int)
                metrics[model_name][sample]['f1_score'] = round(f1_score(y_true, y_pred, sample_weight=exposure), 4)
                metrics[model_name][sample]['precision_score'] = round(
                    precision_score(y_true, y_pred, sample_weight=exposure),
                    4
                )
                metrics[model_name][sample]['recall_score'] = round(
                    recall_score(y_true, y_pred, sample_weight=exposure),
                    4
                )

                # Расчет матрицы ошибок
                matrix = confusion_matrix(y_true, y_pred, sample_weight=exposure)
                metrics[model_name][sample]['matrix'] = matrix
                if (sample == 'train' and self._plot_matrix_train) or (sample == 'test' and self._plot_matrix_test):
                    matrix_plot = ConfusionMatrixDisplay(confusion_matrix=matrix).plot(cmap=self._cmap)
                    matrix_plot.ax_.set_title(sample)
                    matrix_plot.ax_.set_xlabel('Прогноз')
                    matrix_plot.ax_.set_ylabel('Факт')
                    if self._xticklabels:
                        matrix_plot.ax_.set_xticklabels(self._xticklabels)
                    if self._yticklabels:
                        matrix_plot.ax_.set_yticklabels(self._yticklabels)

        return metrics
