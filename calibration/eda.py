import numpy as np
import pandas as pd
# noinspection PyUnresolvedReferences
import phik
from catboost import EFeaturesSelectionAlgorithm, EShapCalcType
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from shap import TreeExplainer
import matplotlib.pyplot as plt

__version__ = '1.20.0'


def calc_features_nulls(data, features, min_percent=None, features_description=None):
    filling = np.round(data[features].notnull().sum() / len(data) * 100, 5)
    filling = pd.DataFrame(filling, columns=['Процент'])
    filling['Признак'] = filling.index
    if features_description is not None:
        filling['Описание'] = filling['Признак'].replace(features_description)
    filling.sort_values(['Процент', 'Признак'], ascending=True, inplace=True)
    filling.reset_index(drop=True, inplace=True)
    if 'Описание' in filling.columns:
        filling = filling[['Признак', 'Описание', 'Процент']].copy()
    else:
        filling = filling[['Признак', 'Процент']].copy()
    nulls = filling[filling['Процент'] < 100].copy()
    not_nulls = filling[filling['Процент'] == 100].copy()
    if min_percent:
        nulls_threshold = filling[filling['Процент'] < min_percent].copy()
        return nulls, not_nulls, nulls_threshold
    return nulls, not_nulls


def calc_features_not_zeros(data, features, min_percent=None, features_description=None):
    not_zeros = np.round((data[features] > 0).sum() / len(data) * 100, 2)
    not_zeros = pd.DataFrame(not_zeros, columns=['Процент'])
    not_zeros['Признак'] = not_zeros.index
    if features_description is not None:
        not_zeros['Описание'] = not_zeros['Признак'].replace(features_description)
    not_zeros.sort_values(['Процент', 'Признак'], ascending=True, inplace=True)
    not_zeros.reset_index(drop=True, inplace=True)
    if 'Описание' in not_zeros.columns:
        not_zeros = not_zeros[['Признак', 'Описание', 'Процент']].copy()
    else:
        not_zeros = not_zeros[['Признак', 'Процент']].copy()
    if min_percent:
        not_zeros_threshold = not_zeros[not_zeros['Процент'] < min_percent].copy()
        return not_zeros, not_zeros_threshold
    return not_zeros


def calc_features_nunique(data, features, features_description=None):
    nunique = data[features].nunique()
    nunique = pd.DataFrame(nunique, columns=['Кол-во'])
    nunique['Признак'] = nunique.index
    if features_description is not None:
        nunique['Описание'] = nunique['Признак'].replace(features_description)
    nunique.sort_values(['Кол-во', 'Признак'], ascending=False, inplace=True)
    nunique.reset_index(drop=True, inplace=True)
    if 'Описание' in nunique.columns:
        nunique = nunique[['Признак', 'Описание', 'Кол-во']].copy()
    else:
        nunique = nunique[['Признак', 'Кол-во']].copy()
    return nunique


def calc_features_min_max(
    data,
    features,
    features_description=None,
    exclude_values=None,
):
    if exclude_values:
        data = data.copy()
        for exclude_value in exclude_values:
            data.replace(exclude_value, np.nan, inplace=True)

    min_max = pd.concat([data[features].min(), data[features].max()], axis=1)
    min_max.columns = ['Мин', 'Макс']
    min_max['Признак'] = min_max.index
    if features_description is not None:
        min_max['Описание'] = min_max['Признак'].replace(features_description)
    min_max.reset_index(drop=True, inplace=True)
    if 'Описание' in min_max.columns:
        min_max = min_max[['Признак', 'Описание', 'Мин', 'Макс']].copy()
    else:
        min_max = min_max[['Признак', 'Мин', 'Макс']].copy()
    return min_max


def get_features_cat_distr(data, features, skip_count=30, features_description=None, sort_index=False, normalize=True):
    features_distr = []
    features_skip = []
    for feature in features:
        if data[feature].nunique() <= skip_count:
            if normalize:
                distr = np.round(data[feature].value_counts(dropna=False, normalize=True) * 100, 5)
            else:
                distr = data[feature].value_counts(dropna=False, normalize=False)
            if sort_index:
                distr.sort_index(inplace=True)
            if features_description is not None:
                distr.index.name = f'{distr.index.name} {features_description[distr.index.name]}'
            features_distr.append(distr)
        else:
            features_skip.append(feature)
    if len(features) == 1:
        return features_distr[0]
    return features_distr, features_skip


def plot_features_num_distr(
    data,
    features_hist_config,
    features_description=None,
    features_min_max=None,
    bins=30,
    figsize=(7, 2),
):
    for feature_config in features_hist_config:
        if isinstance(feature_config, str):
            feature = feature_config
            data_hist = data[feature].copy()
            if features_min_max and feature in features_min_max:
                data_hist.clip(lower=features_min_max[feature][0], upper=features_min_max[feature][1], inplace=True)
            title = feature
            if features_description is not None:
                title = f'{feature} {features_description[title]}'
            feature_bins = bins
        else:
            feature = feature_config['feature']
            data_hist = data[feature]
            title = feature_config['title']
            if 'min' in feature_config:
                data_hist = data_hist[data_hist >= feature_config['min']]
            if 'max' in feature_config:
                data_hist = data_hist[data_hist <= feature_config['max']]
            if 'bins' in feature_config:
                feature_bins = feature_config['bins']
            else:
                feature_bins = bins
        data_hist.hist(bins=feature_bins, figsize=figsize)
        plt.xlabel('Значения')
        plt.ylabel('Кол-во строк')
        plt.title(title)
        plt.show()


def plot_features_time_distr(
    data,
    features,
    period,
    date_col,
    features_description=None,
    normalize=False,
    normalize_col=None,
    agg_func=None,
    figsize=(7, 2),
    xlabel=None,
    ylabel=None,
    ylim=None,
):
    period = data[date_col].dt.to_period(period)
    if agg_func is None:
        agg_func = 'count'
    for feature in features:
        time_distr = data.groupby(period)[feature].agg(agg_func)
        if normalize:
            time_distr = time_distr / data.groupby(period)[normalize_col].count() * 100
        if features_description is not None:
            title = f'{feature} {features_description[feature]}'
        else:
            title = feature
        time_distr.plot(figsize=figsize)
        if xlabel:
            plt.xlabel(xlabel)
        else:
            plt.xlabel('Время')
        if ylabel:
            plt.ylabel(ylabel)
        elif normalize:
            plt.ylabel('Процент строк')
        else:
            plt.ylabel('Кол-во строк')
        if ylim:
            plt.ylim(*ylim)
        else:
            plt.ylim(0)
        plt.title(title)
        plt.show()


def plot_features_target(
        data,
        features_categorical,
        features_numerical,
        target_col,
        exposure_col=None,
        numerical_bins=5,
        features_description=None,
        normalize=False,
        agg_func=None,
        min_exposure_percent=None,
        figsize=(7, 3),
        ylabel=None,
        ylim=None,
        ticks_len_rotation45=30,
        ticks_len_rotation90=100,
        sort_type=None,
        max_count=None,
):
    drop_exposure_col = False
    if not exposure_col:
        exposure_col = 'exposure'
        data[exposure_col] = 1
        drop_exposure_col = True

    """Распределение таргета по значениям признаков."""
    if agg_func is None:
        agg_func = 'sum'
    for feature in features_categorical + features_numerical:
        if feature in features_categorical:
            group = data.groupby(data[feature], observed=False)
        else:
            group = data.groupby(pd.cut(data[feature].dropna(), bins=numerical_bins), observed=False)
        target_distr = group[target_col].agg(agg_func)
        if normalize:
            target_distr = target_distr / group[target_col].count() * 100

        exposure_distr = group[exposure_col].sum()
        exposure_distr_percent = exposure_distr / len(data) * 100

        if len(target_distr) > 20 and min_exposure_percent:
            exposure_distr = exposure_distr[exposure_distr_percent > min_exposure_percent]
            target_distr = target_distr[exposure_distr_percent > min_exposure_percent]

        drop_index = target_distr[target_distr.isnull()].index
        target_distr.drop(index=drop_index, inplace=True)
        exposure_distr.drop(index=drop_index, inplace=True)

        if sort_type == 'asc':
            target_distr.sort_values(ascending=True, inplace=True)
            exposure_distr = exposure_distr.loc[target_distr.index].copy()
        elif sort_type == 'desc':
            target_distr.sort_values(ascending=False, inplace=True)
            exposure_distr = exposure_distr.loc[target_distr.index].copy()
        elif sort_type == 'index_asc':
            target_distr.sort_index(ascending=True, inplace=True)
            exposure_distr = exposure_distr.loc[target_distr.index].copy()
        elif sort_type == 'index_desc':
            target_distr.sort_index(ascending=False, inplace=True)
            exposure_distr = exposure_distr.loc[target_distr.index].copy()
        if max_count:
            target_distr = target_distr.iloc[:max_count].copy()
            exposure_distr = exposure_distr.loc[target_distr.index].copy()

        target_distr.index = target_distr.index.astype(str)
        exposure_distr.index = exposure_distr.index.astype(str)

        if features_description is not None:
            title = f'{feature} {features_description[feature]}'
        else:
            title = feature

        figure, ax = plt.subplots(figsize=figsize)
        ax.plot(target_distr, label='Признак')
        if ylabel:
            ax.set_ylabel(ylabel)
        elif normalize:
            ax.set_ylabel('% строк')
        else:
            ax.set_ylabel('Кол-во строк')
        if ylim:
            ax.set_ylim(*ylim)
        ticks_len = len(''.join(target_distr.index))
        if ticks_len > ticks_len_rotation45:
            ax.tick_params(labelrotation=45)
        if ticks_len > ticks_len_rotation90:
            ax.tick_params(labelrotation=90)
        second_ax = ax.twinx()
        second_ax.bar(exposure_distr.index, exposure_distr, color='orange', alpha=0.7, label='Экспозиция')
        second_ax.set_ylabel('Экспозиция')
        figure.legend()
        ax.set_title(title)
        figure.show()

    if drop_exposure_col:
        data.drop(columns=[exposure_col], inplace=True)


def calc_features_singularity(
    data,
    features,
    features_description=None,
    min_percent=1,
    exclude_values=None,
):
    features_singularity = []
    for feature in features:
        distr = data[feature].value_counts(dropna=False, normalize=True) * 100
        distr.index = distr.index.get_level_values(0)
        if np.nan in distr.index:
            distr.drop(index=[np.nan], inplace=True)
        if None in distr.index:
            distr.drop(index=[None], inplace=True)
        if exclude_values:
            for exclude_value in exclude_values:
                if exclude_value in distr.index:
                    distr.drop(index=[exclude_value], inplace=True)
        allow_fraction_count = len(distr[distr > min_percent])
        if allow_fraction_count <= 1:
            features_singularity.append(feature)

    features_singularity = pd.DataFrame(features_singularity, columns=['Признак'])
    if features_description is not None:
        features_singularity['Описание'] = features_singularity['Признак'].replace(features_description)

    return features_singularity


def calc_features_singularity_clear(
    data,
    features,
    features_description=None,
    max_percent=99,
    exclude_values=None,
):
    features_singularity = []
    for feature in features:
        values = data[feature].dropna()
        if exclude_values:
            for exclude_value in exclude_values:
                drop_index = values[values == exclude_value].index
                values.drop(index=drop_index, inplace=True)
        distr = values.value_counts(normalize=True) * 100
        distr.index = distr.index.get_level_values(0)
        forbid_fraction_count = len(distr[distr > max_percent])
        if forbid_fraction_count > 0:
            features_singularity.append(feature)

    features_singularity = pd.DataFrame(features_singularity, columns=['Признак'])
    if features_description is not None:
        features_singularity['Описание'] = features_singularity['Признак'].replace(features_description)

    return features_singularity


def get_features_values_singularity(data, features, min_percent=1, features_description=None):
    """Получение значений каждого признака с долей меньше заданной."""
    values_singularity = []
    for feature in features:
        distr = np.round(data[feature].value_counts(dropna=False, normalize=True) * 100, 5)
        if features_description is not None:
            distr.index.name = f'{distr.index.name} {features_description[distr.index.name]}'
        if np.nan in distr.index:
            distr.drop(index=[np.nan], inplace=True)
        if None in distr.index:
            distr.drop(index=[None], inplace=True)
        distr = distr[distr < min_percent].copy()
        values_singularity.append(distr)

    if len(features) == 1:
        return values_singularity[0]
    return values_singularity


def calc_features_stability(
    data,
    features,
    features_categorical,
    target_col,
    model_type,
    objective,
    scoring,
    exposure_col=None,
    features_default=None,
    features_description=None,
    params=None,
    cv_diff_threshold=None,
    inplace=False,
    verbose=False,
):
    data, y, exp_filter, model_object, params = _model_prepare(
        data,
        target_col,
        model_type,
        exposure_col,
        features_default,
        params,
        inplace,
        verbose,
    )

    if not isinstance(scoring, str):
        scoring = make_scorer(scoring)
        if verbose:
            print('Using make_scorer')

    features_stability = []
    for feature in features:
        # noinspection PyPep8Naming
        X = data[exp_filter][[feature]]
        if feature in features_categorical:
            cat_features = [feature]
            X[feature] = X[feature].astype(str)
        else:
            cat_features = []
        model = model_object(objective=objective, cat_features=cat_features, verbose=False, **params)
        scores = cross_val_score(model, X, y, cv=3, scoring=scoring)
        max_score = np.max(scores)
        min_score = np.min(scores)
        diff = abs(max_score / min_score - 1)
        features_stability.append([feature, max_score, min_score, diff])
        if verbose:
            if len(cat_features) > 0:
                feature_type = 'Категориальный'
            else:
                feature_type = 'Числовой'
            print(feature, round(max_score, 5), round(min_score, 5), round(diff, 5), feature_type)

    features_stability = pd.DataFrame(features_stability, columns=['Признак', 'Макс', 'Мин', 'Разница'])
    features_stability.sort_values('Разница', ascending=False, inplace=True)
    if features_description is not None:
        features_stability['Описание'] = features_stability['Признак'].replace(features_description)
        features_stability = features_stability[['Признак', 'Описание', 'Макс', 'Мин', 'Разница']].copy()
    if cv_diff_threshold is not None:
        features_stability['Стабильный'] = True
        features_stability.loc[features_stability['Разница'] > cv_diff_threshold, 'Стабильный'] = False

    return features_stability


def calc_features_correlation(
    data,
    features,
    features_numerical=None,
    features_description=None,
    method='phik',
    corr_threshold=None,
):
    if method == 'phik':
        corr_matrix = data[features].phik_matrix(interval_cols=features_numerical)
    else:
        corr_matrix = data[features].corr(method='spearman')
    corr_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    if corr_threshold is not None:
        features_corr = []
        for index, row in corr_matrix.iterrows():
            for col in corr_matrix.columns:
                if row[col] > corr_threshold:
                    features_corr.append([index, col])
        features_corr = pd.DataFrame(features_corr, columns=['Признак1', 'Признак2'])
        if features_description is not None:
            features_corr['Описание1'] = features_corr['Признак1'].replace(features_description)
            features_corr['Описание2'] = features_corr['Признак2'].replace(features_description)
            features_corr = features_corr[['Признак1', 'Описание1', 'Признак2', 'Описание2']].copy()
    else:
        features_corr = None
    return corr_matrix, features_corr


def calc_features_importance(
    data,
    features,
    features_categorical,
    target_col,
    model_type,
    objective,
    exposure_col=None,
    features_default=None,
    add_random=False,
    features_description=None,
    params=None,
    inplace=False,
    verbose=False,
):
    data, y, exp_filter, model_object, params = _model_prepare(
        data,
        target_col,
        model_type,
        exposure_col,
        features_default,
        params,
        inplace,
        verbose,
    )

    # noinspection PyPep8Naming
    X = data[exp_filter][features]
    if add_random:
        X['RANDOM'] = np.random.randint(0, 100, size=len(data))
    for feature in features_categorical:
        X[feature] = X[feature].astype(str)
    model = model_object(objective=objective, cat_features=features_categorical, verbose=False, **params)

    steps = X.shape[1] - 1
    # noinspection PyPep8Naming
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=params['random_state'])
    train_pool = Pool(X_train, y_train, feature_names=list(X.columns), cat_features=features_categorical)
    test_pool = Pool(X_test, y_test, feature_names=list(X.columns), cat_features=features_categorical)
    if add_random:
        X.drop(columns=['RANDOM'], inplace=True)

    select_features_result = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=f'0-{steps}',
        num_features_to_select=1,
        steps=steps,
        algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=False,
        logging_level='Silent',
        plot=False,
    )

    features_importance = {
        'Признак':  (
                select_features_result['eliminated_features_names'] +
                select_features_result['selected_features_names']
        ),
        'Потери': select_features_result['loss_graph']['loss_values']
    }
    features_importance = pd.DataFrame(features_importance)
    features_importance.sort_values('Потери', ascending=False, inplace=True)
    features_importance.index = features_importance.reset_index().index + 1
    if features_description is not None:
        if add_random and 'RANDOM' not in features_description:
            features_description = features_description.copy()
            features_description['RANDOM'] = 'Случайный признак'
        features_importance['Описание'] = features_importance['Признак'].replace(features_description)
        features_importance = features_importance[['Признак', 'Описание', 'Потери']].copy()

    return features_importance


def calc_features_shap_importance(
    data=None,
    features=None,
    features_categorical=None,
    target_col=None,
    model_type=None,
    objective=None,
    model=None,
    exposure_col=None,
    features_default=None,
    add_random=False,
    use_shap=True,
    features_description=None,
    params=None,
    inplace=False,
    verbose=False,
):
    if model is None:
        data, y, exp_filter, model_object, params = _model_prepare(
            data,
            target_col,
            model_type,
            exposure_col,
            features_default,
            params,
            inplace,
            verbose,
        )
        # noinspection PyPep8Naming
        X = data[exp_filter][features]
        if add_random:
            rand_generator = np.random.default_rng(1)
            X['RANDOM'] = rand_generator.integers(1, 100, size=len(data))
        for feature in features_categorical:
            X[feature] = X[feature].astype(str)
        model = model_object(objective=objective, cat_features=features_categorical, verbose=False, **params)

        # noinspection PyPep8Naming
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=params['random_state'])

        train_pool = Pool(X_train, y_train, feature_names=list(X.columns), cat_features=features_categorical)
        test_pool = Pool(X_test, y_test, feature_names=list(X.columns), cat_features=features_categorical)
        model.fit(train_pool, eval_set=test_pool)
        if add_random:
            X.drop(columns=['RANDOM'], inplace=True)
    else:
        # noinspection PyPep8Naming
        X_test = data

    if use_shap:
        # Расчет важности методов SHAP по тестовой выборке
        explainer = TreeExplainer(model)
        shap_values = explainer(X_test)
        importance_col = 'SHAP'
        features_importance = {
            'Признак': shap_values.feature_names,
            importance_col: np.abs(shap_values.values).mean(axis=0)
        }
        importance_col = 'SHAP'
    else:
        # Получение важности из CatBoost
        importance_col = 'CatBoost'
        features_importance = {
            'Признак': model.feature_names_,
            importance_col: model.get_feature_importance(),
        }

    features_importance = pd.DataFrame(features_importance)
    features_importance.sort_values(importance_col, ascending=False, inplace=True)
    features_importance.index = features_importance.reset_index().index + 1
    if features_description is not None:
        if add_random and 'RANDOM' not in features_description:
            features_description = features_description.copy()
            features_description['RANDOM'] = 'Случайный признак'
        features_importance['Описание'] = features_importance['Признак'].replace(features_description)
        features_importance = features_importance[['Признак', 'Описание', importance_col]].copy()

    return features_importance


def calc_features_directions(
    data,
    features_numerical,
    target_col,
    exposure_col=None,
    features_categorical_numerical=None,
    features_categorical_replace=None,
    features_description=None,
    zero_corr_threshold=0,
    verbose=False,
):
    data = data.copy()
    if features_categorical_replace is not None:
        data.replace(features_categorical_replace, inplace=True)
        for feature in features_categorical_replace.keys():
            data[feature] = data[feature].astype(float)

    if exposure_col is not None:
        exp_filter = data[exposure_col] > 0
        y = data[exp_filter][target_col] / data[exp_filter][exposure_col]
        if verbose:
            print('Using exposure')
    else:
        exp_filter = np.full(len(data), True)
        y = data[target_col]

    if features_categorical_numerical is not None:
        features = features_numerical + features_categorical_numerical
    else:
        features = features_numerical
    # noinspection PyPep8Naming
    X = data[exp_filter][features]

    features_direction = []
    for feature in features:
        corr = X[feature].corr(y, method='spearman')
        features_direction.append([feature, corr])

    features_direction = pd.DataFrame(features_direction, columns=['Признак', 'Корр'])
    features_direction['Знак'] = 0
    features_direction.loc[features_direction['Корр'] > zero_corr_threshold, 'Знак'] = 1
    features_direction.loc[features_direction['Корр'] < -zero_corr_threshold, 'Знак'] = -1
    features_direction.drop(columns=['Корр'], inplace=True)
    if features_description is not None:
        features_direction['Описание'] = features_direction['Признак'].replace(features_description)
        features_direction = features_direction[['Признак', 'Описание', 'Знак']].copy()

    return features_direction


def plot_importance(
    features_importance,
    importance_col,
    data=None,
    features_numerical=None,
    features_new=None,
    target_col=None,
    exposure_col=None,
    features_categorical_numerical=None,
    features_categorical_replace=None,
    features_description=None,
    show_directions=True,
    zero_corr_threshold=0,
    title=None,
    figsize=(5, 10),
    split=None,
    verbose=False,
):
    if show_directions:
        features_direction = calc_features_directions(
            data=data,
            features_numerical=features_numerical,
            features_categorical_numerical=features_categorical_numerical,
            features_categorical_replace=features_categorical_replace,
            target_col=target_col,
            exposure_col=exposure_col,
            features_description=features_description,
            zero_corr_threshold=zero_corr_threshold,
            verbose=verbose,
        )
        features_direction = features_direction[['Признак', 'Знак']]
        features_direction.set_index('Признак', drop=True, inplace=True)
        features_direction = features_direction['Знак']
    else:
        features_direction = {}

    features_importance = features_importance.copy()
    features_importance.sort_values(importance_col, ascending=True, inplace=True)
    if show_directions:
        features_importance['color'] = features_importance['Признак']
        features_importance['color'] = features_importance['color'].map(features_direction)
        features_importance['color'] = features_importance['color'].replace(
            {1: 'tab:red', -1: 'tab:blue', 0: 'tab:green'}
        ).fillna('k')
    if 'Описание' in features_importance.columns:
        features_importance.set_index('Описание', drop=True, inplace=True)
    else:
        features_importance.set_index('Признак', drop=True, inplace=True)

    if show_directions:
        legend = pd.DataFrame({
            'name': [
                'Не числовой',
                'Прямое влияние на таргет',
                'Обратное влияние на таргет',
            ],
            'value': [1, 1, 1],
            'color': ['k', 'tab:red', 'tab:blue'],
        })
        if zero_corr_threshold > 0:
            legend.loc[4] = ['Не влияет на таргет', 1, 'tab:green']
        legend.set_index('name', drop=True, inplace=True)
        ax = legend['value'].plot.barh(color=legend['color'], figsize=(2, 1))
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_title('Легенда к цветам')
        plt.show()

    if not split:
        split = 1000
    start = 0
    stop = split
    len_data = len(features_importance)
    first = True
    xlim = None
    while start < len_data:
        data_split = features_importance.sort_values(importance_col, ascending=False).iloc[start:stop][importance_col]
        data_split = data_split.sort_values(ascending=True)
        if show_directions:
            ax = data_split.plot.barh(
                color=features_importance['color'],
                figsize=figsize,
            )
        else:
            ax = data_split.plot.barh(figsize=figsize)
        if first:
            xlim = ax.get_xlim()
        else:
            ax.set_xlim(xlim[0], xlim[1])

        ax.set_xlabel('Значение важности')
        if title is None:
            title = ''
        if features_new:
            title += '. Красные признаки - новые'
        ax.set_title(f'Важность признаков {title}')
        if features_new is not None:
            if features_description is not None:
                features_new_description = [features_description[feature] for feature in features_new]
            else:
                features_new_description = []
            for index, tick in enumerate(ax.get_yticklabels()):
                if tick.get_text() in features_new_description or tick.get_text() in features_new:
                    ax.get_yticklabels()[index].set_color('r')
        plt.show()
        start = stop
        stop += split
        first = False

def encode_features_onehot(
    data,
    features_categorical,
    features_numerical=None,
    features=None,
    update_lists=False,
    dtype=float,
):
    data = pd.get_dummies(data, columns=features_categorical, dtype=dtype)
    if update_lists:
        features_onehot = [col for col in data.columns for col_cat in features_categorical if col_cat in col]
        features_numerical.extend(features_onehot)
        features_categorical = []
        if features:
            features = features_categorical + features_numerical
        return data, features_numerical, features_categorical, features
    return data


def _model_prepare(
    data,
    target_col,
    model_type,
    exposure_col,
    features_default,
    params,
    inplace,
    verbose,
):
    if not inplace:
        data = data.copy()
    if features_default is not None:
        data.fillna(features_default, inplace=True)

    if exposure_col is not None:
        exp_filter = data[exposure_col] > 0
        y = data[exp_filter][target_col] / data[exp_filter][exposure_col]
        if verbose:
            print('Using exposure')
    else:
        exp_filter = np.full(len(data), True)
        y = data[target_col]
    if model_type == 'regressor':
        model_object = CatBoostRegressor
        if verbose:
            print('Using CatBoostRegressor')
    else:
        model_object = CatBoostClassifier
        if verbose:
            print('Using CatBoostClassifier')
    if params is None:
        params = {
            'iterations': 80,
            'random_state': 1,
        }
    return data, y, exp_filter, model_object, params
