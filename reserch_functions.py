>> Гатятулин Чингиз Русланович [01.07.2026 15:23]
# function vizualisation

expsoure = 'expos'
damage_count = 'TARGET_2'
damage_sum =  'TARGET_3_SEV'

       
def plot_cat_vs_target(data, x_min, x_max, figsize, feature, save, model_type, rotation):
    # Функция нижнего уровня для двух функций ниже
    if x_min:
        data = data[data['ratio'] > x_min]
    if x_max:
        data = data[data['ratio'] < x_max]
    n = data.shape[0]
    ind = np.arange(n)
    
    fig, ax = plt.subplots(dpi=100, figsize=figsize)

    ax.bar(ind, data[expsoure])
    # ax.set_yticks(fontsize=20)
    # ax.ylabel()
    ax.set_ylabel(expsoure, fontsize=20)
    ax.set_xticks(ind, data.index.tolist(), fontsize=20, rotation=rotation) #  rotation='vertical'
    
    ax.tick_params(axis='both', labelsize=20)
    
    axes2 = ax.twinx()
    axes2.plot(ind, data['ratio'], color='r', marker='o')
    
    if model_type == 'frequency':
        axes2.set_ylabel('Частота', fontsize=20)
    elif model_type == 'severity':
        axes2.set_ylabel('Severity', fontsize=20)
        
    axes2.tick_params(axis='both', labelsize=20)
    plt.grid(False)
    plt.title(f"""{feature}_{str(model_type).upper()}""", fontsize=25)
    
    if save:
        plt.savefig(f"""plots/{feature}_{str(model_type).upper()}.png""", bbox_inches='tight', dpi=1200)
    plt.show()

def research_continous(data, feature, quantiles, model_type='frequency', figsize:tuple=(55, 10), save=False, rotation=90):
    if model_type == 'frequency':
        data = data[[feature, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates='drop', retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join='outer')
        grouped = data.groupby(feature).agg(sum)
        grouped['ratio'] = grouped[damage_count] / grouped[expsoure]
    elif model_type == 'severity':
        data = data[[feature, damage_sum, expsoure, damage_count]]
        quantiles, bins = pd.qcut(data[feature], quantiles, duplicates='drop', retbins=True)
        data.drop(feature, axis=1, inplace=True)
        data = pd.concat([data, quantiles], axis=1, join='outer')
        grouped = data.groupby(feature).agg(sum)
        grouped['ratio'] = grouped[damage_sum] / grouped[damage_count]
        
    # grouped[expsoure] = grouped[expsoure] / sum(grouped[expsoure])
    grouped[damage_count] = grouped[damage_count] / sum(grouped[damage_count])

    plot_cat_vs_target(grouped, None, None, figsize, feature, save,  model_type, rotation)
    return grouped
    
def research_feature(data, feature, bounds=None, sort_by=None, x_min: float=None, x_max: float=None, 
                     figsize: tuple=(55, 10), max_limit=None, min_limit=None, model_type='frequency', 
                     save=False, rotation=90):

    if model_type == 'frequency':
        data = data[[feature, expsoure, damage_count]]
        grouped = data.groupby(feature, dropna=False).agg(sum)
    
        grouped['ratio'] = grouped[damage_count] / grouped[expsoure]
    elif model_type == 'severity':
        data = data[[feature, expsoure, damage_count, damage_sum]]
        grouped = data.groupby(feature, dropna=False).agg(sum)
        
        grouped['ratio'] = grouped[damage_sum] / grouped[damage_count]
        # grouped['ratio'] = grouped['ratio'] / sum(grouped['ratio'])
        
    # grouped[expsoure] = grouped[expsoure] / sum(grouped[expsoure])
    # grouped[damage_count] = grouped[damage_count] / sum(grouped[damage_count])
    # list(quarter.index)[0].split()[1:]
    if sort_by == 'index':
        grouped = grouped.sort_index()
    elif sort_by == 'index_d':
        
        def quarter(q):
            res = []
            for quarter in q:
                quar, year = str(quarter).split()[1:]
                quar = quar[0]
                res.append(int(year + quar))
            return res
        
grouped =
>> Гатятулин Чингиз Русланович [01.07.2026 15:23]
grouped.sort_index(key=quarter)
        
    elif sort_by is None:
        grouped = grouped.sort_values(by='ratio', ascending=False)
    else:
        grouped = grouped.sort_values(by=sort_by, ascending=False)
        
    
    if max_limit is not None:
        grouped = grouped[grouped[expsoure] < max_limit]
        
    if min_limit is not None:
        grouped = grouped[grouped[expsoure] > min_limit]
        
        
    if bounds is not None:
        groups = concat_group(grouped[['ratio']], bounds)
        grouped = pd.concat([grouped, groups], axis=1)
    plot_cat_vs_target(grouped, x_min, x_max, figsize, feature, save, model_type, rotation)
    return grouped




def value_type(df: pd.DataFrame, isprint=True, count_numeric=100):
    """
    Функция для разделения признаков по количеству значений данных в них

    Parameters
    ----------
    df : pd.DataFrame
        Датафрейм, из которого будут получены данные
    isprint : str
        Флаг, отвечающий за то, будет ли выводиться строка после распределения по каждому признаку

    Returns
    -------
    (bin_list, cat_list, num_list, drop_list, obj_list): Cortage of 5 [list of str]
        (
        Список бинарных признаков (2 значения),
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
    bin_list = []
    cat_list = []
    num_list = []
    drop_list = []
    date_list = []
    obj_list = []
    # Цикл по колонкам датафрейма
    for col in tqdm(df.columns):
        try:
            VC = df[col].nunique(dropna=False)
        except:
            print(col, ' не хэшируемый тип')
            continue
        # Если только 1 значение
        if VC ==1:
            if isprint:
                print('DROP:', col )
            drop_list.append(col)
        # Если только 2 значения
        if VC ==2:
            if isprint:
                print('binary:', col )
            bin_list.append(col)
        # Если значений в столбце от 3 до 100
        if 2 < VC <= count_numeric:
            if isprint:
                print('categorial:', col )
            cat_list.append(col)
    for col in tqdm(df.columns):
        #VC = df[col].value_counts(dropna=False)
        # Теперь рассмотрим колонки, которые не вошли в предыдущие списки
        if col not in bin_list and col not in cat_list:
            # Для строкового типа, например
            if df[col].dtype == object:
                if isprint:
                    print('object:', col )
                obj_list.append(col)
            elif df[col].dtype == '<M8[ns]':
                if isprint:
                    print('date:', col )
                date_list.append(col)
            # Для всего остального
            else:                
                if isprint:
                    print('numeric:', col )
                num_list.append(col)
    print("BINARY:", bin_list)
    print("CATEGORIAL:", cat_list)
    print("NUMERIC:", num_list)
    print("TO_DROP:", drop_list)
    print("OBJECT:", obj_list)
    print("DATE:", date_list)
    #return (bin_list, cat_list, num_list, drop_list, obj_list, date_list)
    return ({"BINARY": bin_list,
             "CATEGORIAL": cat_list,             
"NUMERIC"
>> Гатятулин Чингиз Русланович [01.07.2026 15:23]
: num_list,
             "TO_DROP": drop_list,
             "OBJECT": obj_list, 
             "DATE": date_list
            })