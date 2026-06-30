# TARGET_2
input_types_dict = {
"NUMERIC":[
            'LONGITUDE',
            'LATITUDE'
           ],
"CATEGORIAL": [
           ],
'TO_DROP':[
    'Сумма_взыскано_по_ФУ',
    'Суммы_взыскано_по_иску',
    'Общая_сумма_заявленных_требований_ФУ',
    'Доп_расходы_инцидент',
    'Общая_сумма_заявленных_требований_ИСК',
    'Номер_инциндента',
    'UTSSurchargeValue_cumsum_by_incident_all',
    'SurchargeValue_cumsum_by_incident_all',
    'PAYMENT_VALUE',
    'Сумма_выплат_по_претензиям',
    'FIN',
    'VICTIM_LOSS_SUM_FUTURE',
    'GUILTY_LOSS_SUM_FUTURE',
    'REFUND_FORM_INCIDENT',
    'PAYMENTS_SUM_RUR',
    'VICTIM_LOSS_COUNT_FUTURE',
    'FIX',
    'GUILTY_LOSS_COUNT_FUTURE',
    'LOSS_AMOUNT',
    'Выплата_по_основному_убытку',
    'Взысканный_износ_ИСК',
    'REFUND_FORM_INCIDENT',
    'EVENT_NUMBER',
    'PREMIUM_SUM_FUTURE_ALL',
    'OSAGO_FL_2022_PVU_FREQUENCY_osago',
    'OSAGO_FL_2022_PVU_PVU_RESULT_INFL_osago',
    'EXPECTED_LOSS_PVU3_osago',
    'PVU_FREQ_osago',
    'PVU_SUMMATREBOVANII_osago',
    'PVU_SUMMAFIKS_osago',
    'PVU_SEGMENT_osago',
    'PVU_NULL_CLAIM_osago',
    'PVU_NULL_CLAIM_FREQ_osago',
    'PVU_EXPENSE_COUNT_osago',
    'PVU_EXPENSE_osago',
    'REFUND_FORM',
    'Сумма_износа_по_калькуляции_инцидент',
    'SEASON_WINTER_PRECIPITATION_AVG_MIN_osago',
    'SEASON_SPRING_PRECIPITATION_DEV_PERSENT_MIN_osago',
    'PREMIUM_COUNT_FUTURE_ALL',
    'PREMIUM_SUM_ALL',
    'SEASON_SUMMER_PRECIPITATION_AVG_MAX_osago',
    'total_loss_osago',
    'LOSS_UNIT_DIVISION',
    'FUTURE_PROLONGATION_osago',
    'SEASON_WINTER_TEMPERATURE_DEV_MAX_osago',
    'SEASON_SUMMER_TEMPERATURE_AVG_MAX_osago',
    'total_loss_pay_osago',
    'PAYMENT_TYPE',
    'SEASON_WINTER_TEMPERATURE_AVG_MIN_osago',
    'SEASON_WINTER_PRECIPITATION_NORMAL_osago',
    'APPLICANT_FTRS_PRET_PRETENSION_NUMBER_nunique',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES_nunique',
    'APPLICANT_FTRS_PRET_PRETENSION_KINDS_nunique',
    'APPLICANT_FTRS_PRET_INSURANCE_TYPE_GROUPS_nunique',
    'APPLICANT_FTRS_PRET_PRETENSION_VALUE_SUM',
    'APPLICANT_FTRS_PRET_UTS_VALUE_SUM',
    'APPLICANT_FTRS_PRET_SURCHARGE_VALUE_SUM',
    'APPLICANT_FTRS_PRET_UTS_SURCHARGE_VALUE_SUM',
    'APPLICANT_FTRS_PRET_CESSION_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_VALUE_PENALTY_SUM',
    'APPLICANT_FTRS_PRET_SURCHARGE_VALUE_PENALTY_SUM',
    'APPLICANT_FTRS_PRET_INSURANCE_TYPE_GROUPS__КАСКО+ГО_SUM',
    'APPLICANT_FTRS_PRET_INSURANCE_TYPE_GROUPS__ОСАГО_SUM',
    'APPLICANT_FTRS_PRET_INSURANCE_TYPE_GROUPS__ПРОЧЕЕ_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Жалоба по ОСАГО 5+_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Запрос документов по делу_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Несогласие с суммой выплаты_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Отказ от ремонта (смена формы возмещения)_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__ПРОЧЕЕ_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Претензия на принятое решение_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Претензия на сроки ремонта и согласования_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Претензия по качеству ремонта_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Претензия по смене СТОА_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES__Требование по выплате только неустойки_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__Интернет_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__Лично_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__ПРОЧЕЕ_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__Почта_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__Электронное_SUM',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD__None_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__Выплата_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__Направлен ответ_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__Направлены документы_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__Отказ в удовлетворениипретензии_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__Частичная выплата_SUM',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE__None_SUM',
    'APPLICANT_FTRS_PRET_PRETENSIONTYPE_mode',
    'APPLICANT_FTRS_PRET_PRETENSION_GET_METHOD_mode',
    'APPLICANT_FTRS_PRET_LOSS_UNIT_mode',
    'APPLICANT_FTRS_PRET_LOSS_UNIT_ZONE_mode',
    'APPLICANT_FTRS_PRET_ANSWER_TYPE_mode',
    'APPLICANT_FTRS_PRET_PRETENSION_TYPES_mode',
    'APPLICANT_FTRS_PRET_PRETENSION_KINDS_mode',
    'APPLICANT_FTRS_PRET_INSURANCE_TYPE_GROUPS_mode',
    'APPLICANT_FTRS_COURT_Представитель_SUM',
    'APPLICANT_FTRS_COURT_Цессионарий_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDMAINDEBT_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDPLAINTIFFEXAMINATION_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDCOURTEXAMINATION_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDREPRESENTATIVEEXPENSES_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDPERCENTFORUSES_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDPENALTYFEE_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDFINE_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDMORALDAMAGE_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDOTHEREXPENSES_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDSTATEDUTY_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDLOSSCOMMODYVALUE_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDWEAROUT_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDVALUEWITHOUTSD_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDVALUEWITHSD_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDAMOUNTLOSS_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_3_Е_ЛИЦО_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_НЕУСТОЙКА_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_ПРОЧЕЕ_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_СПОР_ПО_СУММЕ_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_ТРАСОЛОГИЯ_SUM',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ_SUM',
    'APPLICANT_FTRS_COURT_CLAIMEDMAINDEBT_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDPLAINTIFFEXAMINATION_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDCOURTEXAMINATION_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDREPRESENTATIVEEXPENSES_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDPERCENTFORUSES_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDPENALTYFEE_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDFINE_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDMORALDAMAGE_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDOTHEREXPENSES_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDSTATEDUTY_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDLOSSCOMMODYVALUE_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDWEAROUT_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDVALUEWITHOUTSD_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDVALUEWITHSD_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDAMOUNTLOSS_1_MEAN',
    'APPLICANT_FTRS_COURT_CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1_MEAN',
    'APPLICANT_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE',
    'APPLICANT_FTRS_COURT_LINK_LOSS_NUMBER_NUNIQUE',
    'APPLICANT_FTRS_COURT_ОбращениеКФУОтЗаявителяПоступилоПосредством_MODE',
    'APPLICANT_FTRS_COURT_CLAIM_ITEM_MODE',
    'APPLICANT_FTRS_COURT_CLAIMORIGIN_1_MODE',
    'VICTIM_PH_FTRS_COURT_Представитель_SUM',
    'VICTIM_PH_FTRS_COURT_Цессионарий_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDMAINDEBT_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPLAINTIFFEXAMINATION_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDCOURTEXAMINATION_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDREPRESENTATIVEEXPENSES_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPERCENTFORUSES_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPENALTYFEE_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDFINE_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDMORALDAMAGE_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDOTHEREXPENSES_1_SUM',
    'VICTIM_PH _FTRS _COUR T_CLA IMEDS TATED UTY_1 _SUM' ,
    'VICTIM_PH_FTRS_COURT_CLAIMEDLOSSCOMMODYVALUE_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDWEAROUT_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDVALUEWITHOUTSD_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDVALUEWITHSD_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDAMOUNTLOSS_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_3_Е_ЛИЦО_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_НЕ_ПРЕДОСТАВЛЕНИЕ_НА_ОСМОТР_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_НЕКОМПЛЕКТНОСТЬ_ДОКУМЕНТОВ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_НЕПРИЗНАНИЕ_СТРАХОВЫМ_СЛУЧАЕМ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_НЕУСТОЙКА_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_ПРОЧЕЕ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_СПОР_ПО_СУММЕ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_СУДЕБНЫЕ_РАСХОДЫ_И_САНКЦИИ_ЗА_НЕИСПОЛНЕНИЕ_РЕШЕНИЯ_СУДА_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_ТРАСОЛОГИЯ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_ФОРМА_ВОЗМЕЩЕНИЯ_SUM',
    'VICTIM_PH_FTRS_COURT_CLAIMEDMAINDEBT_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPLAINTIFFEXAMINATION_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDCOURTEXAMINATION_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDREPRESENTATIVEEXPENSES_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPERCENTFORUSES_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDPENALTYFEE_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDFINE_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDMORALDAMAGE_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDOTHEREXPENSES_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDSTATEDUTY_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDLOSSCOMMODYVALUE_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDWEAROUT_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDVALUEWITHOUTSD_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDVALUEWITHSD_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDAMOUNTLOSS_1_MEAN',
    'VICTIM_PH_FTRS_COURT_CLAIMEDCOURTEXAMINATIONBEFORERESOLVE_1_MEAN',
    'VICTIM_PH_FTRS_COURT_INCOMING_CLAIM_NUMBER_NUNIQUE',
    'VICTIM_PH_FTRS_COURT_LINK_LOSS_NUMBER_NUNIQUE',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_NUNIQUE',
    'VICTIM_PH_FTRS_COURT_ОбращениеКФУОтЗаявителяПоступилоПосредством_MODE',
    'VICTIM_PH_FTRS_COURT_CLAIM_ITEM_MODE',
    'VICTIM_PH_FTRS_COURT_CLAIMORIGIN_1_MODE',
    'REGION_CORRECTED',
    'PREMIUM_SUM_OSAGO',
    'LOSS_STATE_BY_IA'
],
    'BINARY': ['VICTIM_VEHICLE_IS_JAPAN']
}

other_cols = [
              'INCIDENT_NUMBER', 'LOSS_NUMBER',
               'TARGET', 'TARGET_3_FREQ', 'TARGET_3', 'TARGET_3_SEV', 'TARGET_2'
]

mvp = MVP(df)
mvp.value_type()

for i in mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']:
    try:
        df[i] = df[i].apply(lambda x: int(float(x))).astype(str)
    except:
        df[i] = df[i].astype(str)

mvp.correct_types(input_types_dict, other_cols)
mvp.types_dict

df['TARGET_2'] = df['TARGET_2'].astype(int)
df['TARGET_2'].dtypes

X_train_freq = df[df['LOSS_DATE_TIME'].between('2022-01-01', '2024-05-31')]
y_train_freq = df[df['LOSS_DATE_TIME'].between('2022-01-01', '2024-05-31')]['TARGET_2']

X_test_freq = df[df['LOSS_DATE_TIME'].between('2024-06-01', '2025-06-01')]
y_test_freq = df[df['LOSS_DATE_TIME'].between('2024-06-01', '2025-06-01')]['TARGET_2']


display(X_train_freq.shape)
display(X_test_freq.shape)

display(y_train_freq.shape[0] / (y_train_freq.shape[0] + y_test_freq.shape[0]) * 100)
display(y_test_freq.shape[0] / (y_train_freq.shape[0] + y_test_freq.shape[0]) * 100)

display(y_train_freq.mean())
display(y_test_freq.mean())

pool = Pool(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']] ,
            y_train_freq,
            cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'])],
            feature_names=list(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))

model_freq = CatBoostClassifier(iterations=100,
                           random_state=2026,
                           auto_class_weights='Balanced',
#                           eval_metric='AUC')
#                           custom_metric='AUC:hints=skip_train~false',
#                           early_stopping_rounds=50,
                           verbose=250)

model_freq.fit(pool,
          eval_set=(X_test_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                    y_test_freq),
          plot=True)

#fig, ax = plt.subplots(dpi = 150, figsize = (15,20))
a = dict(zip(model_freq.feature_names_, model_freq.feature_importances_))
a = pd.DataFrame(a.items()).sort_values(1, ascending=True)
#ax.barh(a[0], a[1])

a.sort_values(1, ascending=False).reset_index(drop=True)


diagnostics = ModelDiagnostics(
    X_train=X_train_freq,
    y_train=y_train_freq,
    X_test=X_test_freq,
    y_test=y_test_freq,
    model=model_freq,
    features=mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC'],
    cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'])],
#    fairness='CLAIMEDMAINDEBT_1',
#    target_transform='log1p',
    task_type='classification'
)
# diagnostics.run_full_diagnostics()
diagnostics.compute_metrics(print_metrics=True)


%%time

from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



train_pool = Pool(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                  y_train_freq,
                  cat_features=(mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']),
                  feature_names=list(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))
test_pool = Pool(X_test_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                 y_test_freq,
                 cat_features=(mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']),
                 feature_names=list(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))

model_freq = CatBoostClassifier(iterations=100,
                                early_stopping_rounds=50,
                                random_state=2026,
                                auto_class_weights='Balanced'
                               )

summary = model_freq.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-' + str(X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].shape[1] - 1),
    num_features_to_select=20,
    steps=X_train_freq[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].shape[1],
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=False,
    logging_level='Silent',
    plot=False
)


with open('/home/jovyan/old_home/Litigant/data/processed/features_importance_fist_target_3_freq.pkl', 'wb') as pickle_out:
    pickle.dump(summary, pickle_out)


with open('/home/jovyan/old_home/Litigant/data/processed/features_importance_fist_target_3_freq.pkl', 'rb') as pickle_in:
    summary_freq = pickle.load(pickle_in)


summary_freq['selected_features_names'].remove('RESTRICT_CNT')  # 0.5721657878271276
summary_freq['selected_features_names'].remove('PAYMENT_RECIPIENT_SEX')  # 0.5678995390672719
summary_freq['selected_features_names'].remove('VIC_TS_COUNTRY')  # 0.5712994243774198
summary_freq['selected_features_names'].remove('VICTIM_VEHICLE_MADE_IN_RF')  # 0.5669791271361277
summary_freq['selected_features_names'].remove('VICTIM_NUM_DOORS')  # 0.568496241235262
summary_freq['selected_features_names'].remove('REGION_EVENT')  # 0.5638985968461828
summary_freq['selected_features_names'].remove('GUILTY_MAX_WEIGHT')  # 0.5837922072735072
summary_freq['selected_features_names'].remove('REFUND_FORM_DETAILED')# 0.5722840492340335
summary_freq['selected_features_names'].remove('VICTIM_TYPE_PRIVOD')  # 0.5719185609008979
summary_freq['selected_features_names'].remove('REFUND_FORM_DETAILED')  # 0.5544470417125283
summary_freq['selected_features_names'].remove('EMRValue')  #


pool = Pool(X_train_freq[summary_freq['selected_features_names']],
            y_train_freq,
            cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']) if i in summary_freq['selected_features_names']],
            feature_names=list(X_train_freq[summary_freq['selected_features_names']].columns))

model_freq = CatBoostClassifier(iterations=375,
                           random_state=0,
                           auto_class_weights='Balanced',
#                           eval_metric='AUC')
#                           custom_metric='AUC:hints=skip_train~false',
#                           learning_rate = 0.03,          # медленное обучение
#                           depth = 5,                     # не глубже 6
#                           l2_leaf_reg = 7,               # сильная L2-регуляризация
#                           bagging_temperature = 0.8,
#                           random_strength = 0.5,
#                           eval_metric = 'AUC',           # или 'BalancedAccuracy'
#                           early_stopping_rounds = 50,
                           verbose=250)

model_freq.fit(pool,
          eval_set=(X_test_freq[summary_freq['selected_features_names']], y_test_freq),
          plot=True)


diagnostics = ModelDiagnostics(
    X_train=X_train_freq,
    y_train=y_train_freq,
    X_test=X_test_freq,
    y_test=y_test_freq,
    model=model_freq,
    features=summary_freq['selected_features_names'],
    cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']) if i in summary_freq['selected_features_names']],
#    fairness='CLAIMEDMAINDEBT_1',
#    target_transform='log1p',
    task_type='classification'
)
diagnostics.compute_metrics(print_metrics=True)
# diagnostics.run_full_diagnostics()


a = dict(zip(model_freq.feature_names_, model_freq.feature_importances_))
a = pd.DataFrame(a.items()).sort_values(1, ascending=True)

a.sort_values(1, ascending=False).reset_index(drop=True)


top_features = a.sort_values(1, ascending=False)
top_features = top_features


df['TARGET_2'] = df['TARGET_2'].astype(int)


# Переменные для обозначения колонок, в которых функции визуализации будут искать данные

df['expos'] = 1

to_type = []

for col in [col for col in mvp.types_dict['NUMERIC'] if col not in ['LONGITUDE', 'LATITUDE']]:
    print(col, "nulls:", df[col].isnull().mean() * 100, "%")
    try:
        research_continous(df, col, 10, model_type='frequency', figsize=(25, 10), rotation=90)
    except:
        to_type.append(col)
        df[col] = df[col].astype(float)
        research_continous(df, col, 10, model_type='frequency', figsize=(25, 10), rotation=90)

for col in [col for col in mvp.types_dict['BINARY']]:
    print(col, "nulls:", df[col].isnull().mean() *100, "%")
    print(col, "nunique:", df[col].nunique())
    #visualize_cat(df, col, "target_freq", share=True, depth = 15)
    if df[col].nunique() < 2000:
        research_feature(df, col, model_type = 'frequency', figsize = (25, 10), rotation = 90)

for col in [col for col in mvp.types_dict['CATEGORIAL']]:
    print(col, "nulls:", df[col].isnull().mean() *100, "%")
    print(col, "nunique:", df[col].nunique())
    #visualize_cat(df, col, "target_freq", share=True, depth = 15)
    if df[col].nunique() < 2000:
        research_feature(df, col, model_type = 'frequency', figsize = (25, 10), rotation = 90)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    PrecisionRecallDisplay, 
    RocCurveDisplay, 
    precision_recall_curve, 
    roc_auc_score,
    average_precision_score  # добавили этот импорт
)

# === Предсказания ===
y_pred_train = model_freq.predict_proba(X_train_freq[summary_freq['selected_features_names']])[:, 1]
y_pred_test = model_freq.predict_proba(X_test_freq[summary_freq['selected_features_names']])[:, 1]


# === Вариант 1: Базовая гистограмма (с нормировкой) ===
plt.figure(figsize=(12, 5))

sns.histplot(
    y_pred_train,
    bins=50,
    alpha=0.5,
    label=f'Train (n={len(y_pred_train):,})',
    color='blue',
    stat='density',
    common_norm=False
)

sns.histplot(
    y_pred_test,
    bins=50,
    alpha=0.5,
    label=f'Test (n={len(y_pred_test):,})',
    color='green',
    stat='density',
    common_norm=False
)

plt.axvline(x=0.5, color='red', linestyle='--', label='Порог 0.5')
plt.xlabel('Predicted Probability (класс 1)')
plt.ylabel('Плотность (density)')
plt.title('Распределение предсказанных вероятностей: Train vs Test (нормировано)')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()


# === Вариант 2: Раздельно по классам (с нормировкой) ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Для класса 0 (негативные)
sns.histplot(
    y_pred_train[y_train_freq == 0],
    bins=50,
    alpha=0.5,
    label='Train',
    color='blue',
    ax=axes[0],
    stat='density',
    common_norm=False
)
sns.histplot(
    y_pred_test[y_test_freq == 0],
    bins=50,
    alpha=0.5,
    label='Test',
    color='green',
    ax=axes[0],
    stat='density',
    common_norm=False
)
axes[0].axvline(x=0.5, color='red', linestyle='--')
axes[0].set_xlabel('Predicted Probability')
axes[0].set_ylabel('Плотность (density)')
axes[0].set_title('Класс 0 (негативные)')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Для класса 1 (позитивные)
sns.histplot(
    y_pred_train[y_train_freq == 1],
    bins=50,
    alpha=0.5,
    label='Train',
    color='blue',
    ax=axes[1],
    stat='density',
    common_norm=False
)
sns.histplot(
    y_pred_test[y_test_freq == 1],
    bins=50,
    alpha=0.5,
    label='Test',
    color='green',
    ax=axes[1],
    stat='density',
    common_norm=False
)
axes[1].axvline(x=0.5, color='red', linestyle='--')
axes[1].set_xlabel('Predicted Probability')
axes[1].set_ylabel('Плотность (density)')
axes[1].set_title('Класс 1 (позитивные)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Распределение вероятностей по классам (нормировано)', y=1.02)
plt.tight_layout()
plt.show()


# === Вариант 3: Статистика разделимости ===
print("=" * 50)
print("СТАТИСТИКА РАЗДЕЛИМОСТИ КЛАССОВ")
print("=" * 50)

def calc_stats(probs, labels, name):
    neg = probs[labels == 0]
    pos = probs[labels == 1]
    print(f"\n{name}:")
    print(f"  Класс 0: mean={neg.mean():.3f}, std={neg.std():.3f}, median={np.median(neg):.3f}")
    print(f"  Класс 1: mean={pos.mean():.3f}, std={pos.std():.3f}, median={np.median(pos):.3f}")
    print(f"  Разрыв mean: {pos.mean() - neg.mean():.3f}")
    return pos.mean() - neg.mean()

gap_train = calc_stats(y_pred_train, y_train_freq, 'Train')
gap_test = calc_stats(y_pred_test, y_test_freq, 'Test')

print(f"\n📊 Разница в разрыве: {gap_test - gap_train:+.3f}")
if gap_test > gap_train:
    print("✅ На тесте классы лучше разделены → метрики выше")
else:
    print("⚠️ На тесте классы хуже разделены → ищите другую причину")


# === Вариант 4: Precision-Recall и ROC кривые ===
# Сначала считаем метрики отдельно
ap_train = average_precision_score(y_train_freq, y_pred_train)
ap_test = average_precision_score(y_test_freq, y_pred_test)
auc_train = roc_auc_score(y_train_freq, y_pred_train)
auc_test = roc_auc_score(y_test_freq, y_pred_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

PrecisionRecallDisplay.from_predictions(
    y_train_freq, y_pred_train, 
    ax=axes[0], 
    label=f'Train (AP={ap_train:.3f})'
)
PrecisionRecallDisplay.from_predictions(
    y_test_freq, y_pred_test, 
    ax=axes[0], 
    label=f'Test (AP={ap_test:.3f})'
)
axes[0].set_title('Precision-Recall Curve')
axes[0].grid(alpha=0.3)

RocCurveDisplay.from_predictions(
    y_train_freq, y_pred_train, 
    ax=axes[1], 
    label=f'Train (AUC={auc_train:.3f})'
)
RocCurveDisplay.from_predictions(
    y_test_freq, y_pred_test, 
    ax=axes[1], 
    label=f'Test (AUC={auc_test:.3f})'
)
axes[1].set_title('ROC Curve')
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Вывод метрик в консоль
print("\n" + "=" * 50)
print("СВОДКА ПО МЕТРИКАМ")
print("=" * 50)
print(f"Average Precision: Train={ap_train:.3f}, Test={ap_test:.3f}, Δ={ap_test - ap_train:+.3f}")
print(f"ROC-AUC:           Train={auc_train:.3f}, Test={auc_test:.3f}, Δ={auc_test - auc_train:+.3f}")


# TARGET_3 SEV
mvp = MVP(df)
mvp.value_type()

for i in mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']:
    try:
        df[i] = df[i].apply(lambda x: int(float(x))).astype(str)
    except:
        df[i] = df[i].astype(str)

mvp.correct_types(input_types_dict, other_cols)
mvp.types_dict

X_train_sev = df[(df['LOSS_DATE_TIME'].between('2022-01-01', '2024-05-31')) & (df['TARGET_3_SEV'].between(1, 1.5e6))]
y_train_sev = df[(df['LOSS_DATE_TIME'].between('2022-01-01', '2024-05-31')) & (df['TARGET_3_SEV'].between(1, 1.5e6))]['TARGET_3_SEV']

X_test_sev = df[(df['LOSS_DATE_TIME'].between('2024-06-01', '2025-06-01')) & (df['TARGET_3_SEV'].between(1, 1.5e6))]
y_test_sev = df[(df['LOSS_DATE_TIME'].between('2024-06-01', '2025-06-01')) & (df['TARGET_3_SEV'].between(1, 1.5e6))]['TARGET_3_SEV']

X_train_sev.to_parquet('/home/jovyan/old_home/Litigant/data/processed/X_train_sev_for_DataDrift.parquet')

display(X_train_sev.shape)
display(X_test_sev.shape)

display(y_train_sev.shape[0] / (y_train_sev.shape[0] + y_test_sev.shape[0]) * 100)
display(y_test_sev.shape[0] / (y_train_sev.shape[0] + y_test_sev.shape[0]) * 100)

display(y_train_sev.mean())
display(y_test_sev.mean())

pool = Pool(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']] ,
            y_train_sev,
            cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'])],
            feature_names=list(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))

model_sev = CatBoostRegressor(iterations=100,
                           random_state=2026,
                           verbose=250)

model_sev.fit(pool,
          eval_set=(X_test_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                    y_test_sev),
          plot=True)

#fig, ax = plt.subplots(dpi = 150, figsize = (15,20))
a = dict(zip(model_sev.feature_names_, model_sev.feature_importances_))
a = pd.DataFrame(a.items()).sort_values(1, ascending=True)
#ax.barh(a[0], a[1])

a.sort_values(1, ascending=False).reset_index(drop=True)


sys.path.append("/home/jovyan/old_home")

from modeldiagnostics.src.tuning import TuningHyperparameters
from modeldiagnostics.src.categorical_features_processor import CategoricalFeatureProcessor
from modeldiagnostics.src.modeldiagnostics import ModelDiagnostics

diagnostics = ModelDiagnostics(
    X_train=X_train_sev,
    y_train=y_train_sev,
    X_test=X_test_sev,
    y_test=y_test_sev,
    model=model_sev,
    features=mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC'],
    cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'])],
#    fairness='CLAIMEDMAINDEBT_1',
#    target_transform='log1p',
    task_type='regression'
)
diagnostics.compute_metrics(print_metrics=True)


%%time

from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split



train_pool = Pool(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                  y_train_sev,
                  cat_features=(mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']),
                  feature_names=list(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))
test_pool = Pool(X_test_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']],
                 y_test_sev,
                 cat_features=(mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']),
                 feature_names=list(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].columns))

model_sev = CatBoostRegressor(iterations=100,
                           early_stopping_rounds=50,
                           random_state=2026
                         )
summary = model_sev.select_features(
    train_pool,
    eval_set=test_pool,
    features_for_select='0-' + str(X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].shape[1] - 1),
    num_features_to_select=20,
    steps=X_train_sev[mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL'] + mvp.types_dict['NUMERIC']].shape[1],
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=False,
    logging_level='Silent',
    plot=False
)

with open('/home/jovyan/old_home/Litigant/data/processed/features_importance_fist_target_3_sev.pkl', 'wb') as pickle_out:
    pickle.dump(summary, pickle_out)

with open('/home/jovyan/old_home/Litigant/data/processed/features_importance_fist_target_3_sev.pkl', 'rb') as pickle_in:
    summary_sev = pickle.load(pickle_in)

# 46923.12642288472
summary_sev['selected_features_names'].remove('PAYMENT_RECIPIENT_TYPE')  # 46702.179468860275
summary_sev['selected_features_names'].remove('VICTIM_LOSS_COUNT')  # 46627.39378691882
summary_sev['selected_features_names'].remove('VIC_IS_EV_REG')  # 46938.46805214252
summary_sev['selected_features_names'].remove('GUILTY_VEHICLE_AGE_BAD')  # 46586.06208251816
summary_sev['selected_features_names'].remove('APPLICANT_TYPE')  # 46372.40811809138
summary_sev['selected_features_names'].remove('GUILTY_GOS')  # 47304.650852949744
summary_sev['selected_features_names'].remove('RSA_RE_OUT')  # 47030.448441241904
summary_sev['selected_features_names'].remove('VICTIM_VEHICLE_MADE_IN_RF')  # 46580.01230330239
summary_sev['selected_features_names'].remove('GUIL_IS_EV_REG') # 46652.31864780745
summary_sev['selected_features_names'].remove('Сумма_утс') # 47338.70048390558
summary_sev['selected_features_names'].remove('PAYMENT_RECIPIENT_SEX') # 46637.89392499477
summary_sev['selected_features_names'].remove('APPLICANT_SEX') # 46793.855023909884
summary_sev['selected_features_names'].remove('NOT_NOTIFICATION') # 46622.63936230518
summary_sev['selected_features_names'].remove('GUILTY_CAPACITY_ENGINE') # 46645.02468612788
summary_sev['selected_features_names'].remove('VICTIM_NUM_DOORS')  # 45741.933582677964

# 0.5722840492340335
# summary_sev['selected_features_names'].remove('VICTIM_TYPE_PRIVOD')  # 0.5719185609008979
# summary_sev['selected_features_names'].remove('REFUND_FORM_DETAILED')  # 0.5544470417125283
summary_sev['selected_features_names'].remove('EMRValue')  ## 47320.7929595711
summary_sev['selected_features_names'].remove('RSA_RE_OUT')  # 46929.182359035396
summary_sev['selected_features_names'].remove('VIC_IS_EV_REG')  # 
summary_sev['selected_features_names'].remove('GUIL_IS_EV_REG')  # 
summary_sev['selected_features_names'].remove('GUILTY_VEHICLE_AGE_BAD')  # 
summary_sev['selected_features_names'].remove('NOT_NOTIFICATION')  # 47437.73870242069
summary_sev['selected_features_names'].remove('APPLICANT_TYPE')  # 47504.64452271297
summary_sev['selected_features_names'].remove('GUILTY_GOS')  # 47574.80307205721
summary_sev['selected_features_names'].remove('PAYMENT_RECIPIENT_TYPE')  # 47705.83520813375
summary_sev['selected_features_names'].remove('APPLICANT_SEX')  # 47445.403986056335
summary_sev['selected_features_names'].remove('VICTIM_LOSS_COUNT')  # 47476.750965591076
summary_sev['selected_features_names'].remove('VICTIM_VEHICLE_MADE_IN_RF')  # 47575.90050091662
summary_sev['selected_features_names'].remove('Сумма_утс')  # 47485.26363479663
summary_sev['selected_features_names'].remove('PAYMENT_RECIPIENT_SEX')  # 47810.331307039894
summary_sev['selected_features_names'].remove('VICTIM_NUM_DOORS')  # 47409.2058196742
summary_sev['selected_features_names'].remove('GUILTY_CAPACITY_ENGINE')  # 46511.55286675996

pool = Pool(X_train_sev[summary_sev['selected_features_names']],
            y_train_sev,
            cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']) if i in summary_sev['selected_features_names']],
            feature_names=list(X_train_sev[summary_sev['selected_features_names']].columns))

model_sev = CatBoostRegressor(iterations=100,
                          random_state=0,
                          verbose=250)

model_sev.fit(pool,
          eval_set=(X_test_sev[summary_sev['selected_features_names']], y_test_sev),
          plot=True)

a = dict(zip(model_sev.feature_names_, model_sev.feature_importances_))
a = pd.DataFrame(a.items()).sort_values(1, ascending=True)

a.sort_values(1, ascending=False).reset_index(drop=True)

diagnostics = ModelDiagnostics(
    X_train=X_train_sev,
    y_train=y_train_sev,
    X_test=X_test_sev,
    y_test=y_test_sev,
    model=model_sev,
    features=summary_sev['selected_features_names'],
    cat_features=[i for i in (mvp.types_dict['BINARY'] + mvp.types_dict['CATEGORIAL']) if i in summary_sev['selected_features_names']],
#    fairness='CLAIMEDMAINDEBT_1',
#    target_transform='log1p',
    task_type='regression'
)
diagnostics.compute_metrics(print_metrics=True)
# diagnostics.run_full_diagnostics()

top_features = a.sort_values(1, ascending=False)
top_features = top_features

df['TARGET_2'] = df['TARGET_2'].astype(int)

# Переменные для обозначения колонок, в которых функции визуализации будут искать данные

df['expos'] = 1

to_type = []

for col in [col for col in list(top_features.iloc[:, 0]) if col in mvp.types_dict['NUMERIC'] and col not in mvp.types_dict['TO_DROP']]:
    print(col, "nulls:", df[col].isnull().mean() * 100, "%")
    try:
        research_continous(df, col, 10, model_type='severity', figsize=(25, 10), rotation=90)
    except:
        to_type.append(col)
        df[col] = df[col].astype(float)
        research_continous(df, col, 10, model_type='severity', figsize=(25, 10), rotation=90)

for col in [col for col in mvp.types_dict['BINARY'] if col not in mvp.types_dict['TO_DROP'] and col in list(top_features.iloc[:, 0])]:
    print(col, "nulls:", df[col].isnull().mean() *100, "%")
    print(col, "nunique:", df[col].nunique())
    #visualize_cat(df, col, "target_freq", share=True, depth = 15)
    if df[col].nunique() < 2000:
        research_feature(df, col, model_type = 'severity', figsize = (25, 10), rotation = 90)

for col in [col for col in mvp.types_dict['CATEGORIAL'] if col not in mvp.types_dict['TO_DROP'] and col in list(top_features.iloc[:, 0])]:
    print(col, "nulls:", df[col].isnull().mean() *100, "%")
    print(col, "nunique:", df[col].nunique())
    #visualize_cat(df, col, "target_freq", share=True, depth = 15)
    if df[col].nunique() < 2000:
        research_feature(df, col, model_type = 'severity', figsize = (25, 10), rotation = 90)