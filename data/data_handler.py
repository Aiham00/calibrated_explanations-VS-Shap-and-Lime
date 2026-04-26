import pandas as pd
from ucimlrepo import fetch_ucirepo 
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
PATH="data"
def load_ai4i():
    # fetch dataset 
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
    # data (as pandas dataframes) 
    X = ai4i_2020_predictive_maintenance_dataset.data.features 
    y = ai4i_2020_predictive_maintenance_dataset.data.targets

    y['sum'] = y[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1)
    y= y[y['sum']<2]
    y['none'] = 1-y['sum']
    y = y[["none", "TWF", "HDF", "PWF", "OSF", "RNF"]]
    X = X.iloc[y.index]
    feature_names=X.columns
    labels = list(y.columns.values)
    y = y.idxmax(axis=1)
    labels = y.unique()
    return X, y, feature_names, labels

def load_ai4i_b():
    # fetch dataset 
    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601) 
    # data (as pandas dataframes) 
    X = ai4i_2020_predictive_maintenance_dataset.data.features 
    y = ai4i_2020_predictive_maintenance_dataset.data.targets

    y['sum'] = y[["TWF", "HDF", "PWF", "OSF", "RNF"]].sum(axis=1)
    y= y[y['sum']<2]
    y['none'] = 1-y['sum']
    y = y[["none", "TWF", "HDF", "PWF", "OSF", "RNF"]]
    X = X.iloc[y.index]
    feature_names=X.columns
    labels = list(y.columns.values)
    y = y.idxmax(axis=1)
    labels = y.unique()
    return X, y, feature_names, labels

def load_CIA():
    df = pd.read_csv(os.path.join(PATH,"KagglePredictive Maintenance of Machines/CIA-1 Dataset.csv"))
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Labels
    feature_names=X.columns
    labels = list(y.unique()) #labels = {key:value for key,value in enumerate(labels = list(y.columns.values))}
    return X, y, feature_names, labels

def load_CIA():
    df = pd.read_csv(os.path.join(PATH,"KagglePredictive Maintenance of Machines/CIA-1 Dataset.csv"))

    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Labels
    feature_names=X.columns
    labels = list(y.unique()) #
    #labels = {key:value for key,value in enumerate( list(y.unique()))}
    return X, y, feature_names, labels

def load_pump():
    df = pd.read_csv(os.path.join(PATH,"pump/sensor.csv"))
    df.drop(['sensor_15'], axis=1,inplace=True)

    df['machine_status'] = df['machine_status'].shift(periods=-1)
    df.dropna(inplace=True)
    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Labels
    feature_names=X.columns
    labels = list(y.unique()) #
    #labels = {key:value for key,value in enumerate(list(y.unique()))}
    return X, y, feature_names, labels

def preprocess_azure():
    telemetry = pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/PdM_telemetry.csv'))
    errors = pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/PdM_errors.csv'))
    maint = pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/PdM_maint.csv'))
    failures = pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/PdM_failures.csv'))
    machines = pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/PdM_machines.csv'))

    telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")
    #telemetry = telemetry[telemetry['machineID']==1]
    errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
    errors['errorID'] = errors['errorID'].astype('object')
    maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
    maint['comp'] = maint['comp'].astype('object')
    machines['model'] = machines['model'].astype('object')
    failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")
    failures['failure'] = failures['failure'].astype('object')

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                index='datetime',
                                columns='machineID',
                                values=col).resample('3h', closed='left', label='right').mean().unstack())
        
    telemetry_mean_3h = pd.concat(temp, axis=1)
    telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
    telemetry_mean_3h.reset_index(inplace=True)


    temp = []

    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                index='datetime',
                                columns='machineID',
                                values=col).resample('3h', closed='left', label='right').std().unstack())
    telemetry_sd_3h = pd.concat(temp, axis=1)
    telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
    telemetry_sd_3h.reset_index(inplace=True)

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                                index='datetime',
                                                columns='machineID',
                                                values=col).resample('3h',closed='left',
                                                label='right',
                                                ).first().unstack().rolling(window=24, center=False).mean())
    telemetry_mean_24h = pd.concat(temp, axis=1)
    telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
    telemetry_mean_24h.reset_index(inplace=True)
    telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]

    temp = []
    fields = ['volt', 'rotate', 'pressure', 'vibration']
    for col in fields:
        temp.append(pd.pivot_table(telemetry,
                                                index='datetime',
                                                columns='machineID',
                                                values=col).resample('3h',
                                                closed='left',
                                                label='right',
                                                ).first().unstack().rolling(window=24, center=False).std())
    telemetry_sd_24h = pd.concat(temp, axis=1)
    telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
    telemetry_sd_24h.reset_index(inplace=True)
    telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
    telemetry_feat = pd.concat([telemetry_mean_3h,
                                telemetry_sd_3h.iloc[:, 2:6],
                                telemetry_mean_24h.iloc[:, 2:6],
                                telemetry_sd_24h.iloc[:, 2:6]], axis=1).dropna()
    error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()
    error_count.columns = ['datetime', 'machineID','error1', 'error2', 'error3', 'error4', 'error5']
    error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)
    temp = []
    fields = ['error%d' % i for i in range(1, 6)]
    for col in fields:
        temp.append(pd.pivot_table(error_count,
                                    index='datetime',
                                    columns='machineID',
                                    values=col).resample('3h',
                                    closed='left',
                                    label='right',
                                    ).first().unstack().rolling(window=24, center=False).sum())
    error_count = pd.concat(temp, axis=1)
    error_count.columns = [i + 'count' for i in fields]
    error_count.reset_index(inplace=True)
    error_count = error_count.dropna()
    comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()
    comp_rep.columns = ['datetime', 'machineID',
                        'comp1', 'comp2', 'comp3', 'comp4']

    comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                        on=['datetime',
                                                            'machineID'],
                                                        how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])
    components = ['comp1', 'comp2', 'comp3', 'comp4']
    for comp in components:
        comp_rep.loc[comp_rep[comp] < 1, comp] = None
        comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
        comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

    comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]
    for comp in components:
        comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D") 

    

    final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(
        comp_rep, on=['datetime', 'machineID'], how='left')
    final_feat = final_feat.merge(machines, on=['machineID'], how='left')

    labeled_features = final_feat.merge(
        failures, on=['datetime', 'machineID'], how='left')
    labeled_features = labeled_features.fillna(
        method='bfill', limit=7)
    labeled_features = labeled_features.fillna('none')
    labeled_features.to_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/azure.csv'),sep=";")
    return labeled_features
    
def load_azure(preprocessed=False):
    df = pd.DataFrame()
    if preprocessed:
        df =  pd.read_csv(os.path.join(PATH,'microsoft-azure-predictive-maintenance/azure.csv'),sep=";")
    else:
        df = preprocess_azure()

    X = df.iloc[:, :-1]  # Features
    y = df.iloc[:, -1]  # Labels
    feature_names=X.columns
    labels = list(y.unique()) 
    return X, y, feature_names, labels


def prepare_data_and_split(features, labels, sequence_length=10, test_size=0.1, cal_size=0.1):
    
 
    for col in features.select_dtypes(include=['object']).columns:
        features[col] = OrdinalEncoder().fit_transform(features[col].to_numpy().reshape(-1,1))
    #labels = np.argmax(labels, axis=1)
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    # Encode labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)
    if test_size == 0 and cal_size == 0:
        X_gru = features.reshape(features.shape[0], 1, features.shape[1])
        return X_gru, None, None, labels, None, None, features, None, None, num_classes, encoder
    # Split into train, validation, and test sets
    if isinstance(test_size,int):
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=(test_size/len(labels)) + cal_size, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=test_size , random_state=42)
    else:
        X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=test_size + cal_size, random_state=42)
        X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (test_size + cal_size), random_state=42)
    X_train_gru = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_cal_gru = X_cal.reshape(X_cal.shape[0], 1, X_cal.shape[1])
    X_test_gru = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train_gru, X_cal_gru, X_test_gru, y_train, y_cal, y_test, X_train, X_cal, X_test, num_classes, encoder