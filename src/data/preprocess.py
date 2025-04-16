import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import random
def data_preprocessing():
    # Load the data
    data = pd.read_csv(r"C:\Users\maite\OneDrive\Bureau\Contributions\Conference\IEEE IWCMC 2025 XGBoost-based faillure prediction\data\raw\MetroPT3(CompressorDatase).csv")
    random.seed(42)
    np.random.seed(42)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Define failure periods
    failure_reports = [
        {'start_time': pd.Timestamp('2020-04-18 00:00:00'), 'end_time': pd.Timestamp('2020-04-18 23:59:59')},
        {'start_time': pd.Timestamp('2020-05-29 23:30:00'), 'end_time': pd.Timestamp('2020-05-30 06:00:00')},
        {'start_time': pd.Timestamp('2020-06-05 10:00:00'), 'end_time': pd.Timestamp('2020-06-07 14:30:00')},
        {'start_time': pd.Timestamp('2020-07-15 14:30:00'), 'end_time': pd.Timestamp('2020-07-15 19:00:00')}
    ]
    failure_reports_df = pd.DataFrame(failure_reports)
    fail_start = pd.to_datetime(failure_reports_df['start_time'])
    fail_end = pd.to_datetime(failure_reports_df['end_time'])

    # Copy and index by timestamp
    df = data.copy()
    df = df.set_index('timestamp')

    # Remove failure periods to keep normal operation data
    df_no_failure = pd.DataFrame(df.loc[((df.index < failure_reports_df['start_time'][0]) | (df.index > failure_reports_df['end_time'][0])) &
                                        ((df.index < failure_reports_df['end_time'][1]) | (df.index > failure_reports_df['end_time'][1])) &
                                        ((df.index < failure_reports_df['end_time'][2]) | (df.index > failure_reports_df['end_time'][2])) &
                                        ((df.index < failure_reports_df['end_time'][3]) | (df.index > failure_reports_df['end_time'][3]))])

    # Create pre-failure periods (3 days before failures)
    fail_ranges = [pd.date_range(start=(fail_start - pd.Timedelta(days=3)), end=fail_start) for fail_start in fail_start]
    failure_df = pd.DataFrame(list(zip(fail_ranges, fail_start, fail_end)), columns=['pre_failure', 'failure_start', 'failure_end'])
    for col in failure_df.columns:
        failure_df = failure_df.explode(col)
    day3_pre = failure_df.set_index(failure_df['pre_failure']).sort_index()
    three_days = pd.Timedelta(3, "D")

    # Merge to identify pre-failure periods
    pre_fail_df = pd.merge_asof(df_no_failure, day3_pre, left_index=True, right_index=True, tolerance=three_days, direction='backward')
    pre_fail_df['pre_fail_binary'] = np.where(np.isnan(pre_fail_df['pre_failure'].values), 0, 1)
    pre_failures_df = pd.DataFrame({'pre_failure': pre_fail_df['pre_fail_binary']})
    new_df = df_no_failure.merge(pre_failures_df, left_index=True, right_index=True)

    # Select relevant features (correlation > 0.17)
    correlation_matrix = new_df.corr()
    correlation_with_target = correlation_matrix['pre_failure'].sort_values(ascending=False)
    relevant_features = correlation_with_target[abs(correlation_with_target) > 0.17].index
    print("Relevant features:", relevant_features)

    # Calculate 4-day rolling averages
    for feature in relevant_features:
        if feature != 'pre_failure':
            new_df[f'{feature}_avg_3day'] = new_df[feature].rolling(window=pd.Timedelta(days=4)).mean().fillna(0)

    # Select numeric features
    numeric_features = new_df.select_dtypes(include=[np.number]).columns
    numeric_features = [feature for feature in numeric_features if feature != 'pre_failure']
    X = new_df[numeric_features]

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Remove highly correlated features (> 0.65)
    corr_matrix = pd.DataFrame(X_scaled, columns=numeric_features).corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.65)]
    selected_features = [feature for feature in numeric_features if feature not in to_drop]
    print("Selected features:", selected_features)
    X_selected = pd.DataFrame(X_scaled, columns=numeric_features)[selected_features]

    # Define X and y
    X = X_selected
    y = new_df['pre_failure']

    # Apply undersampling to balance classes
    undersampler = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, stratify=y_resampled, random_state=42)

    return X_train, X_test, y_train, y_test
