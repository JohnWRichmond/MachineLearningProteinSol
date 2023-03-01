import os
import pre_process
from pycaret.regression import *

def experiment(data, target, machine_learning_algorithm):
    setup(data=data,
          target=target,
          train_size=0.8,
          use_gpu=True,

          # Number of folds for k-fold validation
          fold=10,
          session_id=100,
          silent=True)

    model = tune_model(create_model(machine_learning_algorithm), n_iter=500)
    fold_result = pull()

    prediction_result = predict_model(model)

    prediction_result = prediction_result[['log_abundance', 'Label']].set_index('log_abundance')
    prediction_result = prediction_result.rename(columns={'Label': machine_learning_algorithm + '_' + str(len(data.columns) - 1)})

    prediction_metric = pull()
    prediction_metric.insert(loc=1, column='Number_of_features', value =len(data.columns) - 1)

    feature_importance = pd.DataFrame()

    if machine_learning_algorithm in ['br', 'huber', 'ridge', 'lr', 'lar', 'ard', 'tr', 'svm']:
        feature_importance = pd.DataFrame(index=get_config('X_train').columns, data=model.coef_[:], columns=['Feature_importance_' + machine_learning_algorithm + '_' + str(len(data.columns) - 1)])

    if machine_learning_algorithm in ['rf', 'et', 'gbr', 'ada']:
        feature_importance = pd.DataFrame(index=get_config('X_train').columns, data=model.feature_importances_[:], columns=['Feature_importance_' + machine_learning_algorithm + '_' + str(len(data.columns) - 1)])

    return fold_result, prediction_result, prediction_metric, feature_importance

def experiments(config):
    for item in config['datasets'].keys():

        data = pd.read_csv(config['datasets'][item], index_col = 0)
        target = config['target'][item]
        machine_learning_algorithms = config['machine_learning_algorithms']

        processed_data, feature_rankings = pre_process.preprocess(data, target)

        fold_results, prediction_results, prediction_metrics, feature_importances = [], [], [], []

        for machine_learning_algorithm in machine_learning_algorithms:
            for i in range(feature_rankings.max()[0]):
                data_with_feature_selection = processed_data[feature_rankings[feature_rankings <= i+1].dropna().index.values.tolist() + [target]]

                fold_result, prediction_result, prediction_metric, feature_importance = experiment(data_with_feature_selection, target, machine_learning_algorithm)

                fold_results.append(fold_result)
                prediction_results.append(prediction_result)
                prediction_metrics.append(prediction_metric)
                feature_importances.append(feature_importance)

        save_directory = os.path.join(os.path.join(os.getcwd(), 'Results'), item)

        while os.path.exists(save_directory) == True:
            save_directory = save_directory + '1'

        os.mkdir(save_directory)

        pd.concat(feature_importances, axis=1).to_csv(os.path.join(save_directory, 'feature_importances.csv'))
        pd.concat(prediction_metrics, axis=0).to_csv(os.path.join(save_directory, 'prediction_metrics.csv'))
        pd.concat(fold_results, axis=1).to_csv(os.path.join(save_directory, 'fold_results.csv'))
        pd.concat(prediction_results, axis=1).to_csv(os.path.join(save_directory, 'prediction_results.csv'))

    return
