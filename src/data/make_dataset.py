import logging
import os
import pickle
import pandas as pd
import src.data.preprocessing as pp
import src.data.success_dataset as sd
import src.features.clustering as clust
import src.features.lagged as lag
import src.data.profit_10_days_dataset as p10
from src.data import DATA_RAW, DATA_INTERIM, DATA_PROCESSED


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    static_dataset_path = os.path.join(DATA_INTERIM, 'static_data.pkl')
    static_cluster1_path = os.path.join(DATA_PROCESSED, 'static_cluster1.pkl')
    static_cluster3d_path = os.path.join(DATA_PROCESSED, 'static_cluster3d.pkl')
    static_lagged_path = os.path.join(DATA_PROCESSED, 'static_cluster_lagged.pkl')
    static_spent_10_days = os.path.join(DATA_PROCESSED, 'static_spent_10_days.pkl')


    logger = logging.getLogger(__name__)
    logger.info('Making the final datasets from raw data (the entire process can take about 1 hour, more or less, '
                'depending on the computational resources available)')

    # Load the raw data
    print('data raw is here:')
    print(os.path.join(DATA_RAW, 'portfolio.json'))
    portfolio = pd.read_json(os.path.join(DATA_RAW, 'portfolio.json'),
                             orient='records', lines=True)
    profile = pd.read_json(os.path.join(DATA_RAW, 'profile.json'),
                           orient='records', lines=True)
    transcript   = pd.read_json(os.path.join(DATA_RAW, 'transcript.json'),
                              orient='records', lines=True)

    # Initial preprocessing
    logger.info('Preprocessing...')
    data, portfolio = pp.basic_preprocessing(portfolio, profile, transcript)

    # Generate the static dataset, and save it
    logger.info('Generating the static dataset. ' +
                'This may take several minutes...')
    static_data = pp.generate_static_dataset(data)
    static_data.to_pickle(static_dataset_path)

    # Add the 4D clustering features
    logger.info('Generating the 4D clustering features')
    clust.create_cluster_feats_4d(static_dataset_path=static_dataset_path,
                                  output_path=static_cluster1_path,
                                  save=True)

    # Add the 3D clustering features
    logger.info('Generating the 3D clustering features')
    clust.create_cluster_feats_3d(static_dataset_path=static_cluster1_path,
                                  output_path=static_cluster3d_path,
                                  save=True)

    # Add the lagged features
    logger.info('Generating the Lagged features')
    portfolio = pd.read_json(os.path.join(DATA_RAW, 'portfolio.json'),
                             orient='records', lines=True)
    static_data = pd.read_pickle(static_cluster3d_path)
    data_lag = lag.fill_lagged_success(static_data, portfolio)
    data_lag.to_pickle(static_lagged_path)

    # Create the offer-success datasets and save them
    logger.info('Creating the offer-success datsets...')
    X_train_sd, \
    X_test_sd, \
    y_train_sd, \
    y_test_sd, \
    encoder_sd = sd.get_success_data(basic_dataset_path=static_lagged_path)
    X_train_sd.to_pickle(os.path.join(DATA_PROCESSED, 'X_train_success.pkl'))
    X_test_sd.to_pickle(os.path.join(DATA_PROCESSED, 'X_test_success.pkl'))
    y_train_sd.to_pickle(os.path.join(DATA_PROCESSED, 'y_train_success.pkl'))
    y_test_sd.to_pickle(os.path.join(DATA_PROCESSED, 'y_test_success.pkl'))
    with open(os.path.join(DATA_PROCESSED,
                           'encoder_success.pkl'), 'wb') as file:
        pickle.dump(encoder_sd, file)

    # Create spent-10-days static dataset
    logger.info('Creating the spent-10-days static datset')
    static_data = pd.read_pickle(static_lagged_path)
    filled = p10.get_spent_days_static(static_data, data)
    filled.to_pickle(static_spent_10_days)

    # Create the profit-10-days datasets and save them
    logger.info('Creating the profit-10-days datsets...')
    X_train_p10,\
    X_test_p10,\
    y_train_p10,\
    y_test_p10,\
    encoder_p10,\
    view_cols_p10,\
    profit_cols_p10 = p10.get_profit_10_days_data(basic_dataset_path=static_spent_10_days,
                                                  fill_null=True,
                                                  target=['viewed', 'profit_10_days'],
                                                  drop_offer_id=False)
    X_train_p10.to_pickle(os.path.join(DATA_PROCESSED, 'X_train_success.pkl'))
    X_test_p10.to_pickle(os.path.join(DATA_PROCESSED, 'X_test_success.pkl'))
    y_train_p10.to_pickle(os.path.join(DATA_PROCESSED, 'y_train_success.pkl'))
    y_test_p10.to_pickle(os.path.join(DATA_PROCESSED, 'y_test_success.pkl'))
    with open(os.path.join(DATA_PROCESSED,
                           'encoder_profits.pkl'), 'wb') as file:
        pickle.dump(encoder_p10, file)
    with open(os.path.join(DATA_PROCESSED,
                           'view_cols_profits.pkl'), 'wb') as file:
        pickle.dump(view_cols_p10, file)
    with open(os.path.join(DATA_PROCESSED,
                           'profit_cols_profits.pkl'), 'wb') as file:
        pickle.dump(profit_cols_p10, file)

    logger.info('All the datasets were created successfully!')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
