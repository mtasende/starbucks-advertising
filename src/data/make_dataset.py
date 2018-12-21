import logging
import os
import pickle
import pandas as pd
import src.data.preprocessing as pp
import src.data.success_dataset as sd
from src import DATA_RAW, DATA_INTERIM, DATA_PROCESSED


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making the final datasets from raw data')

    # Load the raw data
    portfolio = pd.read_json(os.path.join(DATA_RAW, 'portfolio.json'),
                             orient='records', lines=True)
    profile = pd.read_json(os.path.join(DATA_RAW, 'profile.json'),
                           orient='records', lines=True)
    transcript = pd.read_json(os.path.join(DATA_RAW, 'transcript.json'),
                              orient='records', lines=True)

    # Initial preprocessing
    logger.info('Preprocessing...')
    data, portfolio = pp.basic_preprocessing(portfolio, profile, transcript)

    # Generate the static dataset, and save it
    logger.info('Generating the static dataset. ' +
                'This may take several minutes...')
    static_dataset_path = os.path.join(DATA_INTERIM, 'static_data.pkl')
    static_data = pp.generate_static_dataset(data)
    static_data.to_pickle(static_dataset_path)

    # Create the offer-success datasets and save them
    logger.info('Creating the offer-success basic datsets...')
    X_train_sd, \
    X_test_sd, \
    y_train_sd, \
    y_test_sd, \
    encoder_sd = sd.get_success_data(static_dataset_path)
    X_train_sd.to_pickle(os.path.join(DATA_PROCESSED, 'X_train_success.pkl'))
    X_test_sd.to_pickle(os.path.join(DATA_PROCESSED, 'X_test_success.pkl'))
    y_train_sd.to_pickle(os.path.join(DATA_PROCESSED, 'y_train_success.pkl'))
    y_test_sd.to_pickle(os.path.join(DATA_PROCESSED, 'y_test_success.pkl'))
    with open(os.path.join(DATA_PROCESSED,
                           'encoder_success.pkl'), 'wb') as file:
        pickle.dump(encoder_sd, file)

    logger.info('All the datasets were created successfully!')



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
