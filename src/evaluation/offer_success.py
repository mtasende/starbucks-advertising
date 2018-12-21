"""
Contains functions and classes that generate validations sets, and validate
models for the 'offer success' problem.
"""
import src.data.success_dataset as sd
import src.utils as utils


def get_time_split_val(val_time=370, **kwargs):
    """
    Returns all the datasets necessary to perform a time-split validation.
    Args:
        val_time(int): The time to make the validation split.
        kwargs(dict): Arguments to be passed to inner functions.

    Returns:
        X_train(pd.DataFrame): Training features.
        X_val(pd.DataFrame): Validation features.
        X_test(pd.DataFrame): Test features.
        X_train_val(pd.DataFrame): Training + Validation features, to use when
            testing.
        y_train(pd.Series): Training target values.
        y_val(pd.Series): Validation target values.
        y_test(pd.Series): Test target values.
        y_train_val(pd.Series): Training + Validation target values, to use
        when testing.
    """

    fun_kwargs = utils.filter_args(sd.get_success_data, kwargs)
    X_train_val, \
    X_test, \
    y_train_val, \
    y_test, \
    encoder = sd.get_success_data(drop_time=False, **fun_kwargs)
    X_test = sd.drop_time_dependent(X_test)
    X_train, X_val, y_train, y_val = sd.time_split(X_train_val, y_train_val,
                                                   val_time)
    return X_train, X_val, X_test, X_train_val, y_train, y_val, y_test, \
           y_train_val
