import lifelines
import lifelines.datasets
import numpy
from patsylearn import PatsyTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn_lifelines.estimators_wrappers import CoxPHFitterModel


def test_coxph_model():

    # Set seed for reproducible results
    numpy.random.seed(42)

    data = lifelines.datasets.load_dd()

    # create sklearn pipeline
    coxph_surv_ppl = make_pipeline(PatsyTransformer('un_continent_name + regime + start_year -1', return_type='dataframe'),
                                  CoxPHFitterModel(duration_column='duration',event_col='observed',penalizer=0.001))

    #split data to train and test
    data_train, data_test = train_test_split(data)

    #fit CoxPH model
    coxph_surv_ppl.fit(data_train, y=data_train)

    #use pipeline to predict expected lifetime
    exp_lifetime = coxph_surv_ppl.predict(data_test[0:1])
    assert (exp_lifetime > 4)

