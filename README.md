# sklearn-lifelines
sklearn estimator wrappers for lifeline survival analysis Cox proportional hazard and Aalen Additive models from https://github.com/CamDavidsonPilon/lifelines 

# Installation

```bash 
pip install git+https://github.com/sashaostr/sklearn-lifelines.git
```

# Example
```python
import lifelines
import lifelines.datasets
from sklearn_lifelines.estimators_wrappers import CoxPHFitterModel
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from patsylearn import PatsyTransformer

data = lifelines.datasets.load_dd()

# create sklearn pipeline
coxph_surv_ppl = make_pipeline(PatsyTransformer('un_continent_name + regime + start_year -1', \
                                              return_type='dataframe'),
                              CoxPHFitterModel(duration_column='duration',event_col='observed'))

#split data to train and test
data_train, data_test = train_test_split(data)

#fit CoxPH model
coxph_surv_ppl.fit(data_train, y=data_train)

#use pipeline to predict expected lifetime
exp_lifetime = coxph_surv_ppl.predict(data_test[0:1])
print('expected lifetime: ' + str(exp_lifetime))

#or you can extract the model from the pipeline to access more methods
coxmodel = coxph_surv_ppl.named_steps['coxphfittermodel'].estimator
coxmodel.print_summary()
```

```
> expected lifetime: 4.44003472211
>
> n=1356, number of events=1094
>
>                                 coef  exp(coef)  se(coef)          z         p  lower 0.95  upper 0.95     
>un_continent_name[Africa]    6.664e+00  7.834e+02       nan        nan       nan         nan         nan     
>un_continent_name[Americas]  7.375e+00  1.596e+03       nan        nan       nan         nan         nan     
>un_continent_name[Asia]      7.171e+00  1.301e+03       nan        nan       nan         nan         nan     
>un_continent_name[Europe]    8.190e+00  3.605e+03       nan        nan       nan         nan         nan     
>un_continent_name[Oceania]   4.183e+00  6.554e+01       nan        nan       nan         nan         nan     
>regime[T.Military Dict]      8.758e-02  1.092e+00 3.972e-02  2.205e+00 2.747e-02   9.707e-03   1.655e-01    *
>regime[T.Mixed Dem]          4.137e-01  1.512e+00 4.480e-02  9.235e+00 2.588e-20   3.259e-01   5.015e-01  ***
>regime[T.Monarchy]          -1.924e-01  8.250e-01 4.860e-02 -3.959e+00 7.539e-05  -2.877e-01  -9.712e-02  ***
>regime[T.Parliamentary Dem]  3.740e-01  1.453e+00 4.931e-02  7.584e+00 3.356e-14   2.773e-01   4.706e-01  ***
>regime[T.Presidential Dem]   3.440e-01  1.411e+00 4.660e-02  7.383e+00 1.546e-13   2.527e-01   4.354e-01  ***
>start_year                  -4.382e-02  9.571e-01 3.351e-02 -1.308e+00 1.910e-01  -1.095e-01   2.188e-02     
>
>Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
>
>Concordance = 0.654
```
