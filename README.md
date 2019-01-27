# sklearn-lifelines
sklearn estimator wrappers for lifeline survival analysis Cox proportional hazard and Aalen Additive models from https://github.com/CamDavidsonPilon/lifelines 

# Installation

```bash 
pip install git+https://github.com/sashaostr/sklearn-lifelines.git
```

# Example
```python
import lifelines.datasets
import numpy
from patsylearn import PatsyTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn_lifelines.estimators_wrappers import CoxPHFitterModel

# Set seed for reproducible results
numpy.random.seed(42)

data = lifelines.datasets.load_dd()

# create sklearn pipeline
coxph_surv_ppl = make_pipeline(PatsyTransformer('un_continent_name + regime + start_year -1', \
                                              return_type='dataframe'),
                              CoxPHFitterModel(duration_column='duration',event_col='observed',penalizer=0.001))

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
> expected lifetime: 5.075786636895757
> <lifelines.CoxPHFitter: fitted with 1356 observations, 263 censored>
>       duration col = 'duration'
>          event col = 'observed'
> number of subjects = 1356
>   number of events = 1093
>     log-likelihood = -6801.67
>   time fit was run = 2019-01-27 22:01:35 UTC
> ---
>                              coef  exp(coef)  se(coef)     z      p  log(p)  lower 0.95  upper 0.95     
> un_continent_name[Africa]   -0.26       0.77     36.18 -0.01   0.99   -0.01      -71.17       70.65     
> un_continent_name[Americas] -0.06       0.94     36.18 -0.00   1.00   -0.00      -70.97       70.84     
> un_continent_name[Asia]      0.03       1.03     36.18  0.00   1.00   -0.00      -70.88       70.94     
> un_continent_name[Europe]    0.18       1.20     36.18  0.00   1.00   -0.00      -70.73       71.09     
> un_continent_name[Oceania]   0.05       1.05     36.18  0.00   1.00   -0.00      -70.86       70.96     
> regime[T.Military Dict]      0.27       1.31      0.12  2.28   0.02   -3.80        0.04        0.50    .
> regime[T.Mixed Dem]          1.23       3.42      0.13  9.76 <0.005  -50.15        0.98        1.48  ***
> regime[T.Monarchy]          -1.05       0.35      0.25 -4.16 <0.005  -10.36       -1.54       -0.56  ***
> regime[T.Parliamentary Dem]  0.80       2.22      0.11  7.49 <0.005  -30.28        0.59        1.01  ***
> regime[T.Presidential Dem]   1.04       2.82      0.12  8.50 <0.005  -38.51        0.80        1.27  ***
> start_year                  -0.00       1.00      0.00 -1.12   0.26   -1.33       -0.01        0.00     
> ---
> Signif. codes: 0 '***' 0.0001 '**' 0.001 '*' 0.01 '.' 0.05 ' ' 1
> Concordance = 0.65
> Likelihood ratio test = 293.77 on 11 df, log(p)=-128.36
```
