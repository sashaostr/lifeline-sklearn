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
> expected lifetime: 5.0757889745799805
> n=1356, number of events=1093
> 
>                                 coef  exp(coef)     se(coef)       z      p    lower 0.95   upper 0.95     
> un_continent_name[Africa]   -54.2263     0.0000 2723460.7001 -0.0000 1.0000 -5337939.1118 5337830.6593     
> un_continent_name[Americas] -54.0287     0.0000 2723460.7001 -0.0000 1.0000 -5337938.9142 5337830.8568     
> un_continent_name[Asia]     -53.9316     0.0000 2723460.7001 -0.0000 1.0000 -5337938.8172 5337830.9539     
> un_continent_name[Europe]   -53.7841     0.0000 2723460.7001 -0.0000 1.0000 -5337938.6696 5337831.1014     
> un_continent_name[Oceania]  -53.9123     0.0000 2723460.7001 -0.0000 1.0000 -5337938.7978 5337830.9732     
> regime[T.Military Dict]       0.2687     1.3083       0.1177  2.2836 0.0224        0.0381       0.4994    *
> regime[T.Mixed Dem]           1.2298     3.4204       0.1260  9.7610 0.0000        0.9828       1.4767  ***
> regime[T.Monarchy]           -1.0500     0.3499       0.2524 -4.1600 0.0000       -1.5448      -0.5553  ***
> regime[T.Parliamentary Dem]   0.7967     2.2181       0.1064  7.4869 0.0000        0.5881       1.0052  ***
> regime[T.Presidential Dem]    1.0356     2.8168       0.1218  8.5012 0.0000        0.7968       1.2744  ***
> start_year                   -0.0020     0.9980       0.0018 -1.1170 0.2640       -0.0055       0.0015     
> ---
> Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1 
> 
> Concordance = 0.648
> Likelihood ratio test = 293.767 on 11 df, p=0.00000
```
