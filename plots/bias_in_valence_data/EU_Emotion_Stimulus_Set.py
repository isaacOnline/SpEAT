import logging
import os

from utils import load_dataset_info
import numpy as np
from scipy.stats import shapiro, mannwhitneyu, boxcox, ttest_ind, shapiro, levene
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from plots.bias_in_valence_data.LinearRegDiagnostics import Linear_Reg_Diagnostic

plt.style.use(os.path.join('plots', 'default.mplstyle'))

if __name__ == '__main__':
    plt.style.use(os.path.join('plots', 'default.mplstyle'))

    log_dir = os.path.join('plots','bias_in_valence_data','EU_Emotion_Stimulus_Set')
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join(os.path.join(log_dir, 'EU_Emotion_Stimulus_Set.log'))
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    data = load_dataset_info('EU_Emotion_Stimulus_Set', group_sizes=3)
    logger.info('Loaded metadata on EU_Emotion_Stimulus_Set')

    # Test gender
    logger.info('Testing gender')

    male_data = data[data['speaker_gender'] == 'Male']['valence']
    female_data = data[data['speaker_gender'] == 'Female']['valence']

    logger.info(f'As there are {len(male_data)} samples from male-identified speakers and {len(female_data)} samples'
                f'from female identified speakers, (meaning the samples are large enough for the CLT to apply),'
                f' a two-sample t-test is appropriate')

    test = sm.stats.ttest_ind(male_data, female_data, usevar='unequal')
    logger.info(f"Welch\'s t-test Stat: {test[0]} P-Value: {test[1]}, DF: {test[2]}")


    # Test age

    dimensions=['valence']
    logger.info(f'Running regression for dimension valence')


    with_intercept = sm.add_constant(data['speaker_age'])
    model = sm.OLS(data['valence'], with_intercept).fit()


    data['residual'] = model.resid
    age_data = {}
    ages = data['speaker_age'].unique()
    for age, age_obs in data.groupby('speaker_age'):
        age_data[age] = age_obs[['valence','residual']]

    results = levene(*[d['residual'] for d in age_data.values()], center='median')

    logger.info('We perform a modified levene test (AKA a brown-forsythe, as we use median instead of mean), to '
                 'evaluate whether the residuals have equal error variance. We do not find evidence that residuals for '
                 f'speakers of different ages have different variances (W={results.statistic}, p={results.pvalue})')

    logger.info('We do not perform a Shapiro-Wilk test for normality of the residuals. The sample size (440) is large'
                 ' enough that the Central Limit Theorem likely applies when performing t-tests.')

    diag = Linear_Reg_Diagnostic(model)
    diag.residual_plot()
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'residuals.pdf'))

    diag.leverage_plot()

    logger.info('When looking at a plot of residuals vs fitted values, no non-linear relationship is suggested. '
                'We do not conduct an F Test for Lack of Fit based on this plot (residuals.pdf).')

    logger.info('As there is no time structure for the dataset, we do not conduct a test for autocorrelation between '
                 'errors')

    logger.info(model.summary())











