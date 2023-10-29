import logging
import os

import numpy as np
from utils import load_dataset_info
from scipy.stats import shapiro, mannwhitneyu, boxcox
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from plots.bias_in_valence_data.LinearRegDiagnostics import Linear_Reg_Diagnostic

plt.style.use(os.path.join('plots', 'default.mplstyle'))

if __name__ == '__main__':
    log_dir = os.path.join('plots','bias_in_valence_data','morgan_emotional_speech_set')
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(log_dir)
    logger.setLevel(logging.INFO)
    model_log_path = os.path.join(os.path.join(log_dir, 'morgan_emotional_speech_set.log'))
    fh = logging.FileHandler(model_log_path)
    formatter = logging.Formatter(
        '%(asctime)s (%(levelname)s): %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    data = load_dataset_info('morgan_emotional_speech_set', group_sizes=10)
    logger.info('Loaded metadata on morgan_emotional_speech_set')

    # Test gender
    # First test whether it is fair to assume that ratings are normally distributed, using Shapiro-Wilk
    significant_pvalues = 0
    SW_SIGNIFICANCE_LEVEL = 0.05
    logger.info(f'Using significance level {SW_SIGNIFICANCE_LEVEL} for the Shapiro-Wilk test')
    dimensions = ['valence']
    for gender in data['Gender'].unique():
        gender_data = data[data['Gender'] == gender]
        for dimension in dimensions:
            test = shapiro(gender_data[dimension])
            if test.pvalue < SW_SIGNIFICANCE_LEVEL:
                significant_pvalues += 1
            logger.info(f'Gender: {gender}, Dimension: {dimension} Shapiro-Wilk Test Stat: {test.statistic} '
                        f'Shapiro-Wilk P-Value: {test.pvalue}, Sample Size: {len(gender_data)}')
    logger.info(f'{significant_pvalues}/2 tests were significant at the {SW_SIGNIFICANCE_LEVEL} level, implying that '
                'the emotional dimension data cannot be assumed to be normal.')

    # As ratings cannot be assumed to be normally distributed in general, use Mann-Whitney U to evaluate whether medians
    # are equivalent. I used MWU rather than Wilcoxon Signed-Rank Test as the data aren't paired.
    significant_pvalues = 0
    MW_SIGNIFICANCE_LEVEL = 0.05
    for dimension in dimensions:
        male_data = data[data['Gender'] == 'M'][dimension]
        female_data = data[data['Gender'] == 'F'][dimension]

        test = mannwhitneyu(male_data, female_data)
        if test.pvalue < MW_SIGNIFICANCE_LEVEL:
            significant_pvalues += 1
        logger.info(f'Dimension: {dimension} Mann-Whitney U Test Stat: {test.statistic} '
                    f'Mann-Whitney U P-Value: {test.pvalue}, Male SS: {len(male_data)}, Female SS: {len(female_data)}')
    logger.info(f'{significant_pvalues}/1 tests were significant at the {MW_SIGNIFICANCE_LEVEL} level, indicating that '
                f'we do not have evidence that the distributions are not identical')


    # Test age
    for dimension in dimensions:
        logger.info(f'Running regression for dimension {dimension}')

        plt.clf()
        sns.regplot(data['speaker_age'].rename('Speaker Age'),
                    data[dimension].rename(dimension.title()),
                    lowess=False, robust=False,ci=None)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}.pdf'))

        data['Talker'] = data['Talker'].astype(str) + data['Gender']
        with_intercept = pd.concat([pd.get_dummies(data[v], prefix=v, drop_first=True) for v in ['Gender', 'Talker', 'Emotion', 'Cue']],
                       axis=1)

        target = data[dimension]
        target, lmbda = boxcox(data[dimension] - data[dimension].min() + np.abs(data[dimension].min() * 0.00001))


        with_intercept = sm.add_constant(with_intercept)
        model = sm.OLS(target, with_intercept).fit()
        diag = Linear_Reg_Diagnostic(model)
        diag.qq_plot()
        diag.residual_plot()
        logger.info(model.summary())

        plt.clf()
        sm.qqplot(model.resid, fit=True, line='45')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}_qq.pdf'))

        plt.clf()
        fig = sm.graphics.plot_regress_exog(model, 'speaker_age')
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}_residuals.pdf'))


    for dimension in dimensions:
        logger.info(f'Running BOX-COX TRANFORMED regression for dimension {dimension}')

        depend, lamb = boxcox(data[dimension])

        logging.info(f'Lambda in boxcox is {lamb}')
        plt.clf()
        sns.regplot(data['speaker_age'].rename('Speaker Age'),
                    pd.Series(depend, name= dimension.title()),
                    lowess=False, robust=False,ci=None)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}.pdf'))

        with_intercept = sm.add_constant(data['speaker_age'])
        model = sm.OLS(depend, with_intercept).fit()
        logger.info(model.summary())

        plt.clf()
        sm.qqplot(model.resid, fit=True, line='45')
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}_box_cox_qq.pdf'))

        plt.clf()
        fig = sm.graphics.plot_regress_exog(model, 'speaker_age')
        fig.tight_layout()
        plt.savefig(os.path.join(log_dir, f'{dimension}_box_cox_residuals.pdf'))

