
# Tree-based fair classifier on the same Adult data set, for comparison with
# the adversarial PyTorch approach in fairness-in-torch.py.
#
# Baseline: sklearn HistGradientBoostingClassifier (gradient-boosted trees).
# Fair version: fairlearn's ExponentiatedGradient reduction with a
# DemographicParity constraint. The reduction repeatedly reweights the training
# data and refits the tree model, searching for the best accuracy achievable
# under the fairness constraint - no change to the tree model itself is needed.
#
# DemographicParity targets the same disparity as the p%-rule: the rate of
# positive predictions must not depend on the sensitive attribute.

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio

os.chdir(Path(__file__).resolve().parent)
from my_module import helpers

np.random.seed(7)

#----------------------------#
#     Read in Data           #
#----------------------------#
X, y, Z = helpers.load_ICU_data_fixed('data/adult.data')

# identical split to fairness-in-torch.py (same test_size, stratify and seed)
(X_train, X_test, y_train, y_test,
 Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5,
                                     stratify=y, random_state=7)

# note: no standardization needed - tree splits are invariant to feature scaling


def evaluate(name, scores, fname):
    '''Compute the same metric set used for the neural nets and save the
    prediction-distribution plot. `scores` are P(income>50K) on the test set.'''
    scores = pd.Series(np.asarray(scores).ravel(), index=y_test.index)
    y_hat = scores > 0.5
    res = {
        'model': name,
        'roc_auc': metrics.roc_auc_score(y_test, scores),
        'accuracy': 100 * metrics.accuracy_score(y_test, y_hat),
        'p_rule_race': helpers.p_rule(scores, Z_test['race']),
        'p_rule_sex': helpers.p_rule(scores, Z_test['sex']),
        # fairlearn's own fairness metrics on the hard predictions:
        # difference of positive rates between groups (0 = perfectly fair)
        'dp_diff_race': demographic_parity_difference(y_test, y_hat, sensitive_features=Z_test['race']),
        'dp_diff_sex': demographic_parity_difference(y_test, y_hat, sensitive_features=Z_test['sex']),
        # ratio of positive rates between groups (1 = perfectly fair; = p-rule/100)
        'dp_ratio_race': demographic_parity_ratio(y_test, y_hat, sensitive_features=Z_test['race']),
        'dp_ratio_sex': demographic_parity_ratio(y_test, y_hat, sensitive_features=Z_test['sex']),
    }
    print(f"\n===== {name} =====")
    print(f"ROC AUC: {res['roc_auc']:.2f} | Accuracy: {res['accuracy']:.1f}%")
    print(f"p%-rules -- race: {res['p_rule_race']:.0f}%, sex: {res['p_rule_sex']:.0f}%")
    print(f"demographic parity diff -- race: {res['dp_diff_race']:.3f}, sex: {res['dp_diff_sex']:.3f}")

    fig = helpers.plot_distributions(scores, Z_test,
                                     val_metrics={'ROC AUC': res['roc_auc'],
                                                  'Accuracy': res['accuracy']},
                                     p_rules={'race': res['p_rule_race'],
                                              'sex': res['p_rule_sex']},
                                     fname=fname)
    plt.close(fig)
    return res


results = []

#-----------------------------------#
# Baseline gradient-boosted trees   #
#-----------------------------------#
gbt = HistGradientBoostingClassifier(random_state=7)
gbt.fit(X_train, y_train)
results.append(evaluate('baseline GBT', gbt.predict_proba(X_test)[:, 1],
                        'images/tree_biased.png'))

#--------------------------------------------------#
# Fair version: ExponentiatedGradient reduction    #
#--------------------------------------------------#
# The constraint is applied to BOTH sensitive attributes at once by passing the
# two-column Z DataFrame: fairlearn constrains every race x sex subgroup.
fair_gbt = ExponentiatedGradient(
    estimator=HistGradientBoostingClassifier(random_state=7),
    constraints=DemographicParity(),
)
fair_gbt.fit(X_train, y_train, sensitive_features=Z_train)

# The reduction produces a randomized ensemble of tree models; _pmf_predict
# gives the ensemble probability of the positive class, which plays the same
# role as the sigmoid output of the neural net.
fair_scores = fair_gbt._pmf_predict(X_test)[:, 1]
results.append(evaluate('fair GBT (ExponentiatedGradient + DemographicParity)',
                        fair_scores, 'images/tree_fair.png'))

pd.DataFrame(results).set_index('model').to_csv('tree_metrics.csv')
print('\nsaved metrics to tree_metrics.csv and plots to images/tree_*.png')
