
# In-processing fairness for gradient-boosted trees via a CUSTOM OBJECTIVE.
#
# We keep a SINGLE
# output (income) and put the fairness pressure directly into the loss:
#
#   Loss = BCE(y, p) + sum_k lambda_k * ( mean(p | Z_k=1) - mean(p | Z_k=0) )^2
#
# The penalty is the squared demographic-parity gap in average predicted
# probability - the same disparity the p%-rule measures as a ratio. It is
# differentiable, so we can hand LightGBM its gradient and hessian.
#
# Derivation (f = raw score/logit, p = sigmoid(f)):
#   BCE part:      dL/df_i = p_i - y_i          d2L/df_i2 = p_i (1 - p_i)
#   Penalty part:  d/df_i [ gap^2 ] = 2 * gap * (+-1/n_group(i)) * p_i (1 - p_i)
#                  (+ if sample i is in group Z=1, - if in group Z=0)
# We keep the BCE hessian only: the penalty's hessian contribution is O(1/n)
# per sample and may be negative, which LightGBM's Newton step cannot use.
#
# The sensitive attributes are used ONLY inside the training loss - they are
# never features, and nothing about the model changes at prediction time.

import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

os.chdir(Path(__file__).resolve().parent)
from my_module import helpers

np.random.seed(7)

#----------------------------#
#     Read in Data           #
#----------------------------#
X, y, Z = helpers.load_ICU_data_fixed('data/adult.data')

# identical split to the other two scripts
(X_train, X_test, y_train, y_test,
 Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5,
                                     stratify=y, random_state=7)

def sigmoid(f):
    return 1.0 / (1.0 + np.exp(-np.clip(f, -30, 30)))


def make_fair_objective(Z_train, lambdas):
    '''Build a LightGBM custom objective (sklearn API signature) with the
    demographic-parity penalty for each sensitive attribute in Z_train.
    lambdas: dict {attribute name: penalty weight}.'''
    # closure over the group masks; sklearn API keeps training-row order
    masks = {attr: Z_train[attr].values.astype(bool) for attr in Z_train.columns}

    def fair_objective(y_true, f_pred):
        p = sigmoid(f_pred)
        grad = p - y_true                 # BCE gradient wrt the logit
        hess = np.maximum(p * (1 - p), 1e-6)  # BCE hessian, clamped for stability

        for attr, lam in lambdas.items():
            m = masks[attr]
            gap = p[m].mean() - p[~m].mean()
            # chain rule: through the group mean, then through the sigmoid
            sign_over_n = np.where(m, 1.0 / m.sum(), -1.0 / (~m).sum())
            grad += lam * 2.0 * gap * sign_over_n * p * (1 - p)
        return grad, hess

    return fair_objective


def evaluate(name, scores, fname=None):
    '''Same metric set as the other scripts; scores are P(income>50K) on test.'''
    scores = pd.Series(np.asarray(scores).ravel(), index=y_test.index)
    res = {
        'model': name,
        'roc_auc': metrics.roc_auc_score(y_test, scores),
        'accuracy': 100 * metrics.accuracy_score(y_test, scores > 0.5),
        'p_rule_race': helpers.p_rule(scores, Z_test['race']),
        'p_rule_sex': helpers.p_rule(scores, Z_test['sex']),
    }
    print(f"{name:>14} | ROC AUC {res['roc_auc']:.3f} | acc {res['accuracy']:.1f}% "
          f"| p-rule race {res['p_rule_race']:.0f}% sex {res['p_rule_sex']:.0f}%")
    if fname:
        fig = helpers.plot_distributions(scores, Z_test,
                                         val_metrics={'ROC AUC': res['roc_auc'],
                                                      'Accuracy': res['accuracy']},
                                         p_rules={'race': res['p_rule_race'],
                                                  'sex': res['p_rule_sex']},
                                         fname=fname)
        plt.close(fig)
    return res


def fit_lgbm(objective):
    model = lgb.LGBMRegressor(objective=objective, n_estimators=200,
                              learning_rate=0.1, num_leaves=31,
                              random_state=7, verbose=-1)
    model.fit(X_train, y_train)
    return model


#--------------------------------------------------------#
# Sweep the penalty weight lambda                         #
#--------------------------------------------------------#
# The per-sample penalty gradient carries a 1/n_group factor (n ~ 1.5-14k),
# so lambda must be large before the penalty competes with the BCE gradient.
# Race needs a heavier weight than sex (same finding as the adversarial net,
# which used lambdas [130, 30]). We sweep and keep the smallest setting
# satisfying both 80% p-rules.
LAMBDAS = [None,  # None = plain unpenalized baseline
           {'race': 1e5, 'sex': 1e5},
           {'race': 3e5, 'sex': 1e5},
           {'race': 6e5, 'sex': 1.5e5},
           {'race': 1e6, 'sex': 2.5e5},
           {'race': 1.5e6, 'sex': 4e5}]

results = []
for lam in LAMBDAS:
    name = 'baseline' if lam is None else \
        f"lambda race={lam['race']:g} sex={lam['sex']:g}"
    if lam is None:
        obj = 'binary'  # plain BCE; LGBMRegressor+custom would be identical
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1,
                                   num_leaves=31, random_state=7, verbose=-1)
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_test)[:, 1]
    else:
        objective = make_fair_objective(Z_train, lam)
        model = fit_lgbm(objective)
        scores = sigmoid(model.predict(X_test))  # custom objective -> raw logits
    res = evaluate(name, scores)
    res['lambda'] = 'none' if lam is None else f"{lam['race']:g}/{lam['sex']:g}"
    res['scores'] = scores
    results.append(res)

# smallest lambda that satisfies both p%-rules
fair = next((r for r in results[1:]
             if r['p_rule_race'] >= 80 and r['p_rule_sex'] >= 80),
            results[-1])
print(f"\nselected: {fair['model']}")

# save distribution plots for the baseline and the selected fair model
evaluate('baseline (plot)', results[0]['scores'], fname='images/lgbm_biased.png')
evaluate(f"fair {fair['model']} (plot)", fair['scores'], fname='images/lgbm_fair.png')

summary = pd.DataFrame([{k: v for k, v in r.items() if k != 'scores'}
                        for r in results]).set_index('model')
summary.to_csv('lgbm_customloss_metrics.csv')
print('\nsaved metrics to lgbm_customloss_metrics.csv and plots to images/lgbm_*.png')
