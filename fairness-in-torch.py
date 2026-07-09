
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # render plots to files, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# run from the script's own folder so relative paths (data/, images/) work anywhere
os.chdir(Path(__file__).resolve().parent)
from my_module import helpers
from my_module import models

torch.manual_seed(1) 
np.random.seed(123)



#----------------------------#
#     Read in Data           #
#----------------------------#
# load ICU data set
X, y, Z = helpers.load_ICU_data_fixed('data/adult.data')

n_features = X.shape[1]
n_sensitive = Z.shape[1]

# split into train/test set
(X_train, X_test, y_train, y_test,
 Z_train, Z_test) = train_test_split(X, y, Z, test_size=0.5,
                                     stratify=y, random_state=7)

# standardize the data
#fit first and then transform
# Configure the scaler to return pandas dataframes
scaler = StandardScaler().set_output(transform="pandas") #the output is pandas df so that we can use the PandasDataSet function below to convert it to tensor
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# convert to PyTorch tensors
train_data = models.PandasDataSet(X_train, y_train, Z_train)
test_data = models.PandasDataSet(X_test, y_test, Z_test)

# Create a `DataLoader` that returns SHUFFLED batches of our training set:
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)

print('# training samples:', len(train_data))
print('# batches:', len(train_loader))


# ## Income predictions
# With our datasets in place, we define and pretrain the classifier to make income predictions.
# This classifier will be good in predicting income level but is likely to be unfair - it is only penalized on performance and not on fairness.

#-----------------------------------#
# Time to pretrain the classifier!
#-----------------------------------#

# Initialize the classifier,
#choose binary cross entropy as the loss function
#and let Adam optimize the weights of the classifier:

clf = models.Classifier(n_features=n_features)
clf_criterion = nn.BCELoss()
clf_optimizer = optim.Adam(clf.parameters())

N_CLF_EPOCHS = 10

for epoch in range(N_CLF_EPOCHS):
    clf = models.pretrain_classifier(clf,
                            train_loader,
                            clf_optimizer,
                            clf_criterion)
print(f'pretrained classifier for {N_CLF_EPOCHS} epochs')

#-----------------------------------#
# Detecting unfairness Visually     #
#-----------------------------------#
# With the classifier pretrained, we now define and pretrain the adversary.
# Similar to the classifier, our adversary consists of THREE layers.
# However, the input comes from a SINGLE class (the predicted income class) and the output consists of two sensitive classes (sex and race).

# For our final solution, there will be a trade-off between classifier performance and fairness for our sensitive attributes.
# We will tweak the adversarial loss to incorporate that trade-off: the lambda parameter weighs the adversarial loss of each class.
# This parameter is later also used to scale the adversary performance versus the classifier performance.

# By telling `nn.BCELoss` not to reduce we get the losses for each individual obs and class instead of a single number.
# Multiplying this with our `lambdas` and taking the average, gives us the weighted adversarial loss, our proxy for unfairness.
# getting race-blind is ~4× more important per unit of adversary loss than getting sex-blind — and both are worth a lot compared to raw accuracy (both ≫ 1)
#90/10 imbalance means the adversary's race BCE barely changes between "detecting" and "blind," so each unit of λ buys less fairness pressure for race than for sex. 
# λ has to compensate for how little room the loss has to move
lambdas = torch.Tensor([130, 30]) #race, gender
adv = models.Adversary(Z_train.shape[1])

adv_criterion = nn.BCELoss(reduction='none') #reduction='none' returns the loss for each sample
adv_optimizer = optim.Adam(adv.parameters())

N_ADV_EPOCHS = 50

for epoch in range(N_ADV_EPOCHS):
    models.pretrain_adversary(adv,
                       clf,
                       train_loader,
                       adv_optimizer,
                       adv_criterion,
                       lambdas)
print(f'pretrained adversary for {N_ADV_EPOCHS} epochs')



# Set the model to evaluation mode
#By setting the model to evaluation mode with .eval(),
#dropout layers deactivate the dropping out of neurons and
#use the full capabilities of the network for prediction.
clf.eval()
adv.eval()

# Disable gradient computation since we are only predicting, not training
with torch.no_grad():
    pre_clf_test = clf(test_data.tensors[0])
    pre_adv_test = adv(pre_clf_test)

y_pre_clf_test = pd.Series(pre_clf_test.numpy().reshape(-1),
                      index=y_test.index)  #so to use custom fairness calculations
#By converting the predictions into a pandas Series and explicitly giving it index=y_test.index, 
# can ensure that pandas aligns the rows perfectly when it filters the data based on z_values==1 in the helper.p_rule function

# Calculate ROC AUC and Accuracy of the biased (pre-fairness) classifier
roc_auc = metrics.roc_auc_score(y_test, y_pre_clf_test)
accuracy = 100 * metrics.accuracy_score(y_test, (y_pre_clf_test > 0.5))
print(f"BIASED classifier -- ROC AUC: {roc_auc:.2f}")
print(f"BIASED classifier -- Accuracy: {accuracy:.1f}%")

# How well can the adversary already reconstruct the sensitive attributes?
adv_auc_race = metrics.roc_auc_score(Z_test['race'], pre_adv_test.numpy()[:, 0])
adv_auc_sex = metrics.roc_auc_score(Z_test['sex'], pre_adv_test.numpy()[:, 1])
print(f"Adversary ROC AUC -- race: {adv_auc_race:.2f}, sex: {adv_auc_sex:.2f}")

#--------------------------------------#
# Detecting unfairness quantitatively  #
#--------------------------------------#

p_rule_race = helpers.p_rule(y_pre_clf_test, Z_test['race'])
p_rule_sex = helpers.p_rule(y_pre_clf_test, Z_test['sex'])
print("The classifier satisfies the following %p-rules:")
print(f"\tgiven attribute race; {p_rule_race:.0f}%-rule")
print(f"\tgiven attribute sex;  {p_rule_sex:.0f}%-rule")

fig = helpers.plot_distributions(y_pre_clf_test, Z_test,
                                 val_metrics={'ROC AUC': roc_auc, 'Accuracy': accuracy},
                                 p_rules={'race': p_rule_race, 'sex': p_rule_sex},
                                 fname='images/biased_training.png')
plt.close(fig)

#--------------------------------------#
# ## Training for fairness
#--------------------------------------#

# Now that we have an unfair classifier and an adversary that is able to pick up on unfairness,
# we can engage them in the zero-sum game to make the classifier fair.



# The loss function for the classifier is changed to its original loss
# plus the weighted negative adversarial loss.

#-----------------------------------------------#
# Full Fairness Model Training and Evaluation   #
#-----------------------------------------------#
N_EPOCH_COMBINED = 165
PLOT_EVERY = 20  # save a distribution snapshot every N epochs (plus the final one)

metrics_history = []

for epoch in range(1, N_EPOCH_COMBINED):

    # dropout must be active while training and off while evaluating
    clf.train()
    adv.train()

    # pass the exact same clf and adv Python objects into fairness_train without re-initializing them, 
    # so PyTorch just continues updating their existing weights.
    clf, adv = models.fairness_train(clf, adv, train_loader,
                     clf_criterion, adv_criterion,
                     clf_optimizer, adv_optimizer, lambdas)

    #for predictions on the test data, we need to set the models to evaluation mode and disable gradient computation
    clf.eval()
    adv.eval()
    with torch.no_grad():
        clf_pred = clf(test_data.tensors[0])
        adv_pred = adv(clf_pred)

    y_post_clf = pd.Series(clf_pred.numpy().ravel(), index=y_test.index)
    Z_post_adv = pd.DataFrame(adv_pred.numpy(), columns=Z_test.columns)

    # track metrics for every epoch
    epoch_metrics = {
        'epoch': epoch,
        'clf_roc_auc': metrics.roc_auc_score(y_test, y_post_clf),
        'clf_accuracy': 100 * metrics.accuracy_score(y_test, y_post_clf > 0.5),
        'adv_roc_auc_race': metrics.roc_auc_score(Z_test['race'], Z_post_adv['race']),
        'adv_roc_auc_sex': metrics.roc_auc_score(Z_test['sex'], Z_post_adv['sex']),
        'p_rule_race': helpers.p_rule(y_post_clf, Z_test['race']),
        'p_rule_sex': helpers.p_rule(y_post_clf, Z_test['sex']),
    }
    metrics_history.append(epoch_metrics)

    if epoch % PLOT_EVERY == 0 or epoch == N_EPOCH_COMBINED - 1:
        fig = helpers.plot_distributions(
            y_post_clf, Z_test, iteration=epoch,
            val_metrics={'ROC AUC': epoch_metrics['clf_roc_auc'],
                         'Accuracy': epoch_metrics['clf_accuracy']},
            p_rules={'race': epoch_metrics['p_rule_race'],
                     'sex': epoch_metrics['p_rule_sex']},
            fname=f'images/torch_{epoch:08d}.png')
        plt.close(fig)
        print(f"epoch {epoch:3d} | clf AUC {epoch_metrics['clf_roc_auc']:.3f} "
              f"| acc {epoch_metrics['clf_accuracy']:.1f}% "
              f"| adv AUC race {epoch_metrics['adv_roc_auc_race']:.2f} sex {epoch_metrics['adv_roc_auc_sex']:.2f} "
              f"| p-rule race {epoch_metrics['p_rule_race']:.0f}% sex {epoch_metrics['p_rule_sex']:.0f}%")

metrics_df = pd.DataFrame(metrics_history).set_index('epoch')
metrics_df.to_csv('training_metrics.csv')

#-------------------------------------------------------#
# Plot test set metric trajectories over training        #
#-------------------------------------------------------#
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(metrics_df.index, metrics_df['clf_roc_auc'], label='classifier ROC AUC')
axes[0].plot(metrics_df.index, metrics_df['clf_accuracy'] / 100, label='classifier accuracy')
axes[0].set_title('Classifier performance')
axes[0].set_xlabel('epoch'); axes[0].legend()

axes[1].plot(metrics_df.index, metrics_df['adv_roc_auc_race'], label='race')
axes[1].plot(metrics_df.index, metrics_df['adv_roc_auc_sex'], label='sex')
axes[1].axhline(0.5, color='grey', ls='--', label='0.5 = cannot detect')
axes[1].set_title('Adversary ROC AUC')
axes[1].set_xlabel('epoch'); axes[1].legend()

axes[2].plot(metrics_df.index, metrics_df['p_rule_race'], label='race')
axes[2].plot(metrics_df.index, metrics_df['p_rule_sex'], label='sex')
axes[2].axhline(80, color='grey', ls='--', label='80% threshold')
axes[2].set_title('p%-rules')
axes[2].set_xlabel('epoch'); axes[2].legend()
fig.tight_layout()
fig.savefig('images/NN_test_metrics.png', bbox_inches='tight', dpi=300)
plt.close(fig)

final = metrics_df.iloc[-1]
print("\n===== FINAL (fair) classifier =====")
print(f"ROC AUC: {final['clf_roc_auc']:.2f} (biased baseline: {roc_auc:.2f})")
print(f"Accuracy: {final['clf_accuracy']:.1f}% (biased baseline: {accuracy:.1f}%)")
print(f"Adversary ROC AUC -- race: {final['adv_roc_auc_race']:.2f}, sex: {final['adv_roc_auc_sex']:.2f}")
print(f"p%-rules -- race: {final['p_rule_race']:.0f}% (was {p_rule_race:.0f}%), "
      f"sex: {final['p_rule_sex']:.0f}% (was {p_rule_sex:.0f}%)")

# The classifier starts off unfair, but trades some of its performance for fairness.
# At the end of training, the ROC AUC of the adversary is ~0.50, indicating that it's unable to detect race or gender from the made predictions.
# That's also shown by the p-rules: they're both above 80%.
# We've successfully used an adversarial neural network to make our classifier fair!

# ## Conclusion
# (full write-up with the tree-based comparisons: RESULTS.md and summary.md)
#
# The unconstrained classifier was accurate but clearly unfair: ROC AUC 0.91,
# accuracy 85.1%, but p%-rules of only 45% (race) and 36% (sex) - far below
# the 80% four-fifths threshold - and an adversary could detect sex from the
# predictions alone with ROC AUC 0.70.
#
# After 165 epochs of the zero-sum game, the classifier trades performance for
# fairness: ROC AUC 0.82, accuracy 81.0%, p%-rules 90% (race) and 89% (sex),
# and the adversary is reduced to chance (ROC AUC ~0.51) - the predictions no
# longer carry recoverable information about race or sex.
#
# Sex fairness was achieved by ~epoch 40 while race took until ~epoch 70 and
# needed a 4x larger lambda (130 vs 30): with a 90/10 race imbalance the
# minority group supplies weak training signal, so race fairness is the
# expensive, binding constraint on this dataset. Compared with the tree-based
# alternatives (fairlearn reduction, LightGBM custom loss), this adversarial
# approach pays the most accuracy but achieves the widest fairness margins and
# keeps a smooth, usable score distribution.
