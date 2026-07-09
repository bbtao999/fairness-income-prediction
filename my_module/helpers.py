import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

def load_ICU_data_fixed(path):
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'martial_status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']

    input_data = (pd.read_csv(path, names=column_names, header=None,
                              na_values="?", sep=r'\s*,\s*', engine='python') #the raw file has spaces after every comma
                  .loc[lambda df: df['race'].isin(['White', 'Black'])])

    # Sensitive attributes; identify 'race' and 'sex' as sensitive attributes
    sensitive_attribs = ['race', 'sex']
    Z = (input_data.loc[:, sensitive_attribs]
         .assign(race=lambda df: (df['race'] == 'White').astype(int),
                 sex=lambda df: (df['sex'] == 'Male').astype(int)))

    # Targets; 1 when someone makes over 50k, otherwise 0
    y = (input_data['target'] == '>50K').astype(int)

    # Features; note that the 'target' and sensitive attribute columns are dropped
    #If the data has a column like occupation with some missing values, 
    # filling them with 'Unknown' means pd.get_dummies will generate a new binary column called occupation_Unknown.
    #otw, pd.get_dummies ignores NaN values (unless you pass dummy_na=True
    X_clean = input_data.drop(columns=['target', 'race', 'sex', 'fnlwgt']).fillna('Unknown')
    #convert the categorical features to dummy variables, the numeric stays as it is
    X = pd.get_dummies(X_clean, drop_first=True)
    return X, y, Z



def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.min([odds, 1/odds]) * 100


def plot_distributions(y, Z, iteration=None, val_metrics=None, p_rules=None, fname=None):
    #sharey=True: This argument specifies that the subplots should share the same y-axis. 
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    legend={'race': ['black','white'],
            'sex': ['female','male']}
    
    # create kernel density estimates (KDE) plots with the seaborn library
    #probability density function of a continuous random variable.

    #enumerate is used to keep track of the index (idx) and the name of the column (attr)
    #This loop iterates over each column in the DataFrame Z

    for idx, attr in enumerate(Z.columns): #the idx shows the column number
        #Then a nested loop iterates over a list of attribute values, specifically [0, 1]. 
        for attr_val in [0, 1]:
            ax = sns.kdeplot(data=y[Z[attr] == attr_val], #here, y is yhat_test
                             label=f'{legend[attr][attr_val]}',  #legend is a dict defined above
                             ax=axes[idx], fill=True) #This specifies which subplot to draw the KDE plot in, using the index idx.
        ax.set_xlim(0,1)
        ax.set_ylim(0,7)
        ax.set_yticks([])
        ax.set_title(f'sensitive attibute: {attr}')
        if idx == 0:
            ax.set_ylabel('prediction distribution')
        ax.set_xlabel(f'Prob(income>50K)|z_{attr}')
    if iteration:
        fig.text(1.0, 0.9, f"Training iteration #{iteration}", fontsize='16')
    if val_metrics is not None:
        fig.text(1.0, 0.65, '\n'.join(["Prediction performance:",
                                       f"- ROC AUC: {val_metrics['ROC AUC']:.2f}",
                                       f"- Accuracy: {val_metrics['Accuracy']:.1f}"]),
                 fontsize='16')
    if p_rules is not None:
        fig.text(1.0, 0.4, '\n'.join(["Satisfied p%-rules:"] +
                                     [f"- {attr}: {p_rules[attr]:.0f}%-rule" 
                                      for attr in p_rules.keys()]), 
                 fontsize='16')
    fig.tight_layout()
    if fname is not None:
        plt.savefig(fname, bbox_inches='tight')
    return fig

