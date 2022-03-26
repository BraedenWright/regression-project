import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing


# Functions I've used to explore data



def plot_categorical_and_continuous_vars(cat_var, cont_var, df):
    '''
    This function accepts your dataframe and the name of the columns that hold 
    the continuous and categorical features and outputs 3 different plots 
    for visualizing a categorical variable and a continuous variable.
    '''
    
    # Title
    plt.suptitle(f'{cont_var} by {cat_var}')
                 
    # Lineplot
    sns.lineplot(x=cat_var, y=cont_var, data=df)
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Swarm Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='swarm', palette='Greens')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Box Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='box', palette='Blues')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Bar Plot
    sns.catplot(x=cat_var, y=cont_var, data=df, kind='bar', palette='Purples')
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    # Scatter plot with regression line
    sns.lmplot(x=cat_var, y=cont_var, data=df)
    plt.xlabel(cat_var)
    plt.ylabel(cont_var)
    
    plt.show()



# pairplot

def plot_variable_pairs(train, columns, hue=None):
    '''
    The function takes in a df and a list of columns from the df
    and displays a pair plot wid a regression line.
    '''
    
    kws = {'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}}
    sns.pairplot(train[columns],  kind="reg", plot_kws={'line_kws':{'color':'red'}, 'scatter_kws': {'alpha': 0.5}})
    plt.show()
    
    

# One function to get SSE, MSE, and RMSE for both baseline and model

def regression_metrics(residual, baseline_residual, df):
    '''
    Function takes in the residuals from a regression model, the baseline regression, and the dataframe they are coming from,
    and produces an SSE, MSE, and RMSE for the model and baseline, print the results for easy comparison.
    '''
    
    # Get R^2 first
    #---------------------------
    # Model
    df['residual^2'] = df.residual**2
    # Baseline
    df['baseline_residual^2'] = df.baseline_residual**2
    
    
    # Square of Sum Errors (SSE)
    #----------------------------
    # Model
    SSE = df['residual^2'].sum()
    # Baseline
    Baseline_SSE = df['baseline_residual^2'].sum()

    
    # Mean Square Errors (MSE)
    #----------------------------
    # Model
    MSE = SSE/len(df)
    # Baseline
    Baseline_MSE = Baseline_SSE/len(df)
    
    
    # Root Mean Squared Error (RMSE)
    #-----------------------------
    # Model
    RMSE = sqrt(MSE)
    # Baseline
    Baseline_RMSE = sqrt(Baseline_MSE)

    print(f'SSE')
    print(f'-----------------------')
    print(f'Model SSE --> {SSE:.1f}')
    print(f'Baseline SSE --> {Baseline_SSE:.1f}')
    print(f'MSE')
    print(f'-----------------------')
    print(f'Model MSE --> {MSE:.1f}')
    print(f'Baseline MSE --> {Baseline_MSE:.1f}')
    print(f'RMSE')
    print(f'-----------------------')
    print(f'Model RMSE --> {RMSE:.1f}')
    print(f'Baseline RMSE --> {Baseline_RMSE:.1f}')

    
    
def plot_coef_heatmap(x_train, y_train):
    '''
    Takes in a dataframe and gives a heatmap of the feature coefficients
    '''
    
    logit = LogisticRegression(random_state=1313)

    logit.fit(x_train, y_train)

    features = x_train.columns.tolist()
    
    weights = logit.coef_.flatten()
    
    visualize = pd.DataFrame(weights, features).reset_index().rename(columns={'index':'feature', 
                                                                          0: 'weights'})

    # Apply coefficients to a heatmap
    plt.figure()
    sns.heatmap(visualize)
    
    plt.show()
    
    
    
def rfe_feature_rankings(x_scaled, x, y, k):
    '''
    Takes in the predictors, the target, and the number of features to select,
    and it should return a database of the features ranked by importance
    '''
    
    # Make it
    lm = sklearn.linear_model.LinearRegression()
    rfe = sklearn.feature_selection.RFE(lm, n_features_to_select=k)

    # Fit it
    rfe.fit(x_scaled, y)
    
    var_ranks = rfe.ranking_
    var_names = x.columns.tolist()
    ranks = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    ranks = ranks.sort_values(by="Rank", ascending=True)
    return ranks
    
# *Work in progress*


