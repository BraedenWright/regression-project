import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




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


    
    
    
# *Work in progress*


