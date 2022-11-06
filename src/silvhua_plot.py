import matplotlib as plt

import plotly.graph_objects as go
import plotly 
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns

# Function to plot multiple histograms using Plotly. Show different colours based on classification.
def plot_int_hist(df, columns=None, color=None, label=1):
    """
    Use Plotly to plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    - df: Dataframe.
    - columns (optional): Columns of dataframe on which to create the histogram. If blank, all numeric data will be plotted.
    - color (optional): Provide name of colum containing binary classification values 0 and 1. 
        Data points classified as 1 will be in red.
    - label (optional): Label of classification column. Default is 1.
    
    Make sure to do the following imports:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    """
    # import plotly.express as px
    import plotly.graph_objects as go
    import plotly 
    from plotly.subplots import make_subplots

    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    fig = make_subplots(rows=round((len(columns)+.5)/2), cols=2,subplot_titles=columns)
    for i, feature in enumerate(columns):
        if color:
            bins = dict(
                start = min(df[feature]),
                end =  max(df[feature]),
                # size=
            )
            zero = df[df[color] != label]
            one = df[df[color] == label]
            fig.add_trace(go.Histogram(x=zero[feature],
                marker_color='blue',
                opacity=0.5,
                xbins=bins), 
                row=i//2+1, col=i % 2 + 1
                )
            fig.add_trace(go.Histogram(x=one[feature],
                marker_color='red',
                opacity=0.5,
                xbins=bins),
                row=i//2+1, col=i % 2 + 1)
        else:
            fig.add_trace(go.Histogram(x=df[feature]), 
            row=i//2+1, col=i % 2 + 1)
    if color:
        title = f'Observations with {color} of value {label} are indicated in red'
    else:
        title = 'Value counts'
    fig.update_layout(height=300*round((len(columns)+.5)/2), 
        showlegend=False,barmode=barmode,
        bargap=0.1,
        title = title,
        title_x=0.5,
        title_xanchor='center',
        title_y = 0,
        title_yanchor = 'bottom')
    fig.show()

# Function to plot multiple bar charts using Plotly. Show different colours based on classification.
def plot_int_bar(df, columns=None, color=None, label=1, barmode='stack'):
    """
    Use Plotly to plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    - df: Dataframe.
    - columns (optional): Columns of dataframe on which to create the histogram. If blank, all numeric data will be plotted.
    - color (optional): Provide name of colum containing binary classification values 0 and 1. 
        Data points classified as 1 will be in red.
    - label (optional): Label of classification column. Default is 1.
    - barmode ('stack', 'group', or 'overlay'; optional): How the different will be shown. Default is 'stack'.

    """
    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    fig = make_subplots(rows=round((len(columns)+.5)/2), cols=2,subplot_titles=columns)
    for i, feature in enumerate(columns):
        if color:
            zero = df[df[color] != label]
            one = df[df[color] == label]
            fig.add_trace(go.Histogram(x=zero[feature],
                marker_color='blue',
                opacity=0.5), 
                row=i//2+1, col=i % 2 + 1
                )
            fig.add_trace(go.Histogram(x=one[feature],
                marker_color='red',
                opacity=0.5),
                row=i//2+1, col=i % 2 + 1)
        else:
            fig.add_trace(go.Histogram(x=df[feature]), 
            row=i//2+1, col=i % 2 + 1)
    
    if color:
        title = f'Observations with {color} of value {label} are indicated in red'
    else:
        title = 'Value counts'
    fig.update_layout(height=300*round((len(columns)+.5)/2), 
        showlegend=False,barmode=barmode,
        bargap=0.1,
        title = title,
        title_x=0.5,
        title_xanchor='center',
        title_y = 0,
        title_yanchor = 'bottom')
    fig.show()

# Function to plot multiple histograms
def plot_hist(df, columns=None):
    """
    Plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    df: Dataframe.
    columns (optional): Columns of dataframe on which to create the histogram. If blank, all numeric data will be plotted.
    
    Make sure to `import seaborn as sns`.
    """
    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    nrows = round((len(columns)+.5)/2)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(10,nrows*2))
    for i, feature in enumerate(columns):
        sns.histplot(data=df,x=feature,ax=ax[i//2, i % 2])
    plt.tight_layout()

def correlation_plot(df):
    """
    Plot the correlation matrix.
    Returns the dataframe with the correlation values.
    """

    # Create a mask to exclude the redundant cells that make up half of the graph.
    mask = np.triu(np.ones_like(df.corr(), dtype=bool))

    # Create the heatmap with the mask and with annotation
    sns.heatmap(data=abs(df.corr(numeric_only=True)),mask=mask,annot=True)
    return df.corr()

# Function to plot multiple box plots
def plot_box(df, columns=None,category=None, hue=None):
    """
    Plot multiple histograms using the specified columns of a dataframe.
    Arguments:
    - df: Dataframe.
    - columns (optional): Columns of dataframe on which to create the histogram. If blank, 
        all numeric data will be plotted.
    - category (optional): Categorical feature for y-axis of box plot. If None, 
        only one box will be plotted per dependent variable.
    - hue (optional): Categorical feature by which to split the box plots.

    """
    if columns == None:
        columns = df.dtypes[df.dtypes != 'object'].index.tolist()
    nrows = round((len(columns)+.5)/2)
    fig, ax = plt.subplots(nrows=nrows, ncols=2, figsize=(10,nrows*2))
    for i, feature in enumerate(columns):
        if (category != None) & (hue !=None):
            sns.boxplot(data=df,x=feature,y=category,ax=ax[i//2, i % 2], hue=hue)
        elif category:
            sns.boxplot(data=df,x=feature,y=category,ax=ax[i//2, i % 2])
        else:
            sns.boxplot(data=df,x=feature,ax=ax[i//2, i % 2])
    plt.tight_layout()

# Function to make an interactive histogram combined with a box plot (or violin or rug plot).
def hist_box(df, column=None, marginal='box', color=None):
    """
    Make an interactive histogram combined with a box plot (or violin or rug plot).
    Parameters:
    - column: Column name to plot.
    - color (optional): Column for grouping data. If argument is provided, barmode is 'overlay'.
    - marginal ('box', 'violin', or 'rug'): The second type of plot to include.
    """
    fig = px.histogram(df, x=column, color=color,
        marginal=marginal, # or violin, rug
        opacity=0.5,
        hover_data=df.columns,
        barmode='overlay')
    fig.show()

# Function to plot bar charts where data are normalized for each group. Show different colours based on classification.
def plot_proportion(df, columns=None, classication='Loan_Status', label='Y', barmode='stack'):
    """
    Use Plotly to plot bar charts where data are normalized for each group. 
    Show different colours based on classification.
    Arguments:
    - df: Dataframe.
    - columns (optional): Columns of dataframe containing the categorical variables to plot. 
        If blank, all categorical data will be plotted.
    - classication: Provide name of colum containing binary classification values. 
        Data points classified as 1 will be in red.
    - label (optional): Label of classification column. Default is 1.
    - barmode ('stack', 'group', or 'overlay'; optional): How the different will be shown. Default is 'stack'.

    """
    if columns == None:
        columns = df.dtypes[df.dtypes == 'object'].index.tolist()
    fig = make_subplots(rows=round((len(columns)+.5)/2), cols=2,subplot_titles=columns)
    for i, feature in enumerate(columns):
        pivot = df.pivot_table(df.columns[-1], index=[classication], columns=[feature],aggfunc='count')
        zero_label = list(set(pivot.index)-set(label))[0]
        zero = pivot.loc[zero_label,:]/pivot.sum()*100
        yes = pivot.loc[label,:]/pivot.sum()*100
        # print(pivot)
        fig.add_trace(go.Bar(x=pivot.columns,
            y=zero,
            marker_color='blue',
            opacity=0.5), 
            row=i//2+1, col=i % 2 + 1
            )
        fig.add_trace(go.Bar(x=pivot.columns,
            y=yes,
            marker_color='red',
            opacity=0.5),
            row=i//2+1, col=i % 2 + 1)
    
    title = f'Percentage with {classication} of value {label} by category (indicated in red).'

    fig.update_layout(height=300*round((len(columns)+.5)/2), 
        showlegend=False,barmode=barmode,
        bargap=0.1,
        title = title,
        title_x=0.5,
        title_xanchor='center',
        # title_y = .96,
        title_yanchor = 'bottom')
    fig.show()