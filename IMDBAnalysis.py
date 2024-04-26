import csv
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge

def read_csv_to_list(file_path):
    """
    Reads a CSV file and returns the data as a list of rows.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        list: A list of rows, where each row is a list of values.
    """
    rows = []
    
    with open(file_path, 'r') as file:
        csvreader = csv.reader(file)
        
        for row in csvreader:
            rows.append(row)
    
    return rows

def read_csv_to_df(file_path):
    """
    Reads a CSV file and returns the data as a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pandas.DataFrame: The data from the CSV file as a DataFrame.
    """
    return pd.read_csv(file_path)

def get_top_actors(df, n):
    """
    Finds the top n actors based on the frequency of their appearances in the dataset.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
        n (int): The number of top actors to return.
        
    Returns:
        pandas.Series: A Series containing the top n actors and their frequencies.
    """
    # Isolate the cast column and split it into individual actors
    all_actors = [actor.strip() for cast in df['Cast'] for actor in cast.split(', ')]
    
    # Count the frequency of each actor
    actor_counts = Counter(all_actors)
    
    # Sort the actors by their frequency and get the top n
    top_actors = pd.Series(actor_counts).sort_values(ascending=False)[:n]
    
    return top_actors


def plot_movies_by_decade(df, title='Number of Movies by Decade', x_label='Decade', y_label='Number of Movies'):
    """
    Plots the number of movies by decade.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
        title (str, optional): The title of the plot. Default is 'Number of Movies by Decade'.
        x_label (str, optional): The label for the x-axis. Default is 'Decade'.
        y_label (str, optional): The label for the y-axis. Default is 'Number of Movies'.
    """
    # Convert 'Release Year' column to integer
    df['Release Year'] = pd.to_numeric(df['Release Year'], errors='coerce')
    
    # Drop rows with NaN values in 'Release Year' column
    df = df.dropna(subset=['Release Year'])
    
    # Get the decade from the 'Release Year' column
    df = df.copy() 
    df['Decade'] = (df['Release Year'] // 10) * 10
    
    # Group movies by decade and count the number of movies in each decade
    decade_counts = df.groupby('Decade').size().sort_values(ascending=True)
    
    # Plot the histogram
    plt.figure(figsize=(10, 6))
    decade_counts.plot(kind='bar', color='skyblue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add number value to each bar column
    for i, value in enumerate(decade_counts):
        plt.text(i, value, str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_average_gross_by_year(df, title='Average Gross Value of Movies by Year', x_label='Year', y_label='Average Gross Value', figure_size=(10, 6), xticks_rotation=45):
    """
    Plots the average gross value of movies by release year.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
        title (str, optional): The title of the plot. Default is 'Average Gross Value of Movies by Year'.
        x_label (str, optional): The label for the x-axis. Default is 'Year'.
        y_label (str, optional): The label for the y-axis. Default is 'Average Gross Value'.
        figure_size (tuple, optional): The size of the plot figure in inches. Default is (10, 6).
        xticks_rotation (int, optional): The rotation angle for the x-axis tick labels. Default is 45.
    """
    # Create a copy of the DataFrame (did this to avoid getting errors)
    df_copy = df.copy()
    
    # Convert 'Release Year' column to integer
    df_copy['Release Year'] = pd.to_numeric(df_copy['Release Year'], errors='coerce')
    
    # Drop rows with NaN values in 'Release Year' column
    df_copy = df_copy.dropna(subset=['Release Year'])
    
    # Remove all non-numeric characters from 'Gross' column
    df_copy['Gross'] = df_copy['Gross'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
    
    # Convert 'Gross' column to numeric
    df_copy['Gross'] = pd.to_numeric(df_copy['Gross'], errors='coerce')
    
    # Group movies by year and calculate the average gross value for each year
    average_gross_by_year = df_copy.groupby('Release Year')['Gross'].mean()
    
    # Plot the data in a line graph
    plt.figure(figsize=figure_size)
    average_gross_by_year.plot(kind='line', marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.xticks(rotation=xticks_rotation)
    plt.tight_layout()
    plt.show()

def get_top_1939_movies_by_gross(df, n=5):
    """
    Returns the top n movies made in 1939 by gross value.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
        n (int, optional): The number of top movies to return. Default is 5.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the top n movies made in 1939 by gross value.
    """
    # Filter the DataFrame to only include movies from 1939
    movies_1939 = df[df['Release Year'] == 1939].copy()
    
    # Remove all non-numeric characters from the 'Gross' column
    movies_1939.loc[:, 'Gross'] = movies_1939['Gross'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))
    
    # Convert the 'Gross' column to numeric
    movies_1939.loc[:, 'Gross'] = pd.to_numeric(movies_1939['Gross'], errors='coerce')
    
    # Sort the movies by gross value in descending order and get the top n
    top_1939_movies = movies_1939.sort_values('Gross', ascending=False).head(n)
    
    # Return only 'Movie Name' and 'Gross' columns
    return top_1939_movies[['Movie Name', 'Gross']]

def get_top_highest_grossing_movies(df, n):
    """
    Extracts and displays the top n highest-grossing movies with their names and gross values.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
        n (int): The number of top highest-grossing movies to retrieve.
    
    Returns:
        None
    """
    # Remove all non-numeric characters from 'Gross' column
    df['Gross'] = df['Gross'].apply(lambda x: re.sub(r'[^0-9.]', '', str(x)))

    # Convert 'Gross' column to numeric
    df['Gross'] = pd.to_numeric(df['Gross'], errors='coerce')
    
    # Sort the DataFrame by 'Gross' column in descending order
    top_n_highest_grossing_movies = df.sort_values(by='Gross', ascending=False).head(n)
    
    # Extract 'Movie Name' and 'Gross' columns
    top_n_movie_names_and_gross = top_n_highest_grossing_movies[['Movie Name', 'Gross']]
    
    # Display the top n highest-grossing movies with their names and gross values
    print(top_n_movie_names_and_gross)
    

def plot_average_metascore_by_year(df):
    """
    Groups movies by year and calculates the average Metascore for each year, then plots the data in a line graph.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
    
    Returns:
        None
    """
    # Group movies by year and calculate the average Metascore for each year
    average_metascore_by_year = df.groupby('Release Year')['Metascore'].mean()

    # Plot the data in a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(average_metascore_by_year.index, average_metascore_by_year.values, marker='o', linestyle='-')
    plt.title('Average Metascore of Movies by Year')
    plt.xlabel('Year')
    plt.ylabel('Average Metascore')
    plt.grid(True)
    plt.show()

def plot_average_imdb_rating_by_year(df):
    """
    Groups movies by year and calculates the average IMDb rating for each year, then plots the data in a line graph.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
    
    Returns:
        None
    """
    # Group movies by year and calculate the average IMDb rating for each year
    average_imdb_rating_by_year = df.groupby('Release Year')['IMDB Rating'].mean()

    # Plot the data in a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(average_imdb_rating_by_year.index, average_imdb_rating_by_year.values, marker='o', linestyle='-')
    plt.title('Average IMDb Rating of Movies by Year')
    plt.xlabel('Year')
    plt.ylabel('Average IMDb Rating')
    plt.grid(True)
    plt.show()
    
def plot_total_votes_by_year(df):
    """
    Removes all non-numeric characters from the 'Votes' column, converts it to numeric type, 
    groups movies by year, calculates the total number of votes per movie for each year, 
    and plots the data in a line graph.
    
    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.
    
    Returns:
        None
    """
    # Remove all non-numeric characters from 'Votes' column and convert it to numeric
    df['Votes'] = df['Votes'].replace('[^\d.]', '', regex=True)
    df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce')
    
    # Group movies by year and calculate the total number of votes per movie for each year
    total_votes_by_year = df.groupby('Release Year')['Votes'].sum()

    # Plot the data in a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(total_votes_by_year.index, total_votes_by_year.values, marker='o', linestyle='-')
    plt.title('Total Number of Votes given by Year')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Votes')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_metascore_vs_votes(df):
    """
    Creates a scatter plot of Metascore against the number of votes.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.

    Returns:
        None
    """
    # Extract 'Metascore' and 'Votes' columns
    metascore = df['Metascore']
    votes = df['Votes']

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(votes, metascore, alpha=0.5)
    plt.title('Scatter Plot of Metascore vs. Number of Votes')
    plt.xlabel('Number of Votes')
    plt.ylabel('Metascore')
    plt.grid(True)
    plt.show()
    
def plot_imdb_rating_vs_votes(df):
    """
    Creates a scatter plot of imdb rating against the number of votes.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the movie data.

    Returns:
        None
    """
    # Extract 'Metascore' and 'Votes' columns
    imdb = df['IMDB Rating']
    votes = df['Votes']

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(votes, imdb, alpha=0.5)
    plt.title('Scatter Plot of IMDB Rating vs. Number of Votes')
    plt.xlabel('Number of Votes')
    plt.ylabel('IMDB Rating')
    plt.grid(True)
    plt.show()
    
def remove_commas(x):
    if isinstance(x, str):
        return x.replace(',', '')
    else:
        return str(x)

def one_hot_encode_genre_column(df, column_name='Genre', separator=', '):
    """
    Performs one-hot encoding on a specified column containing genres.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the genre data.
        column_name (str, optional): The name of the column to be one-hot encoded. Default is 'Genre'.
        separator (str, optional): The separator used to split multiple genres. Default is ', '.

    Returns:
        pandas.DataFrame: A DataFrame with the specified column one-hot encoded and the original column dropped.
    """
    # Initialize MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    
    # Split genres and perform one-hot encoding
    genres_encoded = mlb.fit_transform(df[column_name].str.split(separator))
    
    # Create DataFrame with encoded genres
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
    
    # Concatenate encoded genres DataFrame with original DataFrame
    df_encoded = pd.concat([df, genres_df], axis=1)
    
    # Drop original column
    df_encoded.drop(columns=[column_name], inplace=True)
    
    return df_encoded

def test_train_split(df, p):
    """
    Splits the data table into a training set and a testing set.

    Args:
        df (pandas.DataFrame): DataFrame which will contain training and testing data
        p (int): Integer value between 0 and 1 which wil determine the test set proportion of the df.

    Returns:
        pandas.DataFrame: Four separate DataFrames containing training features (X_train), testing features(X_test), training results(y_train), and testing results(y_test).
    """
    # Listing the features
    X = df[df.columns.to_list()]

    # Setting the dependent variable
    y = df['IMDB Rating']

    # Splitting the dataframe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=p, random_state=210)

    return X_train, X_test, y_train, y_test

def linear_pred(X_train, X_test, y_train):
    """
    Performs a simple linear regression model using training and testing data.

    Args:
        X_train (pandas.DataFrame): A DataFrame containing the training features from the test_train_split.
        X_test (pandas.DataFrame): A DataFrame containing the testing features from the test_train_split.
        y_train (pandas.DataFrame): A DataFrame containing the actual rating values corresponding to the X_train features.

    Returns:
        pandas.DataFrame: A DataFrame with the coefficients (weights) of the different features in the linear model.
    """
    # Creating a regressor
    regressor = LinearRegression()

    # Fitting the training data into the regressor
    regressor.fit(X_train, y_train)    

    # Finding the coefficients from the regressor
    coefficients = pd.DataFrame({'Feature': X_test.columns, 'Coefficient': regressor.coef_})

    return coefficients

def lasso_reg(X_train, X_test, y_train):
    """
    Performs a lasso regression using the training and testing data.

    Args:
        X_train (pandas.DataFrame): A DataFrame containing the training features from the test_train_split.
        X_test (pandas.DataFrame): A DataFrame containing the testing features from the test_train_split.
        y_train (pandas.DataFrame): A DataFrame containing the actual rating values corresponding to the X_train features.

    Returns:
        numpy.Array: A list of features with non-zero lasso coefficients, and a list of the lasso coefficients.
    """
    lasso = LassoCV(cv=10)
    lasso.fit(X_train, y_train)

    selected_features = X_test.columns[lasso.coef_ != 0]
    coefficients = lasso.coef_[lasso.coef_ != 0]
    return selected_features, coefficients

def lasso_plot(selected_features, coefficients):
    """
    Plots the lasso regression coefficients.

    Args:
        selected_features (numpy.Array): A list of lasso selected features from lasso_reg.
        coefficients (numpy.Array): A list of lasso coeefficients from lasso_reg

    Returns:
        None.
    """
    plt.figure(figsize=(10, 6))

    plt.barh(selected_features, coefficients)

    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Lasso Regression Coefficients')
    plt.show()

def ridge_reg(num_features, cat_features, X_train, y_train, X_test):
    """
    Performs a Bayesian Ridge Regression using the training and testing features.

    Args:
        num_features (numpy.Array): A list of the numerical features
        cat_features (numpy.Array): A list of the categorical features
        X_train (pandas.DataFrame): A DataFrame containing the training features from the test_train_split.
        X_test (pandas.DataFrame): A DataFrame containing the testing features from the test_train_split.
        y_train (pandas.DataFrame): A DataFrame containing the actual rating values corresponding to the X_train features.

    Returns:
        numpy.Array: A list of rating predictions for each row of X_test 
    """
    # Standardizing the numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', 'passthrough', cat_features)
        ])
    # Creating Bayesian Ridge model using standardized numerical features and one hot encoded categorical features.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', BayesianRidge())
    ])
    # Dropping unused columns and resetting index.
    X_train.drop(['Movie Name', 'Release Year', 'IMDB Rating', 'Metascore', 'Director', 'Cast'], axis = 1)
    X_train.reset_index(drop=True, inplace=True)

    # Fitting the Bayesian Ridge model
    model.fit(X_train, y_train)

    # Getting predicted ratings using the model.
    y_pred = model.predict(X_test)
    X_test.reset_index(drop=True, inplace=True)

    return y_pred

def get_mse(y_test, y_pred):
    """
    Gets the mean squared error by squaring the difference between predicted y and actual y

    Args:
        y_test (numpy.Array): A list of actual rating values in the test set of the initial dataframe.
        y_pred (numpy.Array): A list of predicted rating values from the ridge regression.

    Returns:
        int: The mean squared error.
    """
    mse = mean_squared_error(y_test, y_pred)

    return mse

def get_res_df(y_pred, X_test):

    """
    Enters the predicted and actual ratings into a DataFrame.

    Args:
        y_pred (numpy.Array): A list of predicted values from ridge regression.
        X_test (pandas.DataFrame): A DataFrame containing the test set features from the test_train_split

    Returns:
        pandas.DataFrame: A DataFrame with the predicted ratings corresponding to the testing ratings.
    """
    res = pd.DataFrame({'Predicted': y_pred, 'Actual': X_test['IMDB Rating']})

    return res

def plot_pred_act(res):
    """
    Plots the predicted ratings and actual ratings.

    Args:
        res (pandas.DataFrame): A DataFrame containing the predicted ratings and actual ratings.

    Returns:
        None.
    """
    xax = res.index.to_numpy()
    y1ax = res['Predicted'].to_numpy()
    y2ax = res['Actual'].to_numpy()
    plt.plot(xax, y1ax, label ='Predicted')
    plt.plot(xax, y2ax, '-', label ='Actual')

    plt.xlabel("Index")
    plt.ylabel("Rating")
    plt.legend()
    plt.title('Prediction vs Actual')
    plt.show()