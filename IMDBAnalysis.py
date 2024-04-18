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

