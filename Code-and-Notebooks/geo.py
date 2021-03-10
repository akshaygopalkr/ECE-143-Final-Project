import geopandas as gpd
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def create_world(file_train, file_test):
    '''
    Create a geopandas dataframe to count the
    number of times each location is mentioned
    in news articles.
    
    :param file_train: Source file of training data
    :type file_train: string
    :param file_test: Source file of test data
    :type file_test: string

    '''
    #Read in starter dataframes
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Census data from https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
    #usa = gpd.read_file('state_data/cb_2018_us_state_20m.shp')

    #Replace 'United States of America' with more general term 'United States'
    world.at[4, 'name'] = 'United States'
    #usa.at[36,'NAME'] = 'D.C.'

    #Define list of countries and states
    countries = list(world.name)
    #states = list(usa.NAME)

    # Load in data
    training_data = pd.read_csv(file_train)
    #test_data = pd.read_csv(file_test)

    #Sort real and fake news into two dataframes
    real_news = training_data[training_data.label == 0]
    fake_news = training_data[training_data.label == 1]

    # Get the counts of the countries and states mentioned in all articles
    dict_real_countries = count_location_instances(countries, real_news)
    dict_fake_countries = count_location_instances(countries, fake_news)
    #dict_real_states = count_location_instances(states, real_news)
    #dict_fake_states = count_location_instances(states, fake_news)

    # Create new column in dataframe to keep track of number of mentions in fake and real articles
    world['Mentions_Fake'] = world['name']
    world['Mentions_Real'] = world['name']

    # Fill the column with number of mentions
    world = world.replace({'Mentions_Fake':dict_fake_countries})
    world = world.replace({'Mentions_Real':dict_real_countries})

    # Set unused country mentions to 0
    world['Mentions_Fake'] = pd.to_numeric(world['Mentions_Fake'], errors='coerce')
    world['Mentions_Real'] = pd.to_numeric(world['Mentions_Real'], errors='coerce')
    world.replace('NaN',0)

    return world


def count_location_instances(l, df):
    '''
    Count instances of a location in body of text
    
    :param l: location to look for
    :type l: string
    :param df: dataframe of news articles
    :type df: DataFrame

    '''
    location_count = Counter()
    
    for index, row in df.iterrows():
        text = row['text']
        locations = get_locations(text,l)
        for loc in locations:
            location_count[loc] += 1
    return location_count


def get_locations(text, locations):
    '''
    Extract the mentioned locations from a body of text
    
    :param text: input text to search
    :type text: string
    :param locations: list of countries
    :type locations: list of strings

    '''
    if isinstance(text,float):
        return []
    mentionedLocations = []
    words = text.lower()

    # Look for each source in the block of text and add the name of the source to a list if it's found
    for loc in locations:
        if loc.lower() in words:
            mentionedLocations.append(loc)
    
    return mentionedLocations


if __name__ == "__main__":
    world_df = geo('./Datasets/train.csv', './Datasets/test.csv')
    plt.figure()
    plt.show()