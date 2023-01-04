import pandas as pd
import numpy as np
import synthia as syn
import warnings, pickle, os
warnings.filterwarnings('ignore')


def get_synthetic_data():

    # Load the original data
    data  = pd.read_csv('../../data/data.csv', index_col=0)
    # Get file datatypes
    dtypes = data.dtypes
    # Get the names of the columns with numeric types
    numeric = data.columns[dtypes.apply(pd.api.types.is_numeric_dtype)]
    # Extract numeric subset
    subset = data.loc[:,numeric].replace(np.nan, 0)

    # Create Generator
    generator = syn.CopulaDataGenerator()

    # Define Coupla and Parameterizer
    # if coupled:
    #     parameterizer = syn.QuantileParameterizer(n_quantiles=100)
    #     generator.fit(subset, copula=syn.GaussianCopula(), parameterize_by=parameterizer)
    # else:
    generator.fit(subset, copula=syn.IndependenceCopula())


    print(f'Storage size: {len(pickle.dumps(generator))} bytes')

    # Generate our samples to the same shape as the original data
    samples = generator.generate(n_samples=len(subset), uniformization_ratio=0, stretch_factor=1)
    synthetic = pd.DataFrame(samples, columns = subset.columns, index = subset.index)

    # Create a new dataframe with the synthetic data
    update = data.loc[:]
    update.loc[:,numeric]= synthetic.loc[:,numeric]
    update = update.astype(dtypes)

    # save the new dataframe
    i = 0
    while os.path.exists(f'../../data/data_synthetic_{i}.csv'):
        i += 1
    update.to_csv(f'../../data/data_synthetic_{i}.csv')


if __name__ == '__main__':
    get_synthetic_data()
