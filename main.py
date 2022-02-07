import numpy as np
import tensorflow as tf

import data_handler


def main():
    print("=================================")
    print("======== STARTING PROGRAM =======")
    print("=================================")

    # Declare Variables
    filepath = 'facebook_large/facebook.npz'
    train = 80
    test = 10
    validate = 10

    # Parse Data
    data = np.load(filepath)
    features, edges, target = data_handler.parse_npz_files(data)

    print(type(features))
    print(type(edges))
    print(type(target))



    # Split Train/Test/Validate

    # Train Model

    # Test Model




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

