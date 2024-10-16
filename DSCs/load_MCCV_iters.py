import os
import pickle


def load_mccv_data(data_set_name, num_folds):
    # Define the path where the pickled files are stored
    load_path = f'C:/Users/Jesse/PycharmProjects/DataSetClasses/DSCs/data/MCCV/{data_set_name}/'

    # Initialize an empty list to store the loaded objects
    data_set_class_list = []

    # Loop through each fold and load the corresponding pickled file
    for fold in range(num_folds):
        file_path = os.path.join(load_path, f'data_set_class_{fold}.pkl')

        # Check if the file exists
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data_set_class = pickle.load(f)
                data_set_class_list.append(data_set_class)
            print(f"Loaded data_set_class for fold {fold} from {file_path}")
        else:
            print(f"Warning: File not found for fold {fold} at {file_path}")

    return data_set_class_list


if __name__ == "__main__":
    # Example usage
    data_set_name = 'Bimanual 3D'  # or 'Movements CMU'
    num_folds = 5

    loaded_data = load_mccv_data(data_set_name, num_folds)
    print(f"\nLoaded {len(loaded_data)} data_set_class objects.")

    # You can now use the loaded_data list, which contains all the data_set_class objects
    # For example, to access the first loaded object:
    for LD in loaded_data:
        print(LD.indices)