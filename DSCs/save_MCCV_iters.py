from DSCs.data_func import data_func
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import pickle  # Add this import

if __name__ == "__main__":
    # Main configuration
    data_set_name = 'Bimanual 3D'
    #data_set_name = 'Movements CMU'

    if data_set_name == "Bimanual 3D":
        actions = [0, 1, 2, 3, 4]
        num_sequences_per_action_train = 1
        num_sequences_per_action_test = 9
        people = [1]
    elif data_set_name == "Movements CMU":
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        num_sequences_per_action_train = 1
        num_sequences_per_action_test = 5
        people = [0]
    else:
        raise ValueError("Dataset does not exist.")

    seq_len = 100
    save_path = 'C:/Users/Jesse/PycharmProjects/DataSetClasses/DSCs/data/MCCV/' + data_set_name + '/'

    num_folds = 50  # 5 * (num_sequences_per_action_train + num_sequences_per_action_test)
    fold_num = 0
    data_set_class_list = []

    for fold in range(num_folds):
        data_set_class_dict = {
            'data_set_name': data_set_name,
            'actions': actions,
            'people': people,
            'seq_len': seq_len,
            'num_folds': num_folds,
            'fold_num': fold,
            'num_sequences_per_action_train': num_sequences_per_action_train,
            'num_sequences_per_action_test': num_sequences_per_action_test,
            'X_init': None
        }
        file_path = save_path + 'data_set_class_' + str(fold) + '.pkl'  # Added .pkl extension
        data_set_class = data_func(data_set_class_dict)

        # Pickle and save the data_set_class object
        with open(file_path, 'wb') as f:
            pickle.dump(data_set_class, f)

        print(f"Saved data_set_class for fold {fold} to {file_path}")