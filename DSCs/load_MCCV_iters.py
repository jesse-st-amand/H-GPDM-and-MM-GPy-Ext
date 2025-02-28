import os
import pickle
import numpy as np


def load_mccv_data(data_set_name, num_folds):
    # Define the path where the pickled files are stored
    load_path = f'C:/Users/Jesse/Documents/Python/HGP_concise/DSCs/data/MCCV/{data_set_name}/'

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

def IFs_setup(stick_dict_list, data_set_class):
    IFs = []
    for i, Y in enumerate(stick_dict_list):
        IFs.append(data_set_class.IFs_func([Y]))
    return IFs

if __name__ == "__main__":
    # Example usage
    #data_set_name = 'Bimanual 3D'
    data_set_name = 'Movements CMU'
    num_folds = 1
    play_one_DSC = True

    DSC_list = load_mccv_data(data_set_name, num_folds)
    
    for LD in DSC_list:
        print(LD.indices)
    print(f"\nLoaded {len(DSC_list)} data_set_class objects.")

    if play_one_DSC:
        DSC = DSC_list[0]
        true_sequences = np.concatenate([DSC.Y_train_CCs,DSC.Y_test_CCs,DSC.Y_validation_CCs])
        Y_arr_list = DSC.CC_dict_list_to_CC_array_list_min_PV(true_sequences)
        Y_stick_dict_list = DSC.CC_2D_list_to_stick_dict_list(Y_arr_list)
        IFs = IFs_setup(Y_stick_dict_list, DSC)
        

        # Visualization code
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.animation as animation

        anis = []
        for i, IFs_i in enumerate(IFs):
            fig, animate = IFs_i.plot_animation_all_figures()
            anis.append(animation.FuncAnimation(fig, animate, 100, interval=100))
        matplotlib.pyplot.show()
    

    