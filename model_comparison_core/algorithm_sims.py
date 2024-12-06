from functools import partial
import numpy as np
from itertools import product
import random
from model_comparison_core.compile_results_internal import CRI
from joblib import Parallel, delayed
import copy
import psutil
import os

class func_simulator():

    def __init__(self, data_func, comp_func, space=None, parallelize=False):
        self.set_data_func(data_func)
        self.set_comp_func(comp_func)
        self.space = space
        self.parallelize = parallelize

    def init_worker(self, seed):
        from multiprocessing import current_process
        worker_id = current_process().name

        # Set the global NumPy seed
        np.random.seed(int(seed))
        # Set the Python random seed
        random.seed(int(seed))
        print(f"Worker {worker_id} initialized with seed {seed}")

    def worker_wrapper(self, func, seed, variable_args):
        # Initialize worker with the provided seed
        self.init_worker(seed)
        # Call the actual function
        return func(variable_args_dict=variable_args)


    def obj_func(self, space, args=(), kwargs={}):
        '''try:
            fold_list = kwargs['data_set_class_dict']['fold_list']
            space_dict = {}
            for i, param in enumerate(self.space):
                space_dict[param.name] = space[i]

            # Create a base filename using space_dict parameters
            base_filename = ''
            for key in space_dict.keys():
                key_name = ''.join([k[0] for k in key.split(':')[-1].split('_')])
                dict_val = str(space_dict[key])
                base_filename += key_name + '_' + dict_val + '-'
            base_filename = base_filename[:-1]  # Remove the trailing underscore
            base_save_path = os.path.join(kwargs['dir_path'], base_filename).replace('\\', '/')
            os.makedirs(base_save_path, exist_ok=True)

            def process_fold(fold_num, base_save_path):
                local_kwargs = copy.deepcopy(kwargs)
                local_kwargs['data_set_class_dict']['fold_num'] = fold_num
                data_set_class = self.data_func(*args, fold_num=fold_num, space_dict=space_dict, **local_kwargs)
                _ = self.comp_func(
                    data_set_class, *args, space_dict=space_dict,
                    save_path=base_save_path, fold_num=fold_num, **local_kwargs
                )

            # Determine the number of cores to use
            num_cores = max(1, psutil.cpu_count() // 2)

            # Function to split fold_list into batches
            def chunks(lst, n):
                """Yield successive n-sized chunks from lst."""
                for i in range(0, len(lst), n):
                    yield lst[i:i + n]

            # Process folds in batches
            for batch in chunks(fold_list, num_cores):
                Parallel(n_jobs=num_cores)(
                    delayed(process_fold)(fold_num, base_save_path) for fold_num in batch
                )

            # Pass the base_save_path to CRI
            score = CRI(*args, space_dict=space_dict, csv_path=base_save_path, **kwargs)
            return score
        except Exception as e:
            print(f"Error with parameters {space}: {e}")
            return 1000000'''
        fold_list = kwargs['data_set_class_dict']['fold_list']
        space_dict = {}
        for i, param in enumerate(self.space):
            space_dict[param.name] = space[i]

        # Create a base filename using space_dict parameters
        base_filename = ''
        for key in space_dict.keys():
            key_name = ''.join([k[0] for k in key.split(':')[-1].split('_')])
            dict_val = str(space_dict[key])
            base_filename += key_name + '_' + dict_val + '-'
        base_filename = base_filename[:-1]  # Remove the trailing underscore
        base_save_path = os.path.join(kwargs['dir_path'], base_filename).replace('\\', '/')
        os.makedirs(base_save_path, exist_ok=True)

        def process_fold(fold_num, base_save_path):
            local_kwargs = copy.deepcopy(kwargs)
            local_kwargs['data_set_class_dict']['fold_num'] = fold_num
            data_set_class = self.data_func(*args, fold_num=fold_num, space_dict=space_dict, **local_kwargs)
            _ = self.comp_func(
                data_set_class, *args, space_dict=space_dict,
                save_path=base_save_path, fold_num=fold_num, **local_kwargs
            )

        # Determine the number of cores to use
        num_cores = max(1, psutil.cpu_count() // 2)

        # Function to split fold_list into batches
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        # Process folds in batches
        for batch in chunks(fold_list, num_cores):
            Parallel(n_jobs=num_cores)(
                delayed(process_fold)(fold_num, base_save_path) for fold_num in batch
            )

        # Pass the base_save_path to CRI
        score = CRI(*args, space_dict=space_dict, csv_path=base_save_path, **kwargs)
        return score

    def data_func(self, seed, args_dict):
        raise NotImplementedError("Must set a data function.")

    def comp_func(self, seed, data_set_class, args_dict=None):
        raise NotImplementedError("Must set a computation function.")

    def sim(self, constant_args_dict, variable_args_dict, return_model = False):
        return self.sim_func(return_model = return_model,**constant_args_dict['data_args_dict'], **constant_args_dict['comp_args_dict'],
                             **variable_args_dict)

    def sim_func(self, *args, **kwargs):
        if self.space is None:
            data_set_class = self.data_func(fold_num = kwargs['data_set_class_dict']['fold_num'],**kwargs)
            return self.comp_func(data_set_class, *args, save_path = kwargs['dir_path'], fold_num = kwargs['data_set_class_dict']['fold_num'], **kwargs)

        else:
            from skopt import gp_minimize
            from skopt.space import Categorical, Integer
            total_combinations = 1
            for dim in self.space:
                if isinstance(dim, Categorical):
                    total_combinations *= len(dim.categories)
                elif isinstance(dim, Integer):
                    total_combinations *= (dim.high - dim.low + 1)
            if total_combinations < 50:
                n_calls = total_combinations
                n_inits = total_combinations#int(n_calls * .3)
                if n_inits == 0:
                    n_inits = 1
            elif total_combinations >= 50:
                n_calls = int(np.log10(total_combinations) * 25)
                n_inits = int(n_calls * .4)
            partial_objective = partial(self.obj_func, args=args, kwargs=kwargs)
            return gp_minimize(partial_objective, self.space, n_initial_points=n_inits, n_calls=n_calls, random_state=0)

    def set_data_func(self, func):
        self.data_func = func

    def set_comp_func(self, func):
        self.comp_func = func

    def set_obj_func(self, func):
        self.obj_func = func

    def parallelize_funcs(self, func, func_args, seeds, return_model = False):
        import multiprocessing
        from joblib import Parallel, delayed, dump
        param_combinations = list(
            product(seeds['seed'], *func_args['comp_args_dict'].values(), *func_args['data_args_dict'].values()))
        func_args_list = [dict(
            zip(['seed'] + list(func_args['comp_args_dict'].keys()) + list(func_args['data_args_dict'].keys()),
                combination)) for combination in param_combinations]

        n_jobs = max(multiprocessing.cpu_count() - 1, 1)  # Number of cores to use
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.worker_wrapper)(func, args['seed'], args) for args in func_args_list
        )

        return results



    def serialize_funcs(self, func, func_args, seeds,return_model = False):
        param_combinations = list(
            product(seeds['seed'], *func_args['comp_args_dict'].values(), *func_args['data_args_dict'].values()))
        func_args_list = [dict(zip(['seed'] + list(func_args['comp_args_dict'].keys()) + list(
            func_args['data_args_dict'].keys()), combination)) for combination in param_combinations]
        results = []
        for i, args in enumerate(func_args_list):
            self.init_worker(args['seed'])
            args['sim_index'] = i  # Add sim_index to args
            args['dict_index'] = func_args['dict_index']  # Add dict_index to args
            args['dir_path'] = func_args['dir_path']  # Add dir_path to args
            results.append(func(variable_args_dict=args,return_model = return_model))
        return results


    def run_sims(self, data_args_dict, comp_args_dict, seed=1000, dict_index=0, sim_index=0, dir_path='',return_model = False):
        constant_args_dict = {'data_args_dict': data_args_dict['constant_args_dict'],
                              'comp_args_dict': comp_args_dict['constant_args_dict']}
        variable_args_dict = {'data_args_dict': data_args_dict['variable_args_dict'],
                              'comp_args_dict': comp_args_dict['variable_args_dict']}

        seeds = {'seed': [seed]}
        partial_sim_func = partial(self.sim, constant_args_dict=constant_args_dict)
        variable_args_dict['dict_index'] = dict_index
        variable_args_dict['dir_path'] = dir_path
        if self.parallelize:
            return self.parallelize_funcs(partial_sim_func, variable_args_dict, seeds, return_model = return_model)
        else:
            return self.serialize_funcs(partial_sim_func, variable_args_dict, seeds, return_model = return_model)
