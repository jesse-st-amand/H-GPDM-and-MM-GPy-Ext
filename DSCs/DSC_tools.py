from itertools import permutations
import numpy as np
import random
import math
def is_smallest_rotation(perm):
    n = len(perm)
    for i in range(1, n):
        if perm[i:] + perm[:i] < perm:
            return False
    return True

def indep_orders(t_list):
    indep_orders_list = []
    t_len = len(t_list)
    if t_len % 2 == 0:
        v = list(np.zeros(len(t_list)))

        for i,t in enumerate(t_list):
            if i % 2 == 1:
                v[i-1] = t
            else:
                v[i+1] = t
        v2 = v.copy()
        for i in range(1,t_len-1):
            if i % 2 == 0:
                v2[i-1] = v[i]
            else:
                v2[i+1] = v[i]
    else:
        v2 = list(np.vstack([np.arange(0, t_len, 2)[:,np.newaxis], np.arange(1, t_len, 2)[:,np.newaxis]]).flatten())

    indep_orders_list.append((t_list))
    indep_orders_list.append((t_list[::-1]))
    indep_orders_list.append((v2))
    indep_orders_list.append((v2[::-1]))

    return indep_orders_list

def get_shuffled_vectors(V, n):
    result = []
    for _ in range(n):
        # Create a copy of V and shuffle it
        shuffled_v = V.copy()
        random.shuffle(shuffled_v)
        result.append(shuffled_v)
    return result


def is_equivalent_to_any(new_list, list_of_lists):
    # Sort the new list for comparison

    for sublist in list_of_lists:
        # Sort each sublist for comparison
        if list(new_list) == list(sublist):
            return True

    return False
def find_unique_arrangements(n,max_len,unique_arrangements = []):
    V = list(range(n))
    while len(unique_arrangements) < max_len:
        V_list = get_shuffled_vectors(V, 100)
        for perm in V_list:
            if not is_equivalent_to_any(perm, unique_arrangements):
                if is_smallest_rotation(perm):
                    unique_arrangements.append((perm))

    return unique_arrangements


def dissimilarity(arr1, arr2):
    """
    Calculate dissimilarity between two arrangements based on their infinite sequence T.
    """
    n = len(arr1)
    total_dissimilarity = 0

    for i in range(n):
        for j in range(n):
            pos_diff = abs((arr1.index(i) - arr1.index(j)) % n - (arr2.index(i) - arr2.index(j)) % n)
            total_dissimilarity += min(pos_diff, n - pos_diff)

    return total_dissimilarity


def find_most_dissimilar(prev_arrangements, remaining, lookback):
    """
    Find the arrangement in 'remaining' that is most dissimilar to the previous 'lookback' arrangements.
    """

    def total_dissimilarity(arr):
        return sum(dissimilarity(arr, prev) for prev in prev_arrangements[-lookback:])

    return max(remaining, key=total_dissimilarity)


def order_arrangements(arrangements, lookback, length):
    """
    Order arrangements to maximize dissimilarity between each arrangement and the previous 'lookback' arrangements.
    """
    ordered = arrangements[:lookback]  # Start with the first arrangement
    remaining = set(map(tuple, arrangements[lookback:]))  # Convert to set of tuples for faster removal
    i = 0
    while remaining and i < length:
        next_arr = find_most_dissimilar(ordered, remaining, min(lookback, len(ordered)))
        ordered.append(next_arr)
        remaining.remove(tuple(next_arr))
        i+=1

    return ordered

def generate_uniform_list(a, t):
    t_list_rep_length = len(range(0, a, t)) * t
    return [
        [i] * t_list_rep_length
        for i in range(t)
    ]

def shift_series(lst):
    diag_list = []
    for i in range(len(lst)-2):
        lst[i], lst[i+1] = lst[i+1], lst[i]
        diag_list.append(lst.copy())
    return diag_list




def generate_complex_list(a, t, f):
    u_list = generate_uniform_list(a,t)
    u_len = len(u_list)
    # Create t_list
    t_list = list(range(t))
    if f - u_len < 1:
        return u_list[:f],t_list
    '''else:
        return u_list, t_list'''


    bases = find_unique_arrangements(t,math.ceil(f/t),unique_arrangements=indep_orders(t_list))
    bases = order_arrangements(bases, 4, math.ceil(f/t))
    # Create t_list_rep
    if a <= t:
        repetitions = 1
    else:
        repetitions = (a + t - 1) // t  # Ceiling division

    t_list_rep = t_list.copy() * repetitions
    tuple_set = set()

    # Initialize fold_list
    fold_list = u_list+[t_list_rep.copy()]
    tuple_set.update([tuple(fl) for fl in fold_list])


    # Generate fold_list
    i = 0
    while len(fold_list) < f:
        # Shift the previous list
        new_element = fold_list[-1][1:] + [fold_list[-1][0]]

        if (len(tuple_set)-u_len) % t == 0:
            i += 1
            if i < len(bases):
                new_basis = list(bases[i] * repetitions)

            else:
                new_basis = random.choices(t_list, k=t)
            if tuple(new_basis) not in tuple_set:
                tuple_set.add(tuple(new_basis))
                fold_list.append(new_basis)
        else:
            tuple_set.add(tuple(new_element))
            fold_list.append(new_element)

    return fold_list, t_list
def DSC_format_list(a, t, f):
    f_lst, t_list = generate_complex_list(a, t, f)
    t_set = set(tuple(t_list))
    fold_list = []
    for j,sublst in enumerate(f_lst):
        #sublst_a = sublst[:a]
        set_split = []
        for i,act in enumerate(sublst[:a]):
            set_split.append([[act],list(t_set - set(tuple([act])))])
        fold_list.append(set_split)
    return fold_list

def get_square_diagonal(matrix, num_trials, n_acts):
    return [matrix[i*num_trials:(i+1)*num_trials,i*num_trials:(i+1)*num_trials] for i in range(n_acts)]

def DSC_simple(categories, size, num_sub_groups, num_folds):
        if size > len(categories):
            return "Error: The size is larger than the number of categories."

        if num_folds <= 0:
            return "Error: n must be a positive integer."

        result = []
        current_group = []

        for _ in range(num_sub_groups * num_folds):
            temp_categories = categories.copy()
            random.shuffle(temp_categories)

            group1 = temp_categories[:size]
            group2 = temp_categories[size:]

            current_group.append([group1, group2])

            if len(current_group) == num_folds:
                result.append(current_group)
                current_group = []

        # Add any remaining groupings if iterations * n is not divisible by n
        if current_group:
            result.append(current_group)

        return result

'''a, t, f, s = 5, 6, 30, 5
result = DSC_format_list(a, t, f)
for i, sublist in enumerate(result):
    print(f"fold_list[{i}] = {sublist}")'''
'''a, t, f = 6, 9, 300
result,t_list = generate_complex_list(a, t, f)
for i, sublist in enumerate(result):
    print(f"fold_list[{i}] = {sublist}")'''

'''
# Example usage
a, t, f = 9, 6, 30
result = DSC_format_list(a, t, f)
for i, sublist in enumerate(result):
    print(f"fold_list[{i}] = {sublist}")'''

'''def generate_unique_combinations(a, f, t):
    original_t_list = list(range(t))
    t_list = original_t_list.copy()
    f_list = []
    used_combinations = set()

    for _ in range(f):
        while True:
            if len(t_list) < a:
                t_list = original_t_list.copy()

            # Generate a new a_list, allowing repetitions if necessary
            if len(t_list) >= a:
                a_list = random.sample(t_list, a)
            else:
                a_list = t_list + random.choices(original_t_list, k=a - len(t_list))

            a_tuple = tuple(a_list)

            # Check if this combination is new or has less repetition
            if a_tuple not in used_combinations:
                f_list.append(a_list)
                used_combinations.add(a_tuple)

                # Remove the selected numbers from t_list, but keep at least one of each
                counter = Counter(a_list)
                t_list = [x for x in t_list if counter[x] == 0 or (counter[x] > 0 and counter[x] <= t_list.count(x) - 1)]

                break

    return f_list

def meta_generate_unique_combinations(a, f, t):
    combos_by_folds_list = []
    remaining_a = a

    while remaining_a > 0:
        current_a = min(remaining_a, t)
        combos_by_folds_list.append(generate_unique_combinations(current_a, f, t))
        remaining_a -= current_a

    fold_list = []
    for fold_num in range(f):
        single_fold_list = []
        for combo_set in combos_by_folds_list:
            single_fold_list.extend(combo_set[fold_num])
        fold_list.append(single_fold_list)

    return fold_list

def dissimilarity(combo1, combo2):
    return sum(1 for a, b in zip(combo1, combo2) if a != b)


def total_dissimilarity(combo, group):
    return sum(dissimilarity(combo, other) for other in group)


def all_combinations_dissimilar(lst, n, num_folds, group_size):
    f_list = meta_generate_unique_combinations(n, num_folds, group_size)
    combinations = list(product(lst, repeat=n))
    if len(combinations) < num_folds:
        raise ValueError("Number of folds requested exceeds total number of combinations")

    random.shuffle(combinations)  # Start with a random order

    result = []

    while len(result) < num_folds:
        group = []
        for _ in range(min(group_size, num_folds - len(result))):
            if not group:
                # For the first element in the group, choose randomly
                choice = random.choice(combinations)
            else:
                # For subsequent elements, choose the most dissimilar within the group
                choice = max(combinations, key=lambda x: total_dissimilarity(x, group))

            group.append(choice)
            combinations.remove(choice)

        result.extend(group)

    return result'''