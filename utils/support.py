#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def get_random_indices_for_dataset(num_users, data_per_user, shuffle=True):
    """ Creates and returns a permutation / list of indices for the current given number of users,
    where it keeps all the data per user consecutive to eachother, but randomizes the order of the users
    @param[in] num_users: Number of users
    @param[in] data_per_user: Number of datapoints per user
    @param[in] shuffle: if True, the each users data is shuffled. They stay consecutive but their relative order to eachother is randomized
    @returns
    """
    rand_users = np.random.permutation(range(num_users))
    indices = []
    for user_idx in rand_users:
        user_indices = []
        for i in range(data_per_user):
            user_indices.append(user_idx*data_per_user + i)
        if shuffle:
            indices += list(np.random.permutation(user_indices))
        else:
            indices += user_indices
    return indices

