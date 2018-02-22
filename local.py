#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import json
import sys
from tsneFunctions import normalize_columns, tsne, master_child
from itertools import chain
from collections import OrderedDict


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, OrderedDict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def updateL(Y, G):
    ''' It will take Y and IY of only local site data and
    return the updated Y'''
    return Y + G


def demeanL(Y, average_Y):
    ''' It will take Y and average_Y of only local site data and
    return the updated Y by subtracting IY'''
    return Y - np.tile(average_Y, (Y.shape[0], 1))


def local_noop(args):
    input_list = args["input"]

    computation_output = {
        "output": {
            "computation_phase": 'local_noop',
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"],
            "max_iterations": input_list["max_iterations"]
        },
        "cache": {
            "no_dims": input_list["no_dims"],
            "initial_dims": input_list["initial_dims"],
            "perplexity": input_list["perplexity"]
        }
    }

    return json.dumps(computation_output)


def local_1(args):
    ''' It will load local data and download remote data and
    place it on top. Then it will run tsne on combined data(shared + local)
    and return low dimensional shared Y and IY

       args (dictionary): {
           "shared_X" (str): file path to remote site data,
           "shared_Label" (str): file path to remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" (str):  the low-dimensional remote site data
           }


       Returns:
           computation_phase(local): It will return only low dimensional
           shared data from local site
           computation_phase(final): It will return only low dimensional
           local site data
           computation_phase(computation): It will return only low
           dimensional shared data Y and corresponding IY
       '''

    # corresponds to local

    shared_X = np.array(args["input"]["shared_x"])
    shared_Y = np.array(args["input"]["shared_y"])
    no_dims = args["cache"]["no_dims"]
    initial_dims = args["cache"]["initial_dims"]
    perplexity = args["cache"]["perplexity"]
    sharedRows, sharedColumns = shared_X.shape

    Site1Data = np.loadtxt('test/input/simulatorRun/site1_x.txt')

    # create combinded list by local and remote data
    combined_X = np.concatenate((shared_X, Site1Data), axis=0)
    combined_X = normalize_columns(combined_X)

    # create low dimensional position
    combined_Y = np.random.randn(combined_X.shape[0], no_dims)
    combined_Y[:shared_Y.shape[0], :] = shared_Y

    local_Y, local_dY, local_iY, local_gains, local_P, local_n = tsne(
        combined_X,
        combined_Y,
        sharedRows,
        no_dims=no_dims,
        initial_dims=initial_dims,
        perplexity=perplexity,
        computation_phase="local")

    computation_output = {
        "output": {
            "localSite1SharedY": local_Y.tolist(),
            'computation_phase': "local_1"
        },
        "cache": {
            "local_Y": local_Y,
            "local_dY": local_dY,
            "local_iY": local_iY,
            "local_P": local_P,
            "local_n": local_n,
            "local_gains": local_gains,
            "shared_rows": sharedRows
        }
    }

    return json.dumps(computation_output)


def local_2(args):

    # corresponds to computation

    local_sharedRows = args["cache"]["shared_rows"]

    compAvgError1 = args["input"]["compAvgError"]
    local_Y = args["cache"]["local_Y"]
    local_dY = args["cache"]["local_dY"]
    local_iY = args["cache"]["local_iY"]
    local_P = args["cache"]["local_P"]
    local_n = args["cache"]["local_n"]
    local_gains = args["cache"]["local_gains"]

    shared_Y = args["input"]["shared_Y"]
    local_Y[:local_sharedRows, :] = shared_Y
    C = compAvgError1['error']
    demeanAvg = (np.mean(local_Y, 0))
    demeanAvg[0] = compAvgError1['avgX']
    demeanAvg[1] = compAvgError1['avgY']
    local_Y = demeanL(local_Y, demeanAvg)

    local_Y, local_iY, local_Q, C, local_P = master_child(
        local_Y, local_dY, local_iY, local_gains, local_n, local_sharedRows,
        local_P, iter, C)
    local_Y[local_sharedRows:, :] = updateL(local_Y[local_sharedRows:, :],
                                            local_iY[local_sharedRows:, :])

    meanValue = (np.mean(local_Y, 0))
    computation_output = {
        "output": {
            "MeanX": meanValue[0],
            "MeanY": meanValue[1],
            "error": C,
            "local_iY": local_iY.tolist(),
            "local_Y": local_Y.to_list(),
            "computation_phase": "local_2"
        },
        "cache": {}
    }

    return json.dumps(computation_output)


def local_3(args):
    # corresponds to final
    return 0


def get_all_keys(current_dict):
    children = []
    for k in current_dict:
        yield k
        if isinstance(current_dict[k], dict):
            children.append(get_all_keys(current_dict[k]))
    for k in chain.from_iterable(children):
        yield k


if __name__ == '__main__':
    np.random.seed(0)

    parsed_args = json.loads(sys.argv[1])
    phase_key = list(
        listRecursive(OrderedDict(parsed_args), 'computation_phase'))

    if not phase_key:
        computation_output = local_noop(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_1' in phase_key:
        computation_output = local_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'remote_2' in phase_key:
        computation_output = local_2(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Local")
