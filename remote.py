#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from tsneFunctions import normalize_columns, tsne, listRecursive
import json


def remote_1(args):
    ''' It will receive parameters from dsne_multi_shot.
    After receiving parameters it will compute tsne on high
    dimensional remote data and pass low dimensional values
    of remote site data


       args (dictionary): {
            "shared_X" (str):  remote site data
            "shared_Label" (str): remote site labels
            "no_dims" (int): Final plotting dimensions
            "initial_dims" (int): number of dimensions that PCA should produce
            "perplexity" (int): initial guess for nearest neighbor
            "max_iter" (str):  maximum number of iterations during
                                tsne computation
            }
       computation_phase (string): remote

       normalize_columns:
           Shared data is normalized through this function

       Returns:
           Return args will contain previous args value in
           addition of Y[low dimensional Y values] values of shared_Y.
       args(dictionary):  {
           "shared_X" (str):  remote site data,
           "shared_Label" (str):  remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" : the low-dimensional remote site data
           }
       '''

    shared_X = np.loadtxt('test/input/simulatorRun/shared_x.txt')
    #    shared_Labels = np.loadtxt('test/input/simulatorRun/shared_y.txt')

    no_dims = args["input"]["local0"]["no_dims"]
    initial_dims = args["input"]["local0"]["initial_dims"]
    perplexity = args["input"]["local0"]["perplexity"]
    max_iter = args["input"]["local0"]["max_iterations"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase="remote")

    computation_output = {
        "output": {
            "shared_y": shared_Y.tolist(),
            "shared_x": shared_X.tolist(),
            "computation_phase": 'remote_1',
        },
        "cache": {
            "shared_y": shared_Y.tolist(),
            "max_iterations": max_iter
        }
    }

    return json.dumps(computation_output)


def remote_2(args):
    '''
    args(dictionary):  {
        "shared_X"(str): remote site data,
        "shared_Label"(str): remote site labels
        "no_dims"(int): Final plotting dimensions,
        "initial_dims"(int): number of dimensions that PCA should produce
        "perplexity"(int): initial guess for nearest neighbor
        "shared_Y": the low - dimensional remote site data

    Returns:
        Y: the final computed low dimensional remote site data
        local1Yvalues: Final low dimensional local site 1 data
        local2Yvalues: Final low dimensional local site 2 data
    }
    '''
    Y = args["cache"]["shared_y"]
    average_Y = (np.mean(Y, 0))
    average_Y[0] = 0
    average_Y[1] = 0
    C = 0

    compAvgError = {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}

    computation_output = {
        "output": {
            "compAvgError": compAvgError,
            "computation_phase": 'remote_2'
        },
        "cache": {
            "compAvgError": compAvgError
        }
    }

    return json.dumps(computation_output)


def remote_3(args):
    C = args["cache"]["compAvgError"]["erro"]

    Y = args["input"]["local_Y"]

    average_Y = (np.mean(Y, 0))
    average_Y[0] = np.mean(
        [args['input'][site]['MeanX'] for site in args["input"]])
    average_Y[1] = np.mean(
        [args['input'][site]['MeanY'] for site in args["input"]])

    C = C + np.mean([args['input'][site]['error'] for site in args["input"]])

    meanY = np.mean([args["input"][site]["local_Y"] for site in args["input"]])
    meaniY = np.mean(
        [args["input"][site]["local_iY"] for site in args["input"]])

    Y = meanY + meaniY
    Y -= np.tile(average_Y, (Y.shape[0], 1))

    computation_output = {"output": {"Y": Y.tolist()}, "cache": {}}

    return json.dumps(computation_output)


def remote_4(args):

    # Final aggregation step
    computation_output = {"output": {"final_embedding": 0}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

    np.random.seed(0)
    parsed_args = json.loads(sys.argv[1])

    phase_key = list(listRecursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_1' in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_2' in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_3' in phase_key:
        computation_output = remote_4(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
