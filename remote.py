#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import sys
from tsneFunctions import normalize_columns, tsne
import json
from itertools import chain


def get_all_keys(current_dict):
    children = []
    for k in current_dict:
        yield k
        if isinstance(current_dict[k], dict):
            children.append(get_all_keys(current_dict[k]))
    for k in chain.from_iterable(children):
        yield k


def listRecursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in listRecursive(v, key):
                yield found
        if k == key:
            yield v


def updateS(Y1, Y2, iY_Site_1, iY_Site_2):
    ''' It will collect Y and IY from local sites and return the updated Y

    args:
        Y1: low dimensional shared data from site 1
        Y2: low dimensional shared data from site 2
        iY_Site_1: Comes from site 1
        iY_Site_2: Comes from site 2

    Returns:
        Y: Updated shared value
    '''

    Y = (Y1 + Y2) / 2
    iY = (iY_Site_1 + iY_Site_2) / 2
    Y = Y + iY

    return Y


def demeanS(Y, average_Y):
    ''' Subtract Y(low dimensional shared value )by the average_Y and
    return the updated Y) '''

    return Y - np.tile(average_Y, (Y.shape[0], 1))


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

    raise Exception("I am here")

    shared_X = np.loadtxt('test/input/simulatorRun/shared_x.txt')
    shared_Y = np.loadtxt('test/input/simulatorRun/shared_y.txt')

    no_dims = args["input"]["local0"]["no_dims"]
    initial_dims = args["input"]["local0"]["initial_dims"]
    perplexity = args["input"]["local0"]["perplexity"]
    max_iter = args["input"]["local0"]["max_iterations"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    # shared data computation in tsne
    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase="remote")

    average_Y = (np.mean(shared_Y, 0))
    average_Y[0] = 0
    average_Y[1] = 0
    C = 0

    computation_output = {
        "output": {
            "shared_y": shared_Y.tolist(),
            "shared_x": shared_X.tolist(),
            "computation_phase": 'remote_1',
            'compAvgError': {
                'avgX': average_Y[0],
                'avgY': average_Y[1],
                'error': C
            }
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
    max_iter = args["cache"]["max_iterations"]
    average_Y = (np.mean(Y, 0))
    average_Y[0] = 0
    average_Y[1] = 0
    C = 0
    compAvgError = {
        'output': {
            'avgX': average_Y[0],
            'avgY': average_Y[1],
            'error': C
        }
    }

    localSite1SharedY = local_site1(args, json.dumps(compAvgError))
    #    localSite2SharedY = local_site2(args, json.dumps(compAvgError))

    for i in range(max_iter):
        numOfSites = 0
        # local site 1 computation
        localSite1SharedY, localSite1SharedIY, ExtractMeanErrorSite1 = local_site1(
            args,
            json.dumps(
                compAvgError, sort_keys=True, indent=4, separators=(',', ':')),
            computation_phase='computation')

        Y1 = np.loadtxt(localSite1SharedY["localSite1SharedY"])
        IY1 = np.loadtxt(localSite1SharedIY["localSite1SharedIY"])

        meanError1 = parser.parse_args(['--run', ExtractMeanErrorSite1])

        average_Y = (np.mean(Y1, 0))
        average_Y[0] = meanError1.run['output']['MeanX']
        average_Y[1] = meanError1.run['output']['MeanY']

        C = meanError1.run['output']['error']
        numOfSites += 1

        #        # local site 2 computation
        #        localSite2SharedY, localSite2SharedIY, ExtractMeanErrorSite2 = local_site2(
        #            args, json.dumps(compAvgError), computation_phase='computation')
        #        Y2 = np.loadtxt(localSite2SharedY["localSite2SharedY"])
        #        IY2 = np.loadtxt(localSite2SharedIY["localSite2SharedIY"])
        #
        #        meanError2 = parser.parse_args(['--run', ExtractMeanErrorSite2])
        #        average_Y[0] = average_Y[0] + meanError2.run['output']['MeanX']
        #        average_Y[1] = average_Y[0] + meanError2.run['output']['MeanY']
        #        C = C + (meanError2.run['output']['error'])
        #        numOfSites += 1

        # Here two local sites are considered. That's why it is divided by 2
        average_Y /= 2
        C /= 2

        Y = updateS(Y1, Y2, IY1, IY2)

        Y = demeanS(Y, average_Y)

        args["shared_Y"] = "Y_values.txt"

    # call local site 1 and collect low dimensional shared value of Y
    Y1 = local_site1(args, json.dumps(compAvgError), computation_phase='final')
    local1Yvalues = np.loadtxt(Y1["localSite1"])

    return Y, local1Yvalues, local2Yvalues


# if __name__ == '__main__':
#
#    np.random.seed(0)
#
#    parsed_args = json.loads(sys.argv[1])
#
#    if parsed_args["input"]["local0"]["computation_phase"] == 'local_noop':
#        computation_output = remote_1(parsed_args)
#        sys.stdout.write(computation_output)
#    elif parsed_args["input"]["local0"]["computation_phase"] == 'local_1':
#        computation_output = remote_2(parsed_args)
#        sys.stdout.write(computation_output)
#    else:
#        raise ValueError("Error occurred at Remote")


def remote_3(args):

    computation_output = {"output": {"final_embedding": 0}, "success": True}
    return json.dumps(computation_output)


if __name__ == '__main__':

    np.random.seed(0)

    parsed_args = json.loads(sys.argv[1])

    phase_key = list(
        listRecursive(parsed_args, 'computation_phase'))

    if 'local_noop' in phase_key:
        computation_output = remote_1(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_1' in phase_key:
        computation_output = remote_2(parsed_args)
        sys.stdout.write(computation_output)
    elif 'local_2' in phase_key:
        computation_output = remote_3(parsed_args)
        sys.stdout.write(computation_output)
    else:
        raise ValueError("Error occurred at Remote")
