import numpy as np
import sys
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar


def parseFile(filename):
    '''
        method reads data from file
        INPUT
            full qualified filename
        OUTPUT
            a) Number of facilities (m)
            b) Number of new facilities (n)
            c) m x 4 matrix of data

    '''
    with open(filename) as f:
        content = f.readlines()
    # mn = list [m, n]
    mn = [int(s) for s in content[0].split() if s.isdigit()]
    # initialize array to store date
    data = np.zeros([mn[0],4])

    for i in range(0,mn[0]):
        k = 0
        for s in content[i+1].split():
            if s.isdigit():
                data[i][k] = int(s)
                k = k + 1

    return mn[0],mn[1],data


def kmeans(k, data):
    row = data.shape[0]
    col = data.shape[1]
    # choose random centers to start
    rand = np.array(range(0,k))
    centers = data[rand, 2:4]
    centers = centers + 1

    # The assignments of points to clusters. If idx(i) == c then the point
    # data(i, :) belongs to the cth cluster.
    idx = np.zeros([row, 1])

    itr = 0
    MAX_ITR = 100

    while True:
        old_idx = np.copy(idx)
        # compute weighted distance from each point to centers and assifn
        # to nearest cluster
        for i in range(0,row):
            repeat = np.tile(data[i,[2,3]], (k, 1))
            weight = data[i,1]
            diff = np.sum(abs(repeat - centers),1)
            wdist = weight*diff
            # assign the cluster which is at min weighted distance
            idx[i] = np.argmin(wdist)

        # update the cluster centers
        getnewCenters(k,idx,data,centers)

        centers = np.rint(centers)
        # check if allocation remain same
        if np.array_equal(old_idx,idx):
            break


        itr = itr + 1
        if itr > MAX_ITR:
            break

    printAns(k, idx, data, centers)

def getnewCenters(k, idx, data, centers):
    for i in range(0, k):
        indexes = np.sum(np.transpose((idx == i).nonzero()), 1)

        weights = data[indexes,1]
        # find optimal x cord
        facility_xcord = data[indexes,2]
        args = (weights, facility_xcord)

        xcord = minimize_scalar(newobj,args=args)

        # find optimal y cord
        facility_ycord = data[indexes,3]
        args = (weights,facility_ycord)
        ycord = minimize_scalar(newobj,args=args)

        # update center
        centers[i][0] = xcord.x
        centers[i][1] = ycord.x



def newobj(center, *args):
    weight, facility = args
    cost = np.sum(weight*(abs(facility-center)))
    return cost


def printAns(k, idx, data, centers):
    print "%s %10s %10s %20s".format() % ("Faclity", "X cord", "Y cord", "Cost")
    for i in range(0,k):
        indexes = np.sum(np.transpose((idx == i).nonzero()), 1)

        weights = data[indexes,1]
        facility = data[indexes,2:4]
        recenter = np.tile(centers[i],(facility.shape[0], 1))
        diff =  np.sum(abs(recenter- facility),1)
        cost = np.sum(weights*diff)
        print "%d %18f %12f %20f".format() % (i, centers[i][0], centers[i][1], cost)


def main(argv):
    assert len(argv) == 1
    m,n,data = parseFile(argv[0])
    kmeans(n, data)



if __name__ == "__main__":
    main(sys.argv[1:])
