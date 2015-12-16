import numpy as np
import sys


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
    rand =  np.random.choice(row, k, replace = False)
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
            # assign the cluster which is at min distance
            idx[i] = np.argmin(diff)

        # update the cluster centers
        getnewCenters(k,idx,data,centers)

        centers = np.rint(centers)
        # check if allocation remain same
        if np.array_equal(old_idx,idx):
            print "converge at :" + str(itr)
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

        print "FOR XCORD OF NEW FACILITY :" + str(i + 1)
        xcord = minimizeCost(centers[i][0], args)

        # find optimal y cord
        facility_ycord = data[indexes,3]
        args = (weights,facility_ycord)
        print "FOR YCOORD OF NEW FACILITY :"+ str(i + 1)
        ycord = minimizeCost(centers[i][1], args)
        # update center
        centers[i][0] = xcord
        centers[i][1] = ycord
'''
 Method to find cordinate at which cost is minimum
'''
def minimizeCost(initial_point, args):
    step = 0.009
    coord = initial_point
    val = newobj(initial_point,args)
    delta = 0

    if newobj(initial_point,args) < newobj(initial_point + step, args):
        step = -0.009

    print "obj cost initital :" + str(val)
    while(delta == 0):
        oldcoord = coord
        oldval = val
        coord = coord + step
        val = newobj(coord, args)
        print "obj cost : " + str(val)
        if(val > oldval ):
            delta = 1

    return oldcoord


def newobj(center, args):
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
