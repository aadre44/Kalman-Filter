import inspect
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
#https://stackoverflow.com/questions/13901997/kalman-2d-filter-in-python
#https://statweb.stanford.edu/~candes/acm116/Handouts/Kalman.pdf
'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)

'''
Kalman 2D
'''
scalar = .45 #.43
scalar2 = .423 #3
I = [[1, 0], [0, 1]]
A = np.matrix(I)
B = np.matrix(I)
H = np.matrix(I)
P_init = scalar*np.matrix(I)
P_init2 = scalar2*np.matrix(I)
Q = np.matrix([[.0001, .00002], [.00002, .001]])
R = np.matrix([[.01, .005], [.005, .01]])
Q2 = np.matrix([[2, 0.5], [0.5, 2]])
R2 = np.matrix([[200, 50], [50, 300]])

def kalman2d(data):
    global I, A, B, H, P_init, Q, R
    estimated = []
    xPast = np.matrix([[0], [0]])
    PPast = P_init
    count = 0
    avg = 0
    for i in data:
        u = np.matrix([[i[0]], [i[1]]])
        z = np.matrix([[i[2]], [i[3]]])
        x_esti = (A*xPast)+(B*u) #time update
        P_esti = A*PPast*A+Q
        K = (P_esti*H)/(H*P_esti*H+R) # Kalman Gain
        xNew = x_esti+K*(z-H*x_esti) #measurement update
        PNew = (np.matrix(I)-K*H)*P_esti #new P error matrix
        estimated.append(np.squeeze(np.asarray(xNew)))#[xNew[0], xNew[1]])
        xPast = xNew
        PPast = PNew
        covar = np.squeeze(np.asarray(PNew))
        error = covar[0][0] + covar[1][1]
        errorX = covar[0][0]
        print(str(count)+" Error: "+str(errorX))
        avg+= errorX
        count+=1
    # Your code starts here np.squeeze(np.asarray())
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here 
    #_raise_not_defined()
    avg = avg/count
    #print("AVG Error: " + str(avg))
    return estimated
x_Past = np.matrix([[0], [0]])
p_Past = P_init
def kalmanOnce(u1, u2, z1, z2):
    global I, A, B, H, P_init, Q2, R2, p_Past, x_Past
    estimated = []
    xPast = x_Past
    PPast = p_Past
    u = np.matrix([[u1], [u2]])
    z = np.matrix([[z1], [z2]])
    x_esti = (A*xPast)+(B*u) #time update
    P_esti = A*PPast*A+Q2
    K = (P_esti*H)/(H*P_esti*H+R2) # Kalman Gain
    xNew = x_esti+K*(z-H*x_esti) #measurement update
    PNew = (np.matrix(I)-K*H)*P_esti #new P error matrix
    estimated.append(np.squeeze(np.asarray(xNew)))#[xNew[0], xNew[1]])
    x_Past = xNew
    p_Past = PNew
    covar = np.squeeze(np.asarray(PNew))
    error = covar[0][0] + covar[1][1]
    errorX = covar[0][0]
    #print(str(count) + " Error: " + str(errorX))

    # Your code starts here np.squeeze(np.asarray())
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    #_raise_not_defined()
    return (estimated, PNew)
def matrixMult(m1, m2):
    m1 = np.matrix(m1)
    m2 = np.matrix(m2)
    result = matrix
'''
Plotting
'''
def plot(data, output):
    observationsX=[]
    observationsY = []
    estimatesX=[]
    estimatesY= []
    maxX = None
    lowX = None
    maxY = None
    lowY = None
    for i in data:
        if i[2] > maxX or maxX == None:
            maxX = i[0]
        if i[2] < lowX or lowX == None:
            lowX = i[0]
        if i[3] > maxY or maxY == None:
            maxY = i[1]
        if i[3] < lowY or lowY == None:
            lowY = i[1]
        observationsX.append(i[2])
        observationsY.append(i[3])
    for h in output:
        if h[0] > maxX or maxX == None:
            maxX = h[0]
        if h[0] < lowX or lowX == None:
            lowX = h[0]
        if h[1] > maxY or maxY == None:
            maxY = h[1]
        if h[1] < lowY or lowY == None:
            lowY = h[1]
        estimatesX.append(np.squeeze(np.asarray(h[0])))
        estimatesY.append(np.squeeze(np.asarray(h[1])))
    plt.plot(observationsX, observationsY, 'bo-')
    plt.plot(estimatesX, estimatesY, 'ro-')
    #print("max low "),
    #print([lowX, maxX, lowY, maxY])
    plt.axis()  # creates the x and y bounds for the graph
    plt.show()
    #_raise_not_defined()
    return

'''
Kalman 2D 
'''
scalar2 = .23 #.43
trials = 200
num = 0
errorX_1 = 0
def kalman2d_shoot(ux, uy, ox, oy, reset=False):
    global I, A, B, H, P_init, Q2, R2, p_Past, x_Past,errorX_1, num
    if reset == True:
        x_Past = np.matrix([[0], [0]])
        p_Past = P_init2
        errorX_1=0
        num = 0
    check = kalmanOnce(ux, uy, ox, oy)
    check2 = check[1]
    checkX = check[0][0][0]
    checkY = check[0][0][1]

    covar = np.squeeze(np.asarray(check2))
    error = covar[0][0] + covar[1][1]
    if(num == 0):
        errorX_1 = covar[0][0]
    errorX = covar[0][0]
    if errorX <= 2.2004591869956913 or errorX_1 < errorX:# .0009625008371902825: #0.0009439951772983787: .0036636976: 0.0009501444960858254

        decision = (checkX, checkY, True)
    else:
        decision = (checkX, checkY, False)
    num+=1
    return decision

'''
Kalman 2D 
'''
trials = 200
count = 0
def kalman2d_adv_shoot(ux, uy, ox, oy, reset=False):
    global I, A, B, H, P_init2, Q2, R2, p_Past, x_Past, trials, count, errorX_1
    if reset == True:
        x_Past = np.matrix([[0], [0]])
        p_Past = P_init2
        count = 0
    count+=1
    #print("COUNT: "+ str(count))
    check = kalmanOnce(ux, uy, ox, oy)

    checkX = check[0][0][0]
    checkY = check[0][0][1]

    covar = np.squeeze(np.asarray(check[1]))
    error = covar[0][0] + covar[1][1]
    if (count == 1):
        errorX_1 = covar[0][0]
    errorX = covar[0][0]
    if  errorX_1 < errorX and count >= trials/trials:
        decision = (checkX, checkY, True)
        #print("Error in X: "+str(errorX))
        #score = count+np.log(errorX/2)
        #print("Score: " + str(score))
    elif  errorX_1 > errorX and count >= trials/10:
        decision = (checkX, checkY, True)
        #print("Error in X: "+str(errorX))
        #score = count+np.log(errorX/2)
        #print("Score: " + str(score))
    else:
        decision = (checkX, checkY, False)
    errorX_1 = covar[0][0]
    return decision



