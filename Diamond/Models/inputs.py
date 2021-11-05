#Python file to store the arguments necessary to run RFM code.
#As of now, code is hardcoded to approximate the polynomial kernel of the order 4 i.e. <x,y>^4

dataFilePathA = '../CarbonData/'
trainFile = 'Train.xyz'
testFile = 'Test.xyz'
rCut = 3.7
wCut = 0.5
lMax = 12
nMax = 10
D = 1000 # number of random numbers
lambA = 0.1 # regularization parameter
paramDir = 'PublishedParamsRegRun2/' # directory to store parameters
randSeed = 494949
getRank = False #If true, compute matrix rank and conditional number. Useful to decide on number of features required.
