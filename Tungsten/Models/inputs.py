#Python file to store the arguments necessary to run RFM code.
#As of now, code is hardcoded to approximate the polynomial kernel of the order 4 i.e. <x,y>^4

dataFilePathA = '../TungstenData/'
trainFile = 'GAP_1.xyz'
testFile = 'GAP_6.xyz'
rCut = 5.0#4.2
wCut = 1.0
lMax = 2#14
nMax = 2#14
D =10000 # number of random numbers
lambA = 0.1 # regularization parameter
paramDir = 'PublishedParamsRegRun2/' # directory to store parameters
randSeed = 494949
