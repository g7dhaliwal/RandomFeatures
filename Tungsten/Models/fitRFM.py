# python program to perform RFF model on SOAP features
import numpy as np
import quippy 
from quippy import descriptors
from ase.io import read
import matplotlib.pyplot as plt
import time
import inputs
import os
from os import path
import sys

def computeWeights(A, y):
    weights = np.linalg.lstsq(A,y,rcond= None)[0]
    return weights

def getInitialFeatureDim(trainInputConfig, rCut,wCut,lMax=14,nMax=14,atomSigma=0.5):
    desc = descriptors.Descriptor("soap cutoff="+str(rCut)+" l_max="+str(lMax)+" n_max="+str(nMax)+" atom_sigma="+str(atomSigma)+" ")
    trainInputConfig.set_cutoff(desc.cutoff()+wCut)
    trainInputConfig.calc_connect()
    d = desc.calc(trainInputConfig, grad = True)
    energyFeatures = d['descriptor']
    return energyFeatures.shape[1]

def getSOAPFeatures(trainInputConfig, rCut,wCut,lMax=14,nMax=14,atomSigma=0.5):
    # this function should return the features and gradient of the featres as computed from quippy
    # INPUT:- TRAINING INPUT CONFIGURATION
    # Output:- SOAP FEATURES AND GRADIENTS WITH GRADIENT INDICES
    #----------------------------------
    desc = descriptors.Descriptor("soap cutoff="+str(rCut)+" l_max="+str(lMax)+" n_max="+str(nMax)+" atom_sigma="+str(atomSigma)+" ")
    trainInputConfig.set_cutoff(desc.cutoff()+wCut)
    trainInputConfig.calc_connect()
    d = desc.calc(trainInputConfig, grad = True)
    energyFeatures = d['descriptor']
    gradIndex = d['grad_index_0based']
    gradMatrix = d['grad']

    return (energyFeatures, gradMatrix, gradIndex)

def getEnergyForce(energyFeatures, omega):
    # this function should compute energy based on RMFM model and return to main energy matrix
    # further this function should also return min and max for normalization process
    #Input:-SOAP FEATURES(nAtoms*nFeatures), omega(nFeatures,nOmegas)
    #OUTPUT:- energy,minLE,maxLE in RFF SPACE
    sqD = 1.0/np.sqrt(omega.shape[-1])
    rffInput = np.tensordot(energyFeatures, omega, axes=([1],[0]))
    energyA = sqD*np.prod(rffInput, axis=1)
    minLE = np.min(energyA, axis=0)
    maxLE = np.max(energyA, axis=0)
    energyA = np.sum(energyA, axis=0).reshape(1,-1)
    return (energyA, minLE, maxLE)

def getdEdF(energyFeatures, omega):
    # shape of omega nF*4*nD
    nA = energyFeatures.shape[0]
    nF = omega.shape[0]
    nD = omega.shape[2]
    sqD = 1.0/np.sqrt(omega.shape[-1])

    #CODE BELOW PRODUCES OUTPUT EQUIVALENT TO AUTOGRAD
    #TODO - Need to automate these lines.
    #TODO - The lines below will only work for 4th order kernel
    #TODO - Need to make sure it can run for any kernel order
    pD1 = np.dot(energyFeatures, omega[:,0,:])
    pD2 = np.dot(energyFeatures, omega[:,1,:])
    pD3 = np.dot(energyFeatures, omega[:,2,:])
    pD4 = np.dot(energyFeatures, omega[:,3,:])
    dEdpD1 = (pD2*pD3*pD4*sqD).reshape(nA,1,nD)
    dEdpD2 = (pD1*pD3*pD4*sqD).reshape(nA,1,nD)
    dEdpD3 = (pD1*pD2*pD4*sqD).reshape(nA,1,nD)
    dEdpD4 = (pD1*pD2*pD3*sqD).reshape(nA,1,nD)
    dEdF = dEdpD1*(omega[:,0,:])#.transpose())
    dEdF = dEdF + dEdpD2*(omega[:,1,:])#.transpose())
    dEdF = dEdF + dEdpD3*(omega[:,2,:])#.transpose())
    dEdF = dEdF + dEdpD4*(omega[:,3,:])#.transpose())
    dEdF = dEdF.transpose([0,2,1])
    return dEdF

def computeForceMatrix(energyFeatures, omega, gradMatrix, gradIndex):
    # this function computes the gradient of the computeEnergy
    #INPUT:- SOAP FEATURES
    #OUPUT:- FORCE A MATRIX
    nAtoms = energyFeatures.shape[0]
    nOmega = omega.shape[-1]
    nFeatures = energyFeatures.shape[1]
    forceAuto = getdEdF(energyFeatures, omega)
    forceA = np.zeros((nAtoms,nOmega,3))
    for i in range(gradIndex.shape[0]):
        iAtom = gradIndex[i,0]
        jAtom = gradIndex[i,1]
        forceA[jAtom,:,:] = forceA[jAtom,:,:]+np.dot(forceAuto[iAtom,:,:], gradMatrix[i,:,:].T)
    fX = -forceA[:,:,0]
    fY = -forceA[:,:,1]
    fZ = -forceA[:,:,2]
    return np.vstack((fX,fY,fZ))

def getPythonEnergyForce(omega,trainingIndicesPerProcessor, TrainAtomConfigs, rCut,wCut,quippyConfigs,lMax,nMax):
    # FOR EACH TRAINING INDEX IN TRAININGINDICESPERPROCESSOR
    #CALL SOAP FEATURES FUNCTION
    #CALL GETENERGY FUNCTION
    #CALL COMPUTEAUTOGRADFORCES FUNCTION
    #APPED THE RESULTS IN APPROPRIATE MATRICES
    D = omega.shape[-1]
    energyAMatrix = np.zeros((0,D))
    forceAMatrix = np.zeros((0,D))
    minMatrix = np.zeros((0,D))
    maxMatrix = np.zeros((0,D))
    trueEnergy = np.zeros((0,1))
    trueForce = np.zeros((0,1))
    numAtoms = np.zeros((0,1))
    for i in range(len(trainingIndicesPerProcessor)):
        index = trainingIndicesPerProcessor[i]
        atomConfig = TrainAtomConfigs[index]
        descConfig = quippyConfigs[index]
        energyFeatures, gradMatrix, gradIndex = getSOAPFeatures(descConfig, rCut,wCut,lMax,nMax)
        #print("sum of gradMatrix ", np.sum(gradMatrix))
        numAtoms = np.vstack((numAtoms, energyFeatures.shape[0]))
        energyA, minLE, maxLE = getEnergyForce(energyFeatures, omega)
        #print("SHAPE OF ENERGYA ", energyA.shape)
        forceA = computeForceMatrix(energyFeatures, omega, gradMatrix, gradIndex)
        energyAMatrix = np.vstack((energyAMatrix,energyA))
        forceAMatrix = np.vstack((forceAMatrix, forceA))
        minMatrix = np.vstack((minMatrix,minLE))
        maxMatrix = np.vstack((maxMatrix, maxLE))
        tEnergy = atomConfig.get_total_energy()
        trueEnergy = np.vstack((trueEnergy, tEnergy))
        qForce = np.array(quippyConfigs[index].force).T
        gradX = qForce[:,0].reshape(-1,1)
        gradY = qForce[:,1].reshape(-1,1)
        gradZ = qForce[:,2].reshape(-1,1)
        trueForce = np.vstack((trueForce,gradX,gradY,gradZ))
        # setup toolbar
        #toolbar_width = len(trainingIndicesPerProcessor)
        #sys.stdout.write("[%s]" % (" " * toolbar_width))
        #sys.stdout.flush()
        #sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['
        if i%100==0:
            printMess = 'Generating features for '+str(i+1)+'/'+str(len(trainingIndicesPerProcessor))
            print(printMess)
            #sys.stdout.write("-")
            #sys.stdout.flush()
    #sys.stdout.write("]\n") # this ends the progress bar


    return (energyAMatrix, forceAMatrix, minMatrix, maxMatrix, trueEnergy,trueForce,numAtoms)

def main():
    stTime = time.time()
    dataFilePathA = inputs.dataFilePathA#'../TungstenData/'
    trainFile = inputs.trainFile#'GAP_6.xyz'
    TrainAtomConfigs = read(dataFilePathA+trainFile, index =":")
    rCut = inputs.rCut#5.0#4.2
    wCut = inputs.wCut#1.0
    testFile = inputs.testFile#'GAP_6.xyz'
    TestAtomConfigs = read(dataFilePathA+testFile, index =":")
    trainSampleSize = len(TrainAtomConfigs)
    testSampleSize = len(TestAtomConfigs)
    trainingIndices = [i for i in range(trainSampleSize)]
    print("LENGTH OF TRAINING INDICES ", len(trainingIndices))
    testIndices = [i for i in range(testSampleSize)]
    print("LENGTH OF TEST INDICES ", len(testIndices))
    qTrain = quippy.AtomsList(dataFilePathA+trainFile)
    qTest = quippy.AtomsList(dataFilePathA+testFile)
    lMax = inputs.lMax
    nMax = inputs.nMax
    
    D = inputs.D#10000
    randSeed = inputs.randSeed
    np.random.seed(randSeed)
    inp_space = [-1.0,1.0]
    #TODO - Move the omega generation into getPythonEnergyForce(). This won't be helpful if you are parallelizing the code
    nF = getInitialFeatureDim(qTrain[0], rCut,wCut,lMax,nMax)
    print(lMax,nMax,nF,D)
    omega = np.random.choice(inp_space, size=(nF,4,D))# A TENSOR OF SHAPE DX4XNFEATURES

    args = {'omega':omega,'trainingIndicesPerProcessor':trainingIndices, 
            'TrainAtomConfigs':TrainAtomConfigs, 'rCut':rCut,'wCut':wCut,'quippyConfigs':qTrain,
             'lMax':lMax,'nMax':nMax}
    trainEA, trainFA, minMatrix, maxMatrix, trainTrueEnergy,trainTrueForce,trainNumAtoms =getPythonEnergyForce(**args)
    combA = np.vstack((trainEA,trainFA))
    combY = np.vstack((trainTrueEnergy, trainTrueForce))
    lambA = inputs.lambA 
    modA = np.dot(combA.T,combA) + lambA*np.eye(D)
    modY = np.dot(combA.T,combY)
    weights = computeWeights(combA,combY)
    print("Shape of weights ", weights.shape)
    yPred = np.dot(trainEA , weights).reshape(-1,1)
    fPred = np.dot(trainFA, weights).reshape(-1,1)
    if inputs.getRank:
        print('Rank of energy, force matrix', np.linalg.matrix_rank(trainEA), np.linalg.matrix_rank(trainFA))
        print('Rank of combined regression matrix', np.linalg.matrix_rank(combA))
    print("TRAIN - MAE in energy, forces, ", np.mean(np.abs(yPred-trainTrueEnergy)), 
                                            np.mean(np.abs(fPred- trainTrueForce)))
    print("TRAIN - RMS in energy, forces, ", np.sqrt(np.mean((yPred- trainTrueEnergy)**2)), 
                                             np.sqrt(np.mean((fPred- trainTrueForce)**2)))

    paramDir = inputs.paramDir#'PublishedParamsRegRun2/'#'GAP5_5000Seed3UN'
    if not os.path.exists(paramDir):
        os.mkdir(paramDir)
    #TODO - Make sure that if the above directory does not exist then OS makes it
    np.save(paramDir+'TungWeights.npy', weights)
    np.save(paramDir+'TungOmega.npy',omega)
    #np.save(paramDir+'TungMin.npy',minLE)
    #np.save(paramDir+'TungRange.npy',rangeA)
    
    print("-------TEST RESULTS--------")
    args = {'omega':omega,'trainingIndicesPerProcessor':testIndices,
            'TrainAtomConfigs':TestAtomConfigs, 'rCut':rCut,'wCut':wCut,'quippyConfigs':qTest,
            'lMax':lMax,'nMax':nMax}
    testEA, testFA, minD, maxD, testTrueEnergy,testTrueForce,testNumAtoms =getPythonEnergyForce(**args)

    testEnergyPred = np.dot(testEA, weights)
    testForcePred = np.dot(testFA, weights)
    print("TEST - MAE in energy, forces, ", np.mean(np.abs(testEnergyPred - testTrueEnergy)), 
                                             np.mean(np.abs(testForcePred - testTrueForce)))
    print("TEST - RMS in energy, forces, ", np.sqrt(np.mean((testEnergyPred - testTrueEnergy)**2)), 
                                             np.sqrt(np.mean((testForcePred - testTrueForce)**2)))
    print("TIME TAKE: ", time.time()-stTime)
    saveArray = np.hstack((testForcePred, testTrueForce))
    np.save(paramDir+'forcePred.npy',saveArray)
    saveArray = np.hstack((testEnergyPred, testTrueEnergy))
    np.save(paramDir+'energyPred.npy',saveArray)

main()
