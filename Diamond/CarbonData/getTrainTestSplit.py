from ase.io import read,write

dataFilePathA = './'
trainFile = 'dataC.xyz'
TrainAtomConfigs = read(dataFilePathA+trainFile, index =":")

trainConfigs =[]
testConfigs = []
for i in range(len(TrainAtomConfigs)):
    if i%5!=0:
        trainConfigs.append(TrainAtomConfigs[i])
    else:
        testConfigs.append(TrainAtomConfigs[i])

write(dataFilePathA+'Train.xyz',trainConfigs)
write(dataFilePathA+'Test.xyz',testConfigs)
