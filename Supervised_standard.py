### Imports ###

import numpy as np
import h5py
import tensorflow.keras.backend as K
import tensorflow as tf
import json
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten, Layer, Reshape
from tensorflow.keras.models import Model, load_model
import argparse

### Constants ###

HOME = '.'
particlesTotal = 50
entriesPerParticle = 4
eventDataLength = 6
decayTypeColumn = -1
coldict = {"jet_eta":0,"jet_phi":1,"jet_EhadOverEem":2,"jet_sdmass":3,"jet_pt":4,"jet_mass":5}

### Define arguments ###

parser = argparse.ArgumentParser()

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--nTrain', action="store", dest="nTrain", type=int, default=-1)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=100)
parser.add_argument('--particles', action="store", dest="particles", type=int, default=30)
parser.add_argument('--masscut', action="store", dest="masscut", type=float, default=20.)
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=512)
parser.add_argument('--doAttention', action="store_true", dest="doAttention")
parser.add_argument('--useAlt', action="store_true", dest="useAlt")
parser.add_argument('--noSum', action="store_true", dest="noSum")

### Parse arguments ###

args = parser.parse_args()

particlesConsidered = args.particles

numberOfEpochs = args.nEpochs
batchSize = args.batchSize

print(args)

### Extract data ###

np.random.seed(422022)

print("Extracting")

fOne = h5py.File(f"{HOME}/data/test_nom_bkg.z", 'r')
nom_bkg_data = fOne["taggerInputs"][:]
print('Loaded nom_bkg')

fTwo = h5py.File(f"{HOME}/data/test_nom_sig.z", 'r')
nom_sig_data = fTwo["taggerInputs"][:]
print('Loaded nom_sig')

fThree = h5py.File(f"{HOME}/data/test_alt_bkg.z", 'r')
alt_bkg_data = fThree["taggerInputs"][:]
print('Loaded alt_bkg')

fFour = h5py.File(f"{HOME}/data/test_alt_sig.z", 'r')
alt_sig_data = fFour["taggerInputs"][:]
print('Loaded alt_sig')

nom_data = np.concatenate([
    nom_bkg_data,
    nom_sig_data
    ])
np.random.shuffle(nom_data)

alt_data = np.concatenate([
    alt_bkg_data,
    alt_sig_data
    ])
np.random.shuffle(alt_data)

### Split data and reshape ###

nom_data = nom_data[nom_data[:,eventDataLength]>0.,:]
alt_data = alt_data[alt_data[:,eventDataLength]>0.,:]

nom_data = nom_data[nom_data[:,coldict["jet_sdmass"]]>args.masscut,:]
alt_data = alt_data[alt_data[:,coldict["jet_sdmass"]]>args.masscut,:]

nomTrainingDataLength = int(len(nom_data)*0.8)
nomValidationDataLength = int(len(nom_data)*0.1)
altTrainingDataLength = int(len(alt_data)*0.8)
altValidationDataLength = int(len(alt_data)*0.1)

modelName = "IN_Supervised_%s"%("alt" if args.useAlt else "nom")
if args.label != "":
    modelName = modelName + "_" + args.label

os.mkdir(f'{HOME}/models/{modelName}')

print("Preparing Data")

particleDataLength = particlesTotal * entriesPerParticle

np.random.shuffle(nom_data)

nomLabels = nom_data[:, decayTypeColumn:]

nomParticleData = nom_data[:, eventDataLength:particleDataLength + eventDataLength]

nomParticleTrainingData = np.transpose(
    nomParticleData[0:nomTrainingDataLength, ].reshape(nomTrainingDataLength, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered],
    axes=(0, 2, 1))
nomTrainingLabels = np.array(nomLabels[0:nomTrainingDataLength])

nomParticleValidationData = np.transpose(
    nomParticleData[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength, ].reshape(nomValidationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesTotal)[:,:,:particlesConsidered],
    axes=(0, 2, 1))
nomValidationLabels = np.array(nomLabels[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength])

nomParticleTestData = np.transpose(nomParticleData[nomTrainingDataLength + nomValidationDataLength:, ].reshape(
    len(nomParticleData) - nomTrainingDataLength - nomValidationDataLength, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered],
                                axes=(0, 2, 1))
nomTestLabels = np.array(nomLabels[nomTrainingDataLength + nomValidationDataLength:])

altLabels = alt_data[:, decayTypeColumn:]

altParticleData = alt_data[:, eventDataLength:particleDataLength + eventDataLength]

altParticleTrainingData = np.transpose(
    altParticleData[0:altTrainingDataLength, ].reshape(altTrainingDataLength, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered],
    axes=(0, 2, 1))
altTrainingLabels = np.array(altLabels[0:altTrainingDataLength])

altParticleValidationData = np.transpose(
    altParticleData[altTrainingDataLength:altTrainingDataLength + altValidationDataLength, ].reshape(altValidationDataLength,
                                                                                         entriesPerParticle,
                                                                                         particlesTotal)[:,:,:particlesConsidered],
    axes=(0, 2, 1))
altValidationLabels = np.array(altLabels[altTrainingDataLength:altTrainingDataLength + altValidationDataLength])

altParticleTestData = np.transpose(altParticleData[altTrainingDataLength + altValidationDataLength:, ].reshape(
    len(altParticleData) - altTrainingDataLength - altValidationDataLength, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered],
                                axes=(0, 2, 1))
altTestLabels = np.array(altLabels[altTrainingDataLength + altValidationDataLength:])

# Defines the interaction matrices

# Defines the recieving matrix for particles
RR = []
for i in range(particlesConsidered):
    row = []
    for j in range(particlesConsidered * (particlesConsidered - 1)):
        if j in range(i * (particlesConsidered - 1), (i + 1) * (particlesConsidered - 1)):
            row.append(1.0)
        else:
            row.append(0.0)
    RR.append(row)
RR = np.array(RR)
RR = np.float32(RR)
RRT = np.transpose(RR)

# Defines the sending matrix for particles
RST = []
for i in range(particlesConsidered):
    for j in range(particlesConsidered):
        row = []
        for k in range(particlesConsidered):
            if k == j:
                row.append(1.0)
            else:
                row.append(0.0)
        RST.append(row)
rowsToRemove = []
for i in range(particlesConsidered):
    rowsToRemove.append(i * (particlesConsidered + 1))
RST = np.array(RST)
RST = np.float32(RST)
RST = np.delete(RST, rowsToRemove, 0)
RS = np.transpose(RST)

## Creates and trains the neural net ##

# Particle data interaction NN
inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

XdotRR = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RR, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRR")(inputParticle)
XdotRS = Lambda(lambda tensor: tf.transpose(tf.tensordot(tf.transpose(tensor, perm=(0, 2, 1)), RS, axes=[[2], [0]]),
                                            perm=(0, 2, 1)), name="XdotRS")(inputParticle)
Bpp = Lambda(lambda tensorList: tf.concat((tensorList[0], tensorList[1]), axis=2), name="Bpp")([XdotRR, XdotRS])

convOneParticle = Conv1D(64, kernel_size=1, activation="relu", name="convOneParticle")(Bpp)
convTwoParticle = Conv1D(32, kernel_size=1, activation="relu", name="convTwoParticle")(convOneParticle)
convThreeParticle = Conv1D(32, kernel_size=1, activation="relu", name="convThreeParticle")(convTwoParticle)

Epp = BatchNormalization(momentum=0.6, name="Epp")(convThreeParticle)

if args.doAttention:
    NHEADS = 8
    ATTDIM = 32
    ATTOUT = 32
    EppQ = Reshape((30, 29, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppQueryProj")(Epp))
    EppK = Reshape((30, 29, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppKeyProj")(Epp))
    EppV = Reshape((30, 29, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppValueProj")(Epp))
    EppMHAProjSoft = Lambda(lambda listOfTensors: K.softmax((tf.matmul(tf.transpose(listOfTensors[0], perm=(0, 1, 3, 2, 4)), tf.transpose(listOfTensors[1], perm=(0, 1, 3, 2, 4)), transpose_b=True) / tf.math.sqrt(tf.cast(particlesConsidered-1, listOfTensors[0].dtype)))), name="EppMHAProjSoft")([EppQ, EppK])
    EppMHAProj = Lambda(lambda listOfTensors: K.sum(tf.transpose(tf.matmul(tf.multiply(listOfTensors[0], 
        tf.reshape(tf.matmul(tf.reshape(tf.repeat(tf.cast(listOfTensors[2][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), [-1, particlesConsidered, particlesConsidered-1, 1]), tf.reshape(tf.repeat(tf.cast(listOfTensors[2][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), [-1, particlesConsidered, particlesConsidered-1, 1]), transpose_b=True), [-1, particlesConsidered, 1, particlesConsidered-1, particlesConsidered-1])
        ), tf.transpose(listOfTensors[1], perm=(0, 1, 3, 2, 4))), perm=(0, 1, 4, 2, 3)), axis=-1), name="EppMHAProj")([EppMHAProjSoft, EppV, inputParticle]) #sum over edges (keep pT agnostic)
    EppBar = Conv1D(ATTOUT, kernel_size=1, name="EppBar")(Reshape((30, -1))(EppMHAProj))
else:
    EppBar = Lambda(lambda listOfTensors: tf.transpose(tf.matmul(tf.transpose(listOfTensors[0], perm=(0, 2, 1)), tf.multiply(tf.expand_dims(tf.repeat(tf.cast(listOfTensors[1][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), axis=-1), np.expand_dims(RRT, axis=0))),
                                            perm=(0, 2, 1)), name="EppBar")([Epp, inputParticle])

C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1]), axis=2), name="C")(
    [inputParticle, EppBar])

convPredictOne = Conv1D(64, kernel_size=1, activation="relu", name="convPredictOne")(C)
convPredictTwo = Conv1D(32, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

O = Conv1D(32, kernel_size=1, activation="relu", name="O")(convPredictTwo)

# Calculate output

if args.noSum:
    OPost = Conv1D(16, kernel_size=1, activation="relu", name="OPost")(O)
    OBarPre = Lambda(lambda listOfTensors: tf.multiply(listOfTensors[0], tf.expand_dims(tf.cast(listOfTensors[1][:,:,0]>0., listOfTensors[0].dtype), axis=-1)), name="OBarPre")([OPost, inputParticle])
    OBar = GRU(64, activation="tanh", recurrent_activation="sigmoid", name="OBar")(OBarPre)
else:
    OBar = Lambda(lambda listOfTensors: K.sum(tf.multiply(listOfTensors[0], tf.expand_dims(tf.cast(listOfTensors[1][:,:,0]>0., listOfTensors[0].dtype), axis=-1)), axis=1), name="OBar")([O, inputParticle])


denseEndOne = Dense(64, activation="relu", name="denseEndOne")(OBar)
normEndOne = BatchNormalization(momentum=0.6, name="normEndOne")(denseEndOne)
denseEndTwo = Dense(32, activation="relu", name="denseEndTwo")(normEndOne)
normEndTwo = BatchNormalization(momentum=0.6, name="normEndTwo")(denseEndTwo)
denseEndThree = Dense(8, activation="relu", name="denseEndThree")(normEndTwo)
normEndThree = BatchNormalization(momentum=0.6, name="normEndThree")(denseEndThree)
output = Dense(1, activation="sigmoid", name="output")(normEndThree)

print("Compiling")


model = Model(inputs=[inputParticle], outputs=[output])

print(model.summary())
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print('Calculating')

modelCallbacks = [EarlyStopping(patience=10),
                    ModelCheckpoint(filepath=f"{HOME}/models/{modelName}/weights.h5", save_weights_only=True,
                                    save_best_only=True)]

history = model.fit([altParticleTrainingData[:args.nTrain]] if args.useAlt else [nomParticleTrainingData[:args.nTrain]], altTrainingLabels[:args.nTrain] if args.useAlt else nomTrainingLabels[:args.nTrain], epochs=numberOfEpochs, batch_size=batchSize,
                    callbacks=modelCallbacks,
                    validation_data=([altParticleValidationData] if args.useAlt else [nomParticleValidationData], altValidationLabels if args.useAlt else nomValidationLabels))

print("Loading weights")

model.load_weights(f"{HOME}/models/{modelName}/weights.h5")

model.save(f"{HOME}/models/{modelName}/model")

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])
with open(f"{HOME}/models/{modelName}/history.json", "w") as f:
    json.dump(history.history,f)



