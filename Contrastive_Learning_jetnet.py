### Imports ###

import numpy as np
import h5py
from math import *
import tensorflow.keras.backend as K
import tensorflow as tf
import json
import os
import math
import argparse
try:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
except:
    ...
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv1D, Lambda, Dot, Flatten, Reshape
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from LARS_Opt import LARS
import wasserstein

particlesTotal = 30
entriesPerParticle = 3
eventDataLength = 4
decayTypeColumn = -1
coldict = {"jet_pt":0,"jet_eta":1,"jet_sdmass":2,"jet_nconst":3}

parser = argparse.ArgumentParser()

### Define arguments ###

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--labeladd', action="store", dest="labeladd", type=str, default="")
parser.add_argument('--nSamples', action="store", dest="nSamples", type=int, default=-1)
parser.add_argument('--warmLR', action="store", dest="warmLR", type=float, default=-1.)
parser.add_argument('--LARS', action="store_true", dest="LARS")
parser.add_argument('--Adam', action="store_true", dest="Adam")
parser.add_argument('--LR', action="store", dest="LR", type=float, default=0.1)
parser.add_argument('--fLR', action="store", dest="fLR", type=float, default=-1.)
parser.add_argument('--l2reg', action="store", dest="l2reg", type=float, default=0.0)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=100)
parser.add_argument('--particles', action="store", dest="particles", type=int, default=30)
parser.add_argument('--decayEpochs', action="store", dest="decayEpochs", type=int, nargs=2, default=[10,90])
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=1024)
parser.add_argument('--redoPairs', action="store_true", dest="redoPairs")
parser.add_argument('--negPairs', action="store", dest="negPairs", type=int, default=1)
parser.add_argument('--posPairs', action="store", dest="posPairs", type=int, default=1)
parser.add_argument('--noSigPairs', action="store_true", dest="noSigPairs")
parser.add_argument('--batchNorm', action="store_true", dest="batchNorm")
parser.add_argument('--noTSNE', action="store_true", dest="noTSNE")
parser.add_argument('--fineTune', action="store", dest="fineTune", type=int, default = 0)
parser.add_argument('--DNN', action="store_true", dest="DNN")
parser.add_argument('--EMD', action="store", dest="EMD", type=float, default=None)
parser.add_argument('--EMDNorm', action="store", dest="EMDNorm", type=float, default=None)
parser.add_argument('--patience', action="store", dest="patience", type=int, default=10)
parser.add_argument('--denseLayers', action="store", dest="denseLayers", type=int, nargs='+', default=[128,128,64,64])
parser.add_argument('--projLayers', action="store", dest="projLayers", type=int, nargs='+', default=[64,64])
parser.add_argument('--classLayers', action="store", dest="classLayers", type=int, nargs='+', default=[64,32,8])
parser.add_argument('--useSig', action="store_true", dest="useSig")
parser.add_argument('--masscut', action="store", dest="masscut", type=float, default=20.)
parser.add_argument('--ptcut', action="store", dest="ptcut", type=int, nargs='+', default=[500.,1500.])
parser.add_argument('--pairName', action="store", dest="pairName", type=str, default="")
parser.add_argument('--loss', choices=['sim', 'simclr', 'vicreg'], dest='loss', default='simclr')
parser.add_argument('--lossParams', action="store", dest="lossParams", type=float, nargs='+', default=None)
parser.add_argument('--trainFeat', action="store_true", dest="trainFeat")
parser.add_argument('--reloadFeat', action="store_true", dest="reloadFeat")
parser.add_argument('--trainClass', action="store_true", dest="trainClass")
parser.add_argument('--doAttention', action="store_true", dest="doAttention")
parser.add_argument('--noAugs', action="store_true", dest="noAugs")
parser.add_argument('--augData', action="store", dest="augData", type=float, default=None, nargs=4)
parser.add_argument('--augMC', action="store", dest="augMC", type=float, default=None, nargs=4)
parser.add_argument('--etaphiSmear', action="store", dest="etaphiSmear", type=float, default=[0.1,0.1], nargs=2)
parser.add_argument('--augs', action="store", dest="augs", type=str, default=['rotate','smear'], nargs='+')

### Parse arguments ###

args = parser.parse_args()


particlesConsidered = args.particles

numberOfEpochs = args.nEpochs
batchSize = args.batchSize
embeddingDim = args.projLayers[-1]
lossStr = ''
if args.loss=='simclr':
    lossStr = '_SimCLR'
elif args.loss=='vicreg':
    lossStr = '_VICReg'
modelName = "JetNet_Contrastive_Learning%s"%lossStr
if args.label != "":
    modelName = modelName + "_" + args.label 
modelNameAdd = modelName
if args.labeladd != "":
    modelNameAdd = modelNameAdd + "_" + args.labeladd
if args.fineTune:
    modelNameAdd = modelNameAdd + "_fineTune" + ("" if args.fineTune<0 else str(args.fineTune))

print(args)
if not os.path.isdir('/home/lsheldon/models/'+modelName):
    os.mkdir('/home/lsheldon/models/'+modelName)
    print("Directory created.")
if modelName != modelNameAdd and not os.path.isdir('/home/lsheldon/models/'+modelNameAdd):
    os.mkdir('/home/lsheldon/models/'+modelNameAdd)

np.random.seed(422022)
tf.random.set_random_seed(57)

### Extract data ###

print("Extracting")

fOne = h5py.File("/home/lsheldon/data/q.hdf5", 'r')
nom_bkg_data = fOne["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1]
nom_bkg_data = nom_bkg_data[:,:,[2, 0, 1]]
nom_bkg_jet_data = fOne["jet_features"][:].astype(np.float32)
nom_bkg_data = np.concatenate([nom_bkg_jet_data,np.reshape(np.transpose(nom_bkg_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((nom_bkg_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
print('Loaded nom_bkg')

fTwo = h5py.File("/home/lsheldon/data/w.hdf5", 'r')
nom_sig_data = fTwo["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
nom_sig_data = nom_sig_data[:,:,[2, 0, 1]]
nom_sig_jet_data = fTwo["jet_features"][:].astype(np.float32)
nom_sig_data = np.concatenate([nom_sig_jet_data,np.reshape(np.transpose(nom_sig_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((nom_sig_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
nom_sig_data[:,-1] = 1.
print('Loaded nom_sig')

fThree = h5py.File("/home/lsheldon/data/g.hdf5", 'r')
alt_bkg_data = fThree["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
alt_bkg_data = alt_bkg_data[:,:,[2, 0, 1]]
alt_bkg_jet_data = fThree["jet_features"][:].astype(np.float32)
alt_bkg_data = np.concatenate([alt_bkg_jet_data,np.reshape(np.transpose(alt_bkg_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((alt_bkg_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
print('Loaded alt_bkg')

fFour = h5py.File("/home/lsheldon/data/z.hdf5", 'r')
alt_sig_data = fFour["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
alt_sig_data = alt_sig_data[:,:,[2, 0, 1]]
alt_sig_jet_data = fFour["jet_features"][:].astype(np.float32)
alt_sig_data = np.concatenate([alt_sig_jet_data,np.reshape(np.transpose(alt_sig_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((alt_sig_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
alt_sig_data[:,-1] = 1.
print('Loaded alt_sig')

nom_data = np.concatenate([
    nom_bkg_data,
    nom_sig_data
    ] if args.useSig else [
    nom_bkg_data,
    ])

np.random.shuffle(nom_data)

alt_data = np.concatenate([
    alt_bkg_data,
    #alt_sig_data # can use signal but unrealistic to have signal data
    ])

np.random.shuffle(alt_data)

### Split data and reshape ###

nom_data = nom_data[nom_data[:,eventDataLength]>0.,:]
alt_data = alt_data[alt_data[:,eventDataLength]>0.,:]

nom_data = nom_data[nom_data[:,coldict["jet_sdmass"]]>args.masscut,:]
alt_data = alt_data[alt_data[:,coldict["jet_sdmass"]]>args.masscut,:]

nom_data = nom_data[nom_data[:,coldict["jet_pt"]]>args.ptcut[0],:]
alt_data = alt_data[alt_data[:,coldict["jet_pt"]]>args.ptcut[0],:]
nom_data = nom_data[nom_data[:,coldict["jet_pt"]]<args.ptcut[1],:]
alt_data = alt_data[alt_data[:,coldict["jet_pt"]]<args.ptcut[1],:]

nomTrainingDataLength = int(len(nom_data)*0.8)
nomValidationDataLength = int(len(nom_data)*0.1)
altTrainingDataLength = int(len(alt_data)*0.8)
altValidationDataLength = int(len(alt_data)*0.1)

nom_pts_train = nom_data[:nomTrainingDataLength,coldict["jet_pt"]]
alt_pts_train = alt_data[:altTrainingDataLength,coldict["jet_pt"]]
nom_pts_val = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,coldict["jet_pt"]]
alt_pts_val = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,coldict["jet_pt"]]
nom_pts_test = nom_data[nomTrainingDataLength + nomValidationDataLength:,coldict["jet_pt"]]
alt_pts_test = alt_data[altTrainingDataLength + altValidationDataLength:,coldict["jet_pt"]]

nom_msd_train = nom_data[:nomTrainingDataLength,coldict["jet_sdmass"]]
alt_msd_train = alt_data[:altTrainingDataLength,coldict["jet_sdmass"]]
nom_msd_val = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,coldict["jet_sdmass"]]
alt_msd_val = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,coldict["jet_sdmass"]]
nom_msd_test = nom_data[nomTrainingDataLength + nomValidationDataLength:,coldict["jet_sdmass"]]
alt_msd_test = alt_data[altTrainingDataLength + altValidationDataLength:,coldict["jet_sdmass"]]

nom_rhos_train = np.clip(2.*np.log(nom_msd_train/nom_pts_train), -10., -1.)
alt_rhos_train = np.clip(2.*np.log(alt_msd_train/alt_pts_train), -10., -1.)
nom_rhos_val = np.clip(2.*np.log(nom_msd_val/nom_pts_val), -10., -1.)
alt_rhos_val = np.clip(2.*np.log(alt_msd_val/alt_pts_val), -10., -1.)
nom_rhos_test = np.clip(2.*np.log(nom_msd_test/nom_pts_test), -10., -1.)
alt_rhos_test = np.clip(2.*np.log(alt_msd_test/alt_pts_test), -10., -1.)

nom_train = nom_data[:nomTrainingDataLength,eventDataLength:decayTypeColumn]
alt_train = alt_data[:altTrainingDataLength,eventDataLength:decayTypeColumn]
nom_val = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,eventDataLength:decayTypeColumn]
alt_val = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,eventDataLength:decayTypeColumn]
nom_test = nom_data[nomTrainingDataLength + nomValidationDataLength:,eventDataLength:decayTypeColumn]
alt_test = alt_data[altTrainingDataLength + altValidationDataLength:,eventDataLength:decayTypeColumn]
print(nom_test[0])
nom_train_pt = nom_data[:nomTrainingDataLength,coldict["jet_pt"]]
alt_train_pt = alt_data[:altTrainingDataLength,coldict["jet_pt"]]
nom_val_pt = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,coldict["jet_pt"]]
alt_val_pt = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,coldict["jet_pt"]]
nom_test_pt = nom_data[nomTrainingDataLength + nomValidationDataLength:,coldict["jet_pt"]]
alt_test_pt = alt_data[altTrainingDataLength + altValidationDataLength:,coldict["jet_pt"]]

nom_train_true = nom_data[:nomTrainingDataLength,decayTypeColumn:]
alt_train_true = alt_data[:altTrainingDataLength,decayTypeColumn:]
nom_val_true = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,decayTypeColumn:]
alt_val_true = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,decayTypeColumn:]
nom_test_true = nom_data[nomTrainingDataLength + nomValidationDataLength:,decayTypeColumn:]
alt_test_true = alt_data[altTrainingDataLength + altValidationDataLength:,decayTypeColumn:]

    
nom_test_vars = np.transpose(nom_test.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
alt_test_vars = np.transpose(alt_test.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
nom_train_vars = np.transpose(nom_train.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
alt_train_vars = np.transpose(alt_train.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
nom_val_vars = np.transpose(nom_val.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
alt_val_vars = np.transpose(alt_val.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))

emd_debug_norm = False
emd_debug = False
def rotate_via_numpy(xy, radians):
    """Use numpy to build a rotation matrix and take the dot product."""
    x = xy[...,0]
    y = xy[...,1]
    c, s = np.cos(radians), np.sin(radians)
    if hasattr(radians, "__len__"):
        s = s[:,np.newaxis]
        c = c[:,np.newaxis]
        m = np.array([x*c-y*s, x*s+y*c])
    else:
        j = np.matrix([[c, s], [-s, c]])
        m = np.dot(j, [x, y])  #Nx2x2 , 2xN

    return np.moveaxis(m, 0, -1)
if emd_debug_norm:
    import matplotlib.pyplot as plt
    bins=np.linspace(0.,1.,51)
    emds = wasserstein.PairwiseEMD(dtype=np.float32,print_every=-100)
    for ipart in range(alt_test_vars.shape[1]-1,-1,-1):
        alt_test_vars[:,ipart,1:-1] = rotate_via_numpy(alt_test_vars[:,ipart,1:-1],-np.arctan2(alt_test_vars[:,1,2],alt_test_vars[:,1,1]))
    emds(nom_test_vars[nom_test_true[:,0]==0,:,:-1][:1000], eventsB=alt_test_vars[alt_test_true[:,0]==0,:,:-1][:1000])
    plt.clf()
    emdarr = emds.emds()
    print(emdarr[:5,:5])
    plt.hist(emdarr[np.triu(emdarr)>0], bins=bins, histtype='step', label="QCD-QCD", density=True)
    emds.clear()
    emds(nom_test_vars[nom_test_true[:,0]==1,:,:-1][:1000], eventsB=alt_test_vars[alt_test_true[:,0]==0,:,:-1][:1000])
    emdarr = emds.emds()
    print(emdarr[:5,:5])
    plt.hist(emdarr[np.triu(emdarr)>0], bins=bins, histtype='step', label="QCD-Z'", density=True)
    plt.legend()

    plt.xlabel('EMD (normalized)')
    plt.ylabel('Probability Density')
    plt.savefig('/home/lsheldon/plots/emds_norm.pdf')
    plt.yscale('log')
    plt.savefig('/home/lsheldon/plots/emds_norm_log.pdf')

if emd_debug:
    import matplotlib.pyplot as plt
    bins=np.linspace(0.,500.,51)
    emds = wasserstein.PairwiseEMD(dtype=np.float32,print_every=-100)
    for i in range(particlesConsidered):
        nom_test_vars[:,i,0] = nom_test_vars[:,i,0]*nom_test_pt
        alt_test_vars[:,i,0] = alt_test_vars[:,i,0]*alt_test_pt
    if not emd_debug_norm:
        for ipart in range(alt_test_vars.shape[1]-1,-1,-1):
            alt_test_vars[:,ipart,1:-1] = rotate_via_numpy(alt_test_vars[:,ipart,1:-1],-np.arctan2(alt_test_vars[:,0,2],alt_test_vars[:,0,1]))
    emds(nom_test_vars[nom_test_true[:,0]==0,:,:-1][:1000], eventsB=alt_test_vars[alt_test_true[:,0]==0,:,:-1][:1000])
    plt.clf()
    emdarr = emds.emds()
    print(emdarr[:5,:5])
    plt.hist(emdarr[np.triu(emdarr)>0], bins=bins, histtype='step', label="QCD-QCD", density=True)
    emds.clear()
    emds(nom_test_vars[nom_test_true[:,0]==1,:,:-1][:1000], eventsB=alt_test_vars[alt_test_true[:,0]==0,:,:-1][:1000])
    emdarr = emds.emds()
    print(emdarr[:5,:5])
    plt.hist(emdarr[np.triu(emdarr)>0], bins=bins, histtype='step', label="QCD-Z'", density=True)
    plt.legend()

    plt.xlabel('EMD (unnormalized)')
    plt.ylabel('Probability Density')
    plt.savefig('/home/lsheldon/plots/emds.pdf')
    plt.yscale('log')
    plt.savefig('/home/lsheldon/plots/emds_log.pdf')

if emd_debug or emd_debug_norm:
    exit()

### Make pairs from data ###

def make_pairs_emd(nom, alt, resample=10):
    emds = wasserstein.PairwiseEMD(dtype=np.float32,print_every=-10,request_mode=True)
    pairbatch = batchSize
    pairs = []
    bool_labels = []
    last_perc = -1
    restart = True
    ialt = 0
    for i in range(0,nom.shape[0],pairbatch):
        ialt = ialt+pairbatch
        if restart:
            ialt = 0
            restart = False
        percent = 100*i/nom.shape[0]
        current_perc = round(percent, 1)
        if last_perc != current_perc:
            print(f"{current_perc}%")
            last_perc = current_perc
        emds(nom[i:i+pairbatch], eventsB=alt[ialt:ialt+pairbatch])
        # print(i,emds.nevA(),emds.nevB())
        #emd_array = emds.emds()
        for xi in range(pairbatch):
            if i+xi>=nom.shape[0]: 
                break
            for yi in range(pairbatch):
                if ialt+yi>=alt.shape[0]: 
                    restart = True
                    break
                for ri in range(resample+1):
                    if xi-yi==ri:
                        pairs.append([i+xi,ialt+yi])
                        bool_labels.append([emds.emd(xi,yi)])
        emds.clear()
    return (np.asarray(pairs), np.asarray(bool_labels))

def make_pairs(nom, alt, nbins, numpos = 1, numneg = 1):
    total = np.transpose(np.concatenate([nom,alt]))
    bins = np.linspace(np.min(total), np.max(total)+1, num=nbins)
    nom_ids = np.digitize(nom, bins=bins)
    alt_ids = np.digitize(alt, bins=bins)
    pos_alt_ids = [np.where(alt_ids == i)[0] for i in range(1, nbins)]
    pairs = []
    bool_labels = []
    last_perc = -1
    for i in range(len(nom_ids)):
        percent = 100*i/len(nom_ids)
        current_perc = round(percent, 1)
        if last_perc != current_perc:
            print(f"{current_perc}%")
            last_perc = current_perc
        nom_id = nom_ids[i]-1
        for j in range(numpos):
            pos_alt_i = np.random.choice(pos_alt_ids[nom_id])
            pairs.append([i,pos_alt_i])
            bool_labels.append([1])
            neg_alt_ids = np.where(np.abs(alt_ids-nom_id)>4)[0]
            neg_alt_i = np.random.choice(neg_alt_ids, size=numneg)
            pairs.extend([[i, neg_alt_i[ni]] for ni in range(numneg)])
            bool_labels.extend([[0] for ni in range(numneg)])
    return (np.asarray(pairs), np.asarray(bool_labels))

nom_sig_data = nom_sig_data[nom_sig_data[:,eventDataLength]>0.,eventDataLength:decayTypeColumn]
nom_bkg_data = nom_bkg_data[nom_bkg_data[:,eventDataLength]>0.,eventDataLength:decayTypeColumn]
alt_bkg_data = alt_bkg_data[alt_bkg_data[:,eventDataLength]>0.,eventDataLength:decayTypeColumn]

nom_sig_data = np.transpose(nom_sig_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
nom_bkg_data = np.transpose(nom_bkg_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
alt_bkg_data = np.transpose(alt_bkg_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))

def make_pairs_id(ns, nb, ab, length):
    if length == None: 
        length = min(ab.shape[0],nb.shape[0],ns.shape[0])
    pos_pairs = np.empty([3,length,particlesConsidered,entriesPerParticle])
    neg_pairs = np.empty([3,length,particlesConsidered,entriesPerParticle])
    ns_full = np.repeat(ns,math.ceil(length/ns.shape[0]),axis=0)
    nb_full = np.repeat(nb,math.ceil(length/nb.shape[0]),axis=0)
    ab_full = np.repeat(ab,math.ceil(length/ab.shape[0]),axis=0)
    np.random.shuffle(ns_full)
    np.random.shuffle(nb_full)
    np.random.shuffle(ab_full)
    
    ns_full = ns_full[:length]
    nb_full = nb_full[:length]
    ab_full = ab_full[:length]
    
    pos_pairs[0] = nb_full
    pos_pairs[1] = ab_full
    pos_pairs[2] = np.ones([length,particlesConsidered,entriesPerParticle]) 
    neg_pairs[0] = ns_full
    neg_pairs[1] = ab_full
    neg_pairs[2] = np.zeros([length,particlesConsidered,entriesPerParticle])

    pairs = np.concatenate([pos_pairs,neg_pairs], axis=1)
    pairs=np.transpose(pairs, (1,0,2,3))
    np.random.shuffle(pairs)
    pairs=np.transpose(pairs, (1,0,2,3))
    bool_labels = np.array(pairs[2,:,0,0])
    pairs = pairs[:-1]
    
    return (pairs, bool_labels)

if args.redoPairs:
    if args.EMD is not None or args.EMDNorm is not None:
        if args.EMD is not None:
            for i in range(particlesConsidered):
                nom_train_vars[:,i,0] = nom_train_vars[:,i,0]*nom_train_pt
                alt_train_vars[:,i,0] = alt_train_vars[:,i,0]*alt_train_pt
                nom_val_vars[:,i,0] = nom_val_vars[:,i,0]*nom_val_pt
                alt_val_vars[:,i,0] = alt_val_vars[:,i,0]*alt_val_pt
                nom_test_vars[:,i,0] = nom_test_vars[:,i,0]*nom_test_pt
                alt_test_vars[:,i,0] = alt_test_vars[:,i,0]*alt_test_pt
                print(nom_train_vars[0,0])
        train_pair_indices, train_labels = make_pairs_emd(nom_train_vars, alt_train_vars)
        val_pair_indices, val_labels = make_pairs_emd(nom_val_vars, alt_val_vars)
        test_pair_indices, test_labels = make_pairs_emd(nom_test_vars, alt_test_vars)
        print(train_pair_indices[0,0], train_labels[0,0])
        
    else:
        train_pair_indices, train_labels = make_pairs(nom_rhos_train, alt_rhos_train, particlesConsidered, args.posPairs, args.negPairs)
        val_pair_indices, val_labels = make_pairs(nom_rhos_val, alt_rhos_val, particlesConsidered, args.posPairs, args.negPairs)
        test_pair_indices, test_labels = make_pairs(nom_rhos_test, alt_rhos_test, particlesConsidered, args.posPairs, args.negPairs)

    np.save("/home/lsheldon/pairs/trainingpairs%s.npy"%args.pairName, train_pair_indices, allow_pickle=True)
    np.save("/home/lsheldon/pairs/traininglabels%s.npy"%args.pairName, train_labels, allow_pickle=True)
    np.save("/home/lsheldon/pairs/valpairs%s.npy"%args.pairName, val_pair_indices, allow_pickle=True)
    np.save("/home/lsheldon/pairs/vallabels%s.npy"%args.pairName, val_labels, allow_pickle=True)
    np.save("/home/lsheldon/pairs/testpairs%s.npy"%args.pairName, test_pair_indices, allow_pickle=True)
    np.save("/home/lsheldon/pairs/testlabels%s.npy"%args.pairName, test_labels, allow_pickle=True)
    
    if not args.trainClass or not args.trainFeat:
        exit()

else:
    test_labels = np.load('/home/lsheldon/pairs/testlabels%s.npy'%args.pairName, allow_pickle=True)
    test_pair_indices = np.load('/home/lsheldon/pairs/testpairs%s.npy'%args.pairName, allow_pickle=True)
    train_labels = np.load('/home/lsheldon/pairs/traininglabels%s.npy'%args.pairName, allow_pickle=True)
    train_pair_indices = np.load('/home/lsheldon/pairs/trainingpairs%s.npy'%args.pairName, allow_pickle=True)
    val_labels = np.load('/home/lsheldon/pairs/vallabels%s.npy'%args.pairName, allow_pickle=True)
    val_pair_indices = np.load('/home/lsheldon/pairs/valpairs%s.npy'%args.pairName, allow_pickle=True)
    print('true '+str(train_pair_indices[0,0]))

    if args.EMD is not None:
        train_labels = (train_labels < args.EMD).astype(int)
        val_labels = (val_labels < args.EMD).astype(int)
        test_labels = (test_labels < args.EMD).astype(int)
    elif args.EMDNorm is not None:
        train_labels = (train_labels < args.EMDNorm).astype(int)
        val_labels = (val_labels < args.EMDNorm).astype(int)
        test_labels = (test_labels < args.EMDNorm).astype(int)

if (args.loss=='simclr') or (args.loss=='vicreg'):
    print('labels',train_labels.shape)
    print('indices',train_pair_indices.shape)
    print('nom_true',nom_train_true.shape)
    print(train_labels[0,0], train_pair_indices[0,0])
    if args.loss=='vicreg' and args.noSigPairs: 
        train_labels = train_labels[nom_train_true[train_pair_indices[:,0],0]!=1,:]
        val_labels = val_labels[nom_val_true[val_pair_indices[:,0],0]!=1,:]
        test_labels = test_labels[nom_test_true[test_pair_indices[:,0],0]!=1,:]
        train_pair_indices = train_pair_indices[nom_train_true[train_pair_indices[:,0],0]!=1,:]
        val_pair_indices = val_pair_indices[nom_val_true[val_pair_indices[:,0],0]!=1,:]
        test_pair_indices = test_pair_indices[nom_test_true[test_pair_indices[:,0],0]!=1,:]

    print(train_labels[:,0])
    train_pair_indices = train_pair_indices[train_labels[:,0]!=0,:]
    val_pair_indices = val_pair_indices[val_labels[:,0]!=0,:]
    test_pair_indices = test_pair_indices[test_labels[:,0]!=0,:]
    train_labels = train_labels[train_labels[:,0]!=0,:]
    val_labels = val_labels[val_labels[:,0]!=0,:]
    test_labels = test_labels[test_labels[:,0]!=0,:]
    def x_to_n(x,n):
        return int(n - (n%x))
    train_cap = x_to_n(batchSize,train_labels.shape[0])+1
    train_pair_indices = train_pair_indices[:train_cap]
    train_labels = train_labels[:train_cap]
    val_cap = x_to_n(batchSize,val_labels.shape[0])+1
    val_pair_indices = val_pair_indices[:val_cap]
    val_labels = val_labels[:val_cap]
    test_cap = x_to_n(batchSize,test_labels.shape[0])+1
    test_pair_indices = test_pair_indices[:test_cap]
    test_labels = test_labels[:test_cap]

### Reshape pairs to fit model ###
nom_test_data = np.zeros((test_pair_indices.shape[0],particlesTotal*entriesPerParticle))
alt_test_data = np.zeros((test_pair_indices.shape[0],particlesTotal*entriesPerParticle))
nom_train_data = np.zeros((train_pair_indices.shape[0],particlesTotal*entriesPerParticle))
alt_train_data = np.zeros((train_pair_indices.shape[0],particlesTotal*entriesPerParticle))
nom_val_data = np.zeros((val_pair_indices.shape[0],particlesTotal*entriesPerParticle))
alt_val_data = np.zeros((val_pair_indices.shape[0],particlesTotal*entriesPerParticle))

nom_test_pts = np.zeros(test_pair_indices.shape[0])
alt_test_pts = np.zeros(test_pair_indices.shape[0])
nom_train_pts = np.zeros(train_pair_indices.shape[0])
alt_train_pts = np.zeros(train_pair_indices.shape[0])
nom_val_pts = np.zeros(val_pair_indices.shape[0])
alt_val_pts = np.zeros(val_pair_indices.shape[0])

for i in range(test_pair_indices.shape[0]):
        nom_test_data[i] = nom_test[int(test_pair_indices[i,0])]
        alt_test_data[i] = alt_test[int(test_pair_indices[i,1])]
        nom_test_pts[i] = nom_test_pt[int(test_pair_indices[i,0])]
        alt_test_pts[i] = alt_test_pt[int(test_pair_indices[i,1])]
for i in range(train_pair_indices.shape[0]):
        nom_train_data[i] = nom_train[int(train_pair_indices[i,0])]
        alt_train_data[i] = alt_train[int(train_pair_indices[i,1])]
        nom_train_pts[i] = nom_train_pt[int(train_pair_indices[i,0])]
        alt_train_pts[i] = alt_train_pt[int(train_pair_indices[i,1])]
for i in range(val_pair_indices.shape[0]):
        nom_val_data[i] = nom_val[int(val_pair_indices[i,0])]
        alt_val_data[i] = alt_val[int(val_pair_indices[i,1])]
        nom_val_pts[i] = nom_val_pt[int(val_pair_indices[i,0])]
        alt_val_pts[i] = alt_val_pt[int(val_pair_indices[i,1])]

if args.noSigPairs and args.loss=='simclr':
    train_labels = train_labels*(nom_train_true[train_pair_indices[:,0]]!=1).astype(int)
    val_labels = val_labels*(nom_val_true[val_pair_indices[:,0]]!=1).astype(int)
    test_labels = test_labels*(nom_test_true[test_pair_indices[:,0]]!=1).astype(int)

print(nom_test_pts.shape,test_labels.shape,nom_test_data.shape)

print('pos',np.histogram(nom_test_pts[test_labels[:,0]==1]-alt_test_pts[test_labels[:,0]==1]))
print('neg',np.histogram(nom_test_pts[test_labels[:,0]==0]-alt_test_pts[test_labels[:,0]==0]))

#print(nom_test_data[0,:])
    
reshaped_nom_test_data = np.transpose(nom_test_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
reshaped_alt_test_data = np.transpose(alt_test_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
reshaped_nom_train_data = np.transpose(nom_train_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
reshaped_alt_train_data = np.transpose(alt_train_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
reshaped_nom_val_data = np.transpose(nom_val_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
reshaped_alt_val_data = np.transpose(alt_val_data.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))

print(reshaped_nom_train_data[0], reshaped_alt_train_data[0])
print('frac_train_avg',(np.mean(reshaped_nom_train_data,axis=0)-np.mean(reshaped_alt_train_data,axis=0))/np.mean(reshaped_nom_train_data,axis=0))
print('frac_val_avg',(np.mean(reshaped_nom_val_data,axis=0)-np.mean(reshaped_alt_val_data,axis=0))/np.mean(reshaped_nom_val_data,axis=0))
print('frac_train_avg',(np.mean(reshaped_nom_train_data,axis=0)-np.mean(reshaped_alt_train_data,axis=0))/np.mean(reshaped_nom_train_data,axis=0))

testdata = np.stack([reshaped_nom_test_data, reshaped_alt_test_data], axis = 0)
traindata = np.stack([reshaped_nom_train_data, reshaped_alt_train_data], axis = 0)
valdata = np.stack([reshaped_nom_val_data, reshaped_alt_val_data], axis = 0)



print(traindata[0])

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

### Helper functions ###

## Model Building Helpers ##

# Particle data interaction NN
def build_siamese_model_IN(denseDims=args.denseLayers, projDims=args.projLayers, doAttention=args.doAttention):
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

    if doAttention:
        NHEADS = 8
        ATTDIM = 32
        ATTOUT = 32
        EppQ = Reshape((particlesConsidered, particlesConsidered-1, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppQueryProj")(Epp))
        EppK = Reshape((particlesConsidered, particlesConsidered-1, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppKeyProj")(Epp))
        EppV = Reshape((particlesConsidered, particlesConsidered-1, NHEADS, -1))(Conv1D(ATTDIM*NHEADS, kernel_size=1, name="EppValueProj")(Epp))
        EppMHAProjSoft = Lambda(lambda listOfTensors: K.softmax((tf.matmul(tf.transpose(listOfTensors[0], perm=(0, 1, 3, 2, 4)), tf.transpose(listOfTensors[1], perm=(0, 1, 3, 2, 4)), transpose_b=True) / tf.math.sqrt(tf.cast(particlesConsidered-1, listOfTensors[0].dtype)))), name="EppMHAProjSoft")([EppQ, EppK])
        EppMHAProj = Lambda(lambda listOfTensors: K.sum(tf.transpose(tf.matmul(tf.multiply(listOfTensors[0], 
            tf.reshape(tf.matmul(tf.reshape(tf.repeat(tf.cast(listOfTensors[2][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), [-1, particlesConsidered, particlesConsidered-1, 1]), tf.reshape(tf.repeat(tf.cast(listOfTensors[2][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), [-1, particlesConsidered, particlesConsidered-1, 1]), transpose_b=True), [-1, particlesConsidered, 1, particlesConsidered-1, particlesConsidered-1])
            ), tf.transpose(listOfTensors[1], perm=(0, 1, 3, 2, 4))), perm=(0, 1, 4, 2, 3)), axis=-1), name="EppMHAProj")([EppMHAProjSoft, EppV, inputParticle]) #sum over edges (keep pT agnostic)
        EppBar = Conv1D(ATTOUT, kernel_size=1, name="EppBar")(Reshape((particlesConsidered, -1))(EppMHAProj))
    else:
        EppBar = Lambda(lambda listOfTensors: tf.transpose(tf.matmul(tf.transpose(listOfTensors[0], perm=(0, 2, 1)), tf.multiply(tf.expand_dims(tf.repeat(tf.cast(listOfTensors[1][:,1:,0]>0., listOfTensors[0].dtype), particlesConsidered, axis=1), axis=-1), np.expand_dims(RRT, axis=0))),
                                                perm=(0, 2, 1)), name="EppBar")([Epp, inputParticle])
    C = Lambda(lambda listOfTensors: tf.concat((listOfTensors[0], listOfTensors[1]), axis=2), name="C")(
        [inputParticle, EppBar])

    convPredictOne = Conv1D(64, kernel_size=1, activation="relu", name="convPredictOne")(C)
    convPredictTwo = Conv1D(32, kernel_size=1, activation="relu", name="convPredictTwo")(convPredictOne)

    O = Conv1D(32, kernel_size=1, activation="relu", name="O")(convPredictTwo)

    # Calculate output
    OBar = Lambda(lambda listOfTensors: K.sum(tf.multiply(listOfTensors[0], tf.expand_dims(tf.cast(listOfTensors[1][:,:,0]>0., listOfTensors[0].dtype), axis=-1)), axis=1), name="OBar")([O, inputParticle])
    #OBar = Lambda(lambda tensor: K.sum(tensor, axis=1), name="OBar")(O)

    denseEnd = Dense(denseDims[0], activation="relu", name="denseEnd0", kernel_regularizer=l2(args.l2reg))(OBar)
    for il in range(1,len(denseDims)):
        if(args.batchNorm): denseEnd = BatchNormalization(momentum=0.6, name="batchnormEnd%i"%il)(denseEnd)
        denseEnd = Dense(denseDims[il], activation="relu" if il<len(denseDims)-1 else "linear", name="denseEnd%i"%il, kernel_regularizer=l2(args.l2reg))(denseEnd)

    print("Compiling")

    model = Model(inputs=[inputParticle], outputs=[denseEnd])
        
    inputProjection = Input(shape=(denseDims[-1],), name="inputProjection")
    for il in range(0,len(projDims)):
        denseProj = Dense(projDims[il], activation="relu" if il<len(projDims)-1 else "linear", name="denseProj%i"%il, kernel_regularizer=l2(args.l2reg))(denseProj if il>0 else inputProjection)
        if (args.batchNorm): denseProj = BatchNormalization(momentum=0.6, name="batchnormProj%i"%il)(denseProj)
    projmodel = Model(inputs=[inputProjection],outputs=[denseProj])
    
    return model,projmodel

# Particle DNN
def build_siamese_model_DNN(denseDims=args.denseLayers, projDims=args.projLayers):
    inputParticle = Input(shape=(particlesConsidered, entriesPerParticle), name="inputParticle")

    flattened = Flatten(name="flatten0")(inputParticle)

    denseEnd = Dense(denseDims[0], activation="relu", name="denseEnd0", kernel_regularizer=l2(args.l2reg))(flattened)
    for il in range(1,len(denseDims)-1):
        if(args.batchNorm): denseEnd = BatchNormalization(momentum=0.6, name="batchnormEnd%i"%il)(denseEnd)
        denseEnd = Dense(denseDims[il], activation="relu", name="denseEnd%i"%il, kernel_regularizer=l2(args.l2reg))(denseEnd)
    output = Dense(denseDims[-1], activation="linear", name="output")(denseEnd)

    print("Compiling")

    model = Model(inputs=[inputParticle], outputs=[output])
        
    inputProjection = Input(shape=(denseDims[-1],), name="inputProjection")
    denseProj = Dense(projDims[0], activation="relu", name="denseProj0", kernel_regularizer=l2(args.l2reg))(inputProjection)
    for il in range(1,len(projDims)-1):
        if(args.batchNorm): denseProj = BatchNormalization(momentum=0.6, name="batchnormProj%i"%il)(denseProj)
        denseProj = Dense(projDims[il], activation="relu", name="denseProj%i"%il, kernel_regularizer=l2(args.l2reg))(denseProj)
    denseProj = Dense(projDims[-1], activation="linear", name="denseProjOut")(denseProj)
    projmodel = Model(inputs=[inputProjection],outputs=[denseProj])
    
    return model,projmodel

## Loss Functions and other Helpers ##

def similarity(vectors):
    (m_mc, m_data) = vectors
    m_mc = K.l2_normalize(m_mc, axis=-1)
    m_data = K.l2_normalize(m_data, axis=-1)
    print('m_mc',m_mc.shape)
    return K.sum(m_mc*m_data, axis=-1, keepdims=True)
    #dot = K.sum(m_mc * m_data, axis=1, keepdims=True)
    #mc_norm = K.sqrt(K.sum(m_mc*m_mc, axis=1, keepdims=True))
    #data_norm = K.sqrt(K.sum(m_data*m_data, axis=1, keepdims=True))
    #norm = mc_norm * data_norm
    #return dot/norm

def simCLR(vectors): #expects only pos pairs
    temp = 0.1
    if args.lossParams is not None:
        temp=args.lossParams[0]
    (m_mc, m_data) = vectors
    m_mc = K.l2_normalize(m_mc, axis=-1)
    m_data = K.l2_normalize(m_data, axis=-1)
    print('m_mc',m_mc) #[?, 32]
    posexp = K.exp(K.sum(m_mc*m_data, axis=-1, keepdims=True)/temp) #[?,1]
    stack = tf.concat([m_mc, m_data], axis=0)#[2?,32]
    pairs = tf.repeat(tf.expand_dims(m_mc, axis=-1),2*batchSize,axis=-1) #[?,32,2?]
    negs = K.sum(pairs*tf.expand_dims(tf.transpose(stack), axis=0), axis=1) #[?,2?]
    negexp = K.exp(negs/temp)
    negdiag = tf.expand_dims(tf.linalg.diag_part(negexp), axis=-1)
    negsum = (K.sum(negexp, axis=-1, keepdims=True)-negdiag)/temp #[?,1]
    logval = -1.*K.log(posexp/negsum)
    print('logval',logval)
    return logval

def contrastive_loss_sim(y, preds):
        y = tf.cast(y, preds.dtype)
        loss = K.mean(y*(1.0-preds)+(1.0-y)*(1.0+preds))
        return loss

def contrastive_loss_siamese_simclr(y, preds):
        temp = 0.1
        if args.lossParams is not None:
            temp=args.lossParams[0]
        y = tf.cast(y, preds.dtype)
        exp = K.exp(preds/temp)
        negsum = K.sum( K.squeeze((1.0-y) * exp, axis=1))+K.epsilon()
        log = K.log((exp+K.epsilon())/negsum)
        loss = K.sum(K.squeeze(-y*log,axis=1))
        return loss

def contrastive_loss_simclr(y, preds):
        loss = K.sum(K.squeeze(preds,axis=1))
        return loss

def vicreg_covariance(z):
    zbar = K.mean(z,axis=0)
    CZ2 = K.square(K.sum(tf.matmul(tf.expand_dims(z-zbar,axis=-1),tf.expand_dims(z-zbar, axis=1)), axis=0)/(float(batchSize)-1.))
    print('CZ2',CZ2)
    return (K.sum(K.sum(CZ2,axis=1),axis=0)-tf.linalg.trace(CZ2))/float(embeddingDim)

def vicreg_variance(z, gamma=1.):
    print(gamma,z,z.shape)
    return K.mean(K.maximum(0.,gamma-K.sqrt(K.var(z,axis=0)+K.epsilon())),axis=0)

def vicregSVC(vectors, truths = None): #expects only pos pairs
    sweight=25.
    vweight=25.
    cweight=1.
    if args.lossParams is not None:
        sweight=args.lossParams[0]
        vweight=args.lossParams[1]
        cweight=args.lossParams[2]
    (featsA, featsB) = vectors
    s_term = K.sum(K.square(featsA - featsB)*(truths if truths is not None else 1), axis=1, keepdims=True)
    v_term = (vicreg_variance(featsA) + vicreg_variance(featsB))
    c_term = (vicreg_covariance(featsA) + vicreg_covariance(featsB))
    print('c_term',c_term)
    return s_term*sweight+v_term*vweight+c_term*cweight

def EMDmin(inputsA, inputsB):
    emds = wasserstein.PairwiseEMD(dtype=np.float32,print_every=0)
    emds(inputsA[:,:,:-1], eventsB=inputsB[:,:,:-1])
    print('emd',emds.emds().shape)
    emdmin = np.argmin(emds.emds(),axis=1)
    print('emdmin',emdmin.shape,emdmin)
    return emdmin

def vicregSVC_live(vectors): #expects only pos pairs
    sweight=25.
    vweight=25.
    cweight=1.
    if args.lossParams is not None:
        sweight=args.lossParams[0]
        vweight=args.lossParams[1]
        cweight=args.lossParams[2]
    (featsA, featsB, inputsA, inputsB) = vectors
    emdmin = tf.py_function(func=EMDmin, inp=[inputsA, inputsB], Tout=tf.int64)
    print('featsB',featsB.shape)
    featsB = featsB[emdmin]
    print('e',emdmin,emdmin.shape)
    print('featsBpost',featsB.shape)
    s_term = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    v_term = (vicreg_variance(featsA) + vicreg_variance(featsB,1.1))
    c_term = (vicreg_covariance(featsA) + vicreg_covariance(featsB))
    print('c_term',c_term)
    return s_term*sweight+v_term*vweight+c_term*cweight

def contrastive_loss_vicreg(y, preds):
    return K.sum(K.squeeze(preds,axis=1))/float(batchSize)

def contrastive_loss(y, preds):
    print('loss',args.loss)
    if args.loss=='sim':
        return contrastive_loss_sim(y, preds)
    elif args.loss=='simclr':
        return contrastive_loss_simclr(y, preds)
    elif args.loss=='vicreg':
        return contrastive_loss_vicreg(y, preds)
    else:
        print('UNKNOWN LOSS...')
        exit()

def lr_scheduler(epoch, lr):
    initial_learning_rate = args.LR
    warmup_learning_rate = args.warmLR if args.warmLR>0. else initial_learning_rate
    final_learning_rate = args.fLR if args.fLR>0. else initial_learning_rate
    startepoch = args.decayEpochs[0]
    totalepochs = float(args.decayEpochs[1]-args.decayEpochs[0])
    if epoch<totalepochs+startepoch and epoch>startepoch:
        lr = final_learning_rate + 0.5*(initial_learning_rate - final_learning_rate)*(1.+np.cos(np.pi*float(epoch-startepoch)/totalepochs))
    elif epoch<=startepoch:
        lr = warmup_learning_rate + float(epoch)*(initial_learning_rate-warmup_learning_rate)/float(startepoch)
    else:
        lr = final_learning_rate
    return float(lr)

if args.trainFeat:
    # configure the siamese network
    print("[INFO] building siamese network...")
    nominput = Input(shape=(particlesConsidered, entriesPerParticle), name='nominput')
    altinput = Input(shape=(particlesConsidered, entriesPerParticle), name='altinput')
    pairTruths = Input(shape=(1,), name='pairTruths')
    featureExtractor, projector = build_siamese_model_DNN() if args.DNN else build_siamese_model_IN()
    nomfeats = featureExtractor(nominput)
    altfeats = featureExtractor(altinput)
    
    nomproj = projector(nomfeats)
    altproj = projector(altfeats)
    if args.loss=='vicreg':
        distance = Lambda(vicregSVC)([nomproj, altproj])
    elif args.loss=='simclr':
        distance = Lambda(simCLR)([nomproj, altproj])
    else:
        distance = Lambda(similarity)([nomproj, altproj])
    model = Model(inputs=[nominput, altinput], outputs=distance)
    
    print('model',model.summary())
    print('features',featureExtractor.summary())
    print('projector',projector.summary())
    
    modelCallbacks = [ModelCheckpoint(filepath="/home/lsheldon/models/" + modelName + "/feat.h5", save_weights_only=True, monitor="val_loss" if args.patience>=0 else "loss",
                                      save_best_only=True),
                      ]
    if args.patience>=0:
        modelCallbacks.append(EarlyStopping(monitor="val_loss",patience=args.patience))
    if args.LARS:
        model.compile(optimizer=LARS(args.LR,weight_decay=0.000001), loss=contrastive_loss, metrics=['acc'])
    elif args.Adam:
        model.compile(optimizer='adam', loss=contrastive_loss, metrics=['acc'])
    else:
        modelCallbacks.append(LearningRateScheduler(lr_scheduler, verbose=0))
        model.compile(optimizer='sgd', loss=contrastive_loss, metrics=['acc'])

    def rotate_jet(jet): #expects [batch X parts X feats]
        jet[:,:,1:] = rotate_via_numpy(jet[:,:,1:],np.random.uniform(-np.pi,np.pi,jet[:,0,0].shape))
        return jet
    def smear_eta_phi(jet, jet_pt, scale_eta=args.etaphiSmear[0], scale_phi=args.etaphiSmear[1]): #expects [batch X parts X feats]
        sigma = np.sqrt(np.divide(1.,jet[:,:,0]*jet_pt[:, np.newaxis],out=np.zeros_like(jet[:,:,0]), where=jet[:,:,0]!=0))
        print('sigma',scale_eta*sigma[0])
        jet[:,:,1] = jet[:,:,1]+np.random.normal(0.,1.,size=jet[:,:,1].shape)*sigma*scale_eta
        jet[:,:,2] = jet[:,:,2]+np.random.normal(0.,1.,size=jet[:,:,2].shape)*sigma*scale_phi
        return jet
    
    def align_pts(mc,data):
        nom_pts = mc[:,:,0]
        alt_pts = data[:,:,0]
        nom_pt_mean = np.mean(nom_pts)
        alt_pt_mean = np.mean(alt_pts)
        alt_pts = alt_pts+nom_pt_mean-alt_pt_mean
        data[:,:,0]=alt_pts
        return data

    if args.reloadFeat:
        model.load_weights("/home/lsheldon/models/" + modelName + "/feat.h5")
    print(train_labels.shape, train_pair_indices.shape)
    if args.noAugs:
        history = model.fit([traindata[0,:args.nSamples], traindata[1,:args.nSamples],train_labels[:args.nSamples]], train_labels[:args.nSamples], epochs=numberOfEpochs, batch_size=batchSize,
                        callbacks=modelCallbacks,
                        validation_data=([valdata[0,:args.nSamples], valdata[1,:args.nSamples],val_labels[:args.nSamples]], val_labels[:args.nSamples]))
    else:
        class CustomCallback(Callback):
            def __init__(self):
                self.best = 99999999.
                self.improvedlast = 0
            def on_epoch_end(self, epoch, logs=None):
                if self.best > logs.get('val_loss'):
                    self.best = logs.get('val_loss')
                    self.improvedlast = 0
                    self.model.save_weights("/home/lsheldon/models/" + modelName + "/feat.h5", overwrite=True)
                else:
                    self.improvedlast = self.improvedlast + 1
    
        custom_callback = CustomCallback()
        for ie in range(numberOfEpochs):
            mc_copy = np.array(traindata[0,:args.nSamples], copy=True) 
            data_copy = np.array(traindata[1,:args.nSamples], copy=True) 
            mc_views = np.random.choice(4,mc_copy.shape[0],p=args.augMC)
            data_views = np.random.choice(4,data_copy.shape[0],p=args.augData)
            if 'smear' in args.augs:
                mc_copy = smear_eta_phi(mc_copy,nom_train_pts[:args.nSamples])
                data_copy = smear_eta_phi(data_copy,alt_train_pts[:args.nSamples])
            if 'rotate' in args.augs:
                mc_copy = rotate_jet(mc_copy)
                data_copy = rotate_jet(data_copy)
            if 'shift' in args.augs:
                data_copy = align_pts(mc_copy,data_copy)
	    #train_all = np.transpose(np.array([traindata[1,:args.nSamples],data_copy,traindata[0,:args.nSamples],mc_copy]),axes=[1,0,2,3])
            train_mc = np.transpose(np.array([traindata[0,:args.nSamples],mc_copy,traindata[1,:args.nSamples],data_copy]),axes=[1,0,2,3])
            train_data = np.transpose(np.array([traindata[1,:args.nSamples],data_copy,traindata[0,:args.nSamples],mc_copy]),axes=[1,0,2,3])
            history = model.fit([train_mc[np.arange(train_mc.shape[0]), mc_views], train_data[np.arange(train_data.shape[0]), data_views]], train_labels[:args.nSamples], epochs=1, batch_size=batchSize,
                            callbacks=[custom_callback],
                            validation_data=([valdata[0,:args.nSamples], valdata[1,:args.nSamples]], val_labels[:args.nSamples]))
            if custom_callback.improvedlast==0:
                print('Model improved... Current best valloss is',custom_callback.best)
            if args.patience>=0 and custom_callback.improvedlast>args.patience:
                break
    
    for h in history.history:
        for ie in range(len(history.history[h])):
            history.history[h][ie] = float(history.history[h][ie])
    with open("/home/lsheldon/models/" + modelName + "/history.json", "w") as f:
        print(history.history.keys())
        json.dump(history.history,f)
    
    print("Loading weights")
    
    model.load_weights("/home/lsheldon/models/" + modelName + "/feat.h5")
    
    featureExtractor.save("/home/lsheldon/models/" + modelName + "/featModel.h5") 
    model.save("/home/lsheldon/models/" + modelName + "/modelSiam.h5") 
    projector.save("/home/lsheldon/models/" + modelName + "/modelProj.h5") 

else:
    featureExtractor = load_model("/home/lsheldon/models/" + modelName + "/featModel.h5", custom_objects={'tf': tf, "RR":RR, "RS":RS, "RRT":RRT, "RST":RST, "particlesConsidered":particlesConsidered, "similarity":similarity, "vicreg_covariance":vicreg_covariance, "vicreg_variance":vicreg_variance, "vicregSVC":vicregSVC, "contrastive_loss_sim":contrastive_loss_sim, "contrastive_loss_simclr":contrastive_loss_simclr, "contrastive_loss_vicreg":contrastive_loss_vicreg, "contrastive_loss":contrastive_loss, "simCLR":simCLR})
    print('featurizer model loaded')

if args.trainClass:
    nom_bkg_data = fOne["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1]
    nom_bkg_data = nom_bkg_data[:,:,[2, 0, 1]]
    nom_bkg_jet_data = fOne["jet_features"][:].astype(np.float32)
    nom_bkg_data = np.concatenate([nom_bkg_jet_data,np.reshape(np.transpose(nom_bkg_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((nom_bkg_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
    print('Loaded nom_bkg')

    nom_sig_data = fTwo["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
    nom_sig_data = nom_sig_data[:,:,[2, 0, 1]]
    nom_sig_jet_data = fTwo["jet_features"][:].astype(np.float32)
    nom_sig_data = np.concatenate([nom_sig_jet_data,np.reshape(np.transpose(nom_sig_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((nom_sig_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
    nom_sig_data[:,-1] = 1.
    print('Loaded nom_sig')

    alt_bkg_data = fThree["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
    alt_bkg_data = alt_bkg_data[:,:,[2, 0, 1]]
    alt_bkg_jet_data = fThree["jet_features"][:].astype(np.float32)
    alt_bkg_data = np.concatenate([alt_bkg_jet_data,np.reshape(np.transpose(alt_bkg_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((alt_bkg_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
    print('Loaded alt_bkg')

    alt_sig_data = fFour["particle_features"][:].astype(np.float32)[:,:particlesConsidered,:-1] 
    alt_sig_data = alt_sig_data[:,:,[2, 0, 1]]
    alt_sig_jet_data = fFour["jet_features"][:].astype(np.float32)
    alt_sig_data = np.concatenate([alt_sig_jet_data,np.reshape(np.transpose(alt_sig_data,(0,2,1)),(-1,particlesConsidered*entriesPerParticle)),np.zeros((alt_sig_jet_data.shape[0],(particlesTotal-particlesConsidered)*entriesPerParticle+1))],axis=-1)
    alt_sig_data[:,-1] = 1.
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

    nom_data = nom_data[nom_data[:,eventDataLength]>0.,:]
    alt_data = alt_data[alt_data[:,eventDataLength]>0.,:]

    nom_data = nom_data[nom_data[:,coldict["jet_sdmass"]]>args.masscut,:]
    alt_data = alt_data[alt_data[:,coldict["jet_sdmass"]]>args.masscut,:]

    nomTrainingDataLength = int(len(nom_data)*0.8)
    nomValidationDataLength = int(len(nom_data)*0.1)
    altTrainingDataLength = int(len(alt_data)*0.8)
    altValidationDataLength = int(len(alt_data)*0.1)

    nom_train = nom_data[:nomTrainingDataLength,eventDataLength:decayTypeColumn]
    alt_train = alt_data[:altTrainingDataLength,eventDataLength:decayTypeColumn]
    nom_val = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,eventDataLength:decayTypeColumn]
    alt_val = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,eventDataLength:decayTypeColumn]
    nom_test = nom_data[nomTrainingDataLength + nomValidationDataLength:,eventDataLength:decayTypeColumn]
    alt_test = alt_data[altTrainingDataLength + altValidationDataLength:,eventDataLength:decayTypeColumn]

    nom_train_true = nom_data[:nomTrainingDataLength,decayTypeColumn:]
    alt_train_true = alt_data[:altTrainingDataLength,decayTypeColumn:]
    nom_val_true = nom_data[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength,decayTypeColumn:]
    alt_val_true = alt_data[altTrainingDataLength:altTrainingDataLength + altValidationDataLength,decayTypeColumn:]
    nom_test_true = nom_data[nomTrainingDataLength + nomValidationDataLength:,decayTypeColumn:]
    alt_test_true = alt_data[altTrainingDataLength + altValidationDataLength:,decayTypeColumn:]

    nom_test_vars = np.transpose(nom_test.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
    alt_test_vars = np.transpose(alt_test.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
    nom_train_vars = np.transpose(nom_train.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
    alt_train_vars = np.transpose(alt_train.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
    nom_val_vars = np.transpose(nom_val.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))
    alt_val_vars = np.transpose(alt_val.reshape(-1, entriesPerParticle, particlesTotal)[:,:,:particlesConsidered], axes=(0, 2, 1))

    if not args.fineTune:
        nom_train_space = featureExtractor.predict(nom_train_vars[:args.nSamples])
        alt_train_space = featureExtractor.predict(alt_train_vars[:args.nSamples])
        nom_val_space = featureExtractor.predict(nom_val_vars[:args.nSamples])
        alt_val_space = featureExtractor.predict(alt_val_vars[:args.nSamples])
    else:
        nom_train_space = nom_train_vars[:args.nSamples]
        alt_train_space = alt_train_vars[:args.nSamples]
        nom_val_space = nom_val_vars[:args.nSamples]
        alt_val_space = alt_val_vars[:args.nSamples]
    

    classLayers = args.classLayers if len(args.classLayers)>1 or args.classLayers[0]!=0 else []
    inputContrastive = Input(shape=(featureExtractor.layers[-1].output_shape[1:]), name="inputContrastive")
    for il in range(0,len(classLayers)-1):
        if(args.batchNorm and il>0): denseClass = BatchNormalization(momentum=0.6, name="batchnormClass%i"%il)(denseClass)
        denseClass = Dense(classLayers[il], activation="relu", name="denseClass%i"%il, kernel_regularizer=l2(args.l2reg))(denseClass if il>0 else inputContrastive)
    if(args.batchNorm and len(classLayers)>0): denseClass = BatchNormalization(momentum=0.6, name="batchnormClass")(denseClass)
    denseOutput = Dense(1, activation="sigmoid", name="contrastiveOutput")(denseClass if len(classLayers)>0 else inputContrastive)

    if args.fineTune:
        classmodel = Model(inputs=inputContrastive, outputs=denseOutput)
        featureExtractor._name = "featurizer_model"
        classmodel._name = "classifier_model"
        print(classmodel.summary())
        print(featureExtractor.summary())
        partinput = Input(shape=(particlesConsidered, entriesPerParticle), name='partinput')
        modelout = classmodel(featureExtractor(partinput))
        model = Model(inputs=partinput, outputs=modelout)
    else:
        model = Model(inputs=inputContrastive, outputs=denseOutput)

    modelCallbacks = [EarlyStopping(patience=10),
                  ModelCheckpoint(filepath="/home/lsheldon/models/" + modelNameAdd + "/class.h5", save_weights_only=True,
                                  save_best_only=True),
                  ]
    print(model.summary())

    if args.LARS:
        model.compile(optimizer=LARS(args.LR,weight_decay=0.000001), loss='binary_crossentropy', metrics=['acc'])
    elif args.Adam:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    else:
        modelCallbacks.append(LearningRateScheduler(lr_scheduler, verbose=0))
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['acc'])
	
    del mc_copy
    del data_copy
    if args.fineTune > 0:
        history = model.fit(nom_train_space, nom_train_true[:args.nSamples],
                        epochs=args.fineTune,batch_size=batchSize if args.fineTune else 128,
                        callbacks=modelCallbacks,
                        validation_data=(nom_val_space, nom_val_true[:args.nSamples]))
        featureExtractor.trainable = False
    history = model.fit(nom_train_space, nom_train_true[:args.nSamples],
                        epochs=numberOfEpochs,batch_size=batchSize if args.fineTune else 128,
                        callbacks=modelCallbacks,
                        validation_data=(nom_val_space, nom_val_true[:args.nSamples]))

    for h in history.history:
        for ie in range(len(history.history[h])):
            history.history[h][ie] = float(history.history[h][ie])
    with open("/home/lsheldon/models/" + modelNameAdd + "/classhistory.json", "w") as f:
        print(history.history.keys())
        json.dump(history.history,f)

    print("Loading weights")

    model.load_weights("/home/lsheldon/models/" + modelNameAdd + "/class.h5")


    if not args.fineTune:
        model.save("/home/lsheldon/models/" + modelNameAdd + "/classModel.h5") 
        partinput = Input(shape=(particlesConsidered, entriesPerParticle), name='partinput')
        featureExtractor._name = "featurizer_model"
        model._name = "classifier_model"
        fullout = model(featureExtractor(partinput))
        model = Model(inputs=partinput, outputs=fullout)

    model.save("/home/lsheldon/models/" + modelNameAdd + "/fullModel.h5")
    if args.fineTune:
        featureExtractor.save("/home/lsheldon/models/" + modelNameAdd + "/ftFeatModel.h5")
    print('saved full model')

