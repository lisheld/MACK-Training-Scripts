# MACK: Mismodeling Addressed with Contrastive Knowledge

## Abstract

The use of machine learning methods in high energy physics typically relies on large volumes of precise simulation for training. As machine learning models become more complex they can become increasingly sensitive to differences between this simulation and the real data collected by experiments. We present a generic methodology based on contrastive learning which is able to greatly mitigate this negative effect. Crucially, the method does not require prior knowledge of the specifics of the mismodeling. While we demonstrate the efficacy of this technique using the task of jet-tagging at the Large Hadron Collider, it is applicable to a wide array of different tasks both in and out of the field of high energy physics.

## Datasets

We provide two training scripts for the two types of datasets tested. The "standard" dataset simulates Z' -> qq vs. q/g, while the JetNet dataset simlulates q(g) vs. W(Z). The models are the same, but both require slightly different loading methods.

## Arguments

### Standard Dataset Model
* label: The label added to the end of the model when saves
* labeladd: ?
* nSamples: The number of samples to consider in the dataset
* warmLR: ?
* LARS: ?
* Adam: Whether or not to use the ADAM optimizer
* LR: Sets the learning rate
* fLR: ?
* l2reg: ?
* nEpochs: Defines the number of epochs to train over
* particles: Defines the number of particles in sample
* decayEpochs: ?
* batchSize: Defines the batch size for every epoch
* redoPairs: Whether or not to remake pairs
* negPairs: ?
* posPairs: ?
* noSigPairs: Whether or not to use signal MC data when making pairs
* batchNorm: ?
* noTSNE: Whether or not to use the t-SNE method to reduce dimensions
* fineTune: Whether or not to fineTune (see paper)
* DNN: ?
* EMD: Defines the EMD cutoff for pairs
* EMDNorm: ?
* patience: The number of epochs to employ EarlyStopping after
* denseLayers: Dense layer architecture
* projLayers: Projection layer architecture
* classLayers: Classifier layer architecture
* useSig: Whether or not to consider signal MC data
* masscut: Defines the minimum mass for jets
* pairName: What to save pairs as if redoing them
* loss: Loss function to use
* lossParams: ?
* trainFeat: Whether or not to train the featurizer
* reloadFeat: ?
* trainClass: Whether or not to train the classifier
* doAttention: ?
* noAugs: Whether or not to apply augmentations to data
* augData: Defines the probabilities of each type of data for the first item in the pair. Format is [{P_Alt.}, {P_Aug-Alt}, {P_Nom}, {P_Aug-Nom}]. For example, [1,0,0,0] means the first item in each pair is exclusively non-augmented Alt data. [0.5,0.5,0,0] means the first item has a 50/50 chance to be either non-augmented Alt data or augmented Alt data.
* augMC: Defines the probabilities of each type of data for the second item in the pair, same concept as augData. Format is [{P_Nom}, {P_Aug-Nom}, {P_Alt.}, {P_Aug-Alt}]
* etaphiSmear: ?
* augs: The type of augmentations to use if augmenting data

### JetNet Dataset Model
The arguments are identical to those for the standard dataset model, with the addition of one:
* ptcut: Defines the minimum momentum for jets
