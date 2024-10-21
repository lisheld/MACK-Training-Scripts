# MACK: Mismodeling Addressed with Contrastive Knowledge [arXiv:2410.13947]

## Abstract

The use of machine learning methods in high energy physics typically relies on large volumes of precise simulation for training. As machine learning models become more complex they can become increasingly sensitive to differences between this simulation and the real data collected by experiments. We present a generic methodology based on contrastive learning which is able to greatly mitigate this negative effect. Crucially, the method does not require prior knowledge of the specifics of the mismodeling. While we demonstrate the efficacy of this technique using the task of jet-tagging at the Large Hadron Collider, it is applicable to a wide array of different tasks both in and out of the field of high energy physics.

## Datasets

We provide two training models each (Supervised and Contrastive) for the two types of datasets tested. The "standard" dataset simulates Z' -> qq vs. q/g, while the JetNet dataset simulates q(g) vs. W(Z). Each respective type of model is the same across datasets, but with slight changes to how data is loaded.

## Arguments

### Standard Dataset Model
* `label`: The label added to the end of the model when saves
* `labeladd`: Additional label to add when training classifiers
* `nSamples`: The number of samples to consider in the dataset
* `Adam`: Whether or not to use the ADAM optimizer
* `LR`: Sets the learning rate
* `decayEpochs`: Number of epochs to decay with a cosine scheduler
* `warmLR`: Number of epochs to use during warmup with a cosine scheduler
* `fLR`: Sets the epoch to end the cosine scheduler
* `l2reg`: Whether or not to use L2 regularization
* `nEpochs`: Defines the number of epochs to train over
* `particles`: Defines the number of particles to use in each jet
* `batchSize`: Defines the batch size for every epoch
* `redoPairs`: Whether or not to remake pairs
* `negPairs`: Oversampling factor to use when constructing negative pairs
* `posPairs`: Oversampling factor to use when constructing positive pairs
* `noSigPairs`: Whether or not to use signal MC data when making pairs
* `batchNorm`: Whether or not to use batch normalization
* `noTSNE`: Whether or not to produce a plot of the contrastive feature space using the t-SNE method
* `fineTune`: Whether or not to fineTune (see paper)
* `DNN`: Whether or not to use a fully connected architecture (default is no which uses a GNN)
* `EMD`: Defines the EMD cutoff for pairs
* `EMDNorm`: Whether or not to use normalized or non-normalized EMD
* `patience`: The number of epochs to employ EarlyStopping after
* `denseLayers`: Dense layer architecture
* `projLayers`: Projection layer architecture
* `classLayers`: Classifier layer architecture
* `useSig`: Whether or not to consider signal MC data
* `masscut`: Defines the minimum mass for jets
* `pairName`: What to save pairs as if redoing them
* `loss`: Loss function to use
* `lossParams`: Sets the parameters to use in the contrastive loss
* `trainFeat`: Whether or not to train the featurizer
* `reloadFeat`: Whether or not to reload the weights for the featurizer
* `trainClass`: Whether or not to train the classifier
* `doAttention`: Whether or not to use attention in the GNN architecture
* `noAugs`: Whether or not to apply augmentations to data
* `augData`: Defines the probabilities of each type of data for the first item in the pair. Format is [P<sub>Alt</sub>, P<sub>Aug-Alt</sub>, P<sub>Nom</sub>, P<sub>Aug-Nom</sub>]. For example, [1,0,0,0] means the first item in each pair is exclusively non-augmented Alt data. [0.5,0.5,0,0] means the first item has a 50/50 chance to be either non-augmented Alt data or augmented Alt data.
* `augMC`: Defines the probabilities of each type of data for the second item in the pair, same concept as augData. Format is [P<sub>Nom</sub>, P<sub>Aug-Nom</sub>, P<sub>Alt</sub>, P<sub>Aug-Alt</sub>]
* `etaphiSmear`: Scale to use when using the smearing augmentation
* `augs`: The type of augmentations to use if augmenting data

### JetNet Dataset Model
The arguments are identical to those for the standard dataset model, with the addition of one:
* `ptcut`: Defines the minimum momentum for jets
