# Elizabeth
Scalable malware detection

Elizabeth is a Spark experiment for the [Microsoft Malware Classification Challenge][MMCC]. It was created over the course of two weeks for the Spring 2018 rendition of The University of Georgia's CSCI 8360 Data Science Practicum.

[MMCC]: https://www.kaggle.com/c/malware-classification/


## Getting Started

The experiment is written in Python 3.6 and PySpark 2.2, and is packaged as a traditional Python package. We provide a `setup.py` script for building and installing, but we do not recommend any particular installation procedure. In practice, it is sufficient to launch the experiment from the root of this repo:

```sh
python -m elizabeth <subcommand> [<options>] train_x train_y test_x [test_y]
```

Currently we offer two subcommands, `nb` and `rf`, both accepting similar arguments. The `nb` command trains a naive Bayes classifier on a train set and prints labels for a test set. Likewise the `rf` command trains a random forrest classifier on a train set and prints labels for a test set. For both subcommands, if labels are provided for the test set, the accuracy is printed instead of labels. For a full listing of options, pass the `--help` option to any subcommand.


## Data Format

The inputs to the classifiers are manifest files (`train_x` and `test_x`) and label files (`train_y` and `test_y`). The manifest files are a set of hashes, one per line, indicating a bytes or assembly file in the dataset. The label files have the same number of lines as the corresponding manifest files and each line is an integer label (1-9) for the corresponding instance in the manifest.

The dataset must be stored in a filesystem with the hierarchy `$BASEPATH/$KIND/$HASH.$KIND` where `$BASEPATH` is the path to the dataset, `$KIND` is one of `bytes` or `asm`, and `$HASH` is a hash listed in a manifest file. The default base path is on a Google Storage bucket hosted by @magsol.


## Deploying to Google Cloud Dataproc

This experiment was executed on Dataproc. The `scripts` directory contains the tooling we used to accomplish this. In particular, the `submit.sh` script handles provisioning a cluster, submitting a job, and tearing down the cluster. Any arguments passed to the script are forwarded to the `elizabeth` package.

The top of the submit script contains a few configuration options which you may wish to edit before submitting a job. Specifically, you should upload the `scripts/dataproc-bootstrap.sh` to a Google Storage bucket and point the `INIT_ACTION` variable to the `gs://` url of that script.


## Credits

Copyright (c) 2018

- Chris Barrick <cbarrick1@gmail.com>
- Zach Jones <zach.dean.jones@gmail.com>
- Jeremy Shi <jeremy.shi@uga.edu>
