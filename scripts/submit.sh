#!/bin/bash
set -e

# Configuration
CLUSTER="elizabeth-${RANDOM}"
REGION="us-east1"
ZONE="us-east1-d"
echo "==> INFO"
echo "name:    $CLUSTER"
echo "region:  $REGION"
echo "zone:    $ZONE"
echo "command: $@"
echo

# Compile the module
echo "==> Compiling the module"
./setup.py bdist_egg
echo

# Provision a cluster
echo "==> Provisioning cluster $CLUSTER in $ZONE ($REGION)"
gcloud dataproc clusters create $CLUSTER \
	--region $REGION \
	--zone $ZONE \
	--worker-machine-type n1-standard-4 \
	--num-workers 4 \
	--initialization-actions gs://cbarrick/dataproc-bootstrap.sh
echo

# Submit the job
echo "==> Submitting job: $@"
gcloud dataproc jobs submit pyspark \
	--cluster $CLUSTER \
	--region $REGION \
	--driver-log-levels root=FATAL \
	--py-files ./dist/elizabeth-0.1.0.dev0-py3.6.egg \
	./scripts/driver.py \
	-- $@
echo

# Tear down the cluster
echo "==> Tearing down the cluster"
yes | gcloud dataproc clusters delete $CLUSTER \
	--region $REGION \
	--async
echo
