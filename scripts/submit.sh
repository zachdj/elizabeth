#!/bin/bash

# Configuration
CLUSTER="elizabeth-${RANDOM}"
REGION="us-east1"
ZONE="us-east1-d"
DRIVER_CORES='4'
DRIVER_MEMORY='6g'
EXECUTOR_CORES='4'
EXECUTOR_MEMORY='6g'
WORKER_MEMORY='4g'
INIT_ACTION='gs://cbarrick/dataproc-bootstrap.sh'

echo "==> INFO"
echo "command: $@"
echo "--> Google Cloud"
echo "cluster: $CLUSTER"
echo "region:  $REGION"
echo "zone:    $ZONE"
echo "--> Spark"
echo "spark.driver.cores:          $DRIVER_CORES"
echo "spark.driver.memory:         $DRIVER_MEMORY"
echo "spark.executor.cores:        $EXECUTOR_CORES"
echo "spark.executor.memory:       $EXECUTOR_MEMORY"
echo "spark.python.worker.memory:  $WORKER_MEMORY"
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
	--initialization-actions "$INIT_ACTION"
echo

# Submit the job
echo "==> Submitting job: $@"
PROPERTIES="spark.driver.cores=$DRIVER_CORES"
PROPERTIES+=",spark.driver.memory=$DRIVER_MEMORY"
PROPERTIES+=",spark.executor.cores=$EXECUTOR_CORES"
PROPERTIES+=",spark.executor.memory=$EXECUTOR_MEMORY"
PROPERTIES+=",spark.python.worker.memory=$WORKER_MEMORY"
gcloud dataproc jobs submit pyspark \
	--cluster $CLUSTER \
	--region $REGION \
	--driver-log-levels root=FATAL \
	--properties $PROPERTIES \
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
