#!/bin/bash

./setup.py bdist_egg

gcloud dataproc jobs submit pyspark \
	--cluster hydrus \
	--region us-east1 \
	--py-files ./dist/elizabeth-0.1.0.dev0-py3.6.egg \
	./scripts/driver.py \
	-- $@
