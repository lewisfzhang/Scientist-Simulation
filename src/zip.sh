#!/bin/bash

/usr/bin/tar -C ../data -czf ../data/zipped/run_$(echo $* | tr ' ' '_').tar.gz images pages parameters.txt output.txt saved ../src/tmp/model
