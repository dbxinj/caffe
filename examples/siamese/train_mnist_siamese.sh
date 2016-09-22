#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train --gpu=0 --solver=examples/siamese/mnist_siamese_solver.prototxt
