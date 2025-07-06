#/bin/bash

set -xv

TASK=$1
VERSION=$2

# cd ./math_serving/reasonreason && git checkout main && git pull && cd -

sudo docker build --network host -t harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION} . -f docker/Dockerfile

sudo docker push harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION}
