#/bin/bash

set -xv

TASK=crl
VERSION=0.3

# cd ./math_serving/reasonreason && git checkout main && git pull && cd -

sudo docker build --network host -t harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION} . -f docker/Dockerfile_verl_yb

sudo docker push harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION}
