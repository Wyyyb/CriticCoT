#/bin/bash

set -xv

TASK=yb_eval
VERSION=0.1

# cd ./math_serving/reasonreason && git checkout main && git pull && cd -

sudo docker build --network host -t harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION} . -f docker/Dockerfile_eval

sudo docker push harbor.xaminim.com/minimax-dialogue/$TASK:${VERSION}



