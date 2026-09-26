#!/bin/bash
# ensemble eval with the ftsafe merge (needs the video's L_med)
CK=$1; TAG=$2; RUN=${3:-run5}; MERGE=${4:-ftsafe}
declare -A LM=( [IMG_8370]=536.8 [IMG_8373]=539.6 [IMG_8374]=527.1 )
for v in IMG_8370 IMG_8373 IMG_8374; do
  docker run --rm -v $HOME/dtc:/work dtc:latest python /work/run_dtc2.py /work/clips/${v}_s6.mp4 \
    --scale 6 --block 15 --C 1 --thr 0.3 --tag $TAG --params /work/ft/$RUN/$CK --params2 base \
    --merge $MERGE --lmed ${LM[$v]}
done
docker run --rm -v $HOME/dtc:/work dtc:latest python /work/evaluate.py $TAG
