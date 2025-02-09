#!/bin/bash
set -ex

summary_path=$1
model_dir=$2
start=$3
end=$4

# Calculate the range size and segment size
range_size=$((end - start))
segment_size=$((range_size / 8))

# Launch parallel jobs with calculated ranges
bash eval_7b_1.sh $summary_path $model_dir $start $((start + 1 * segment_size)) &
echo "eval_7b_1.sh processing range: $start to $((start + 1 * segment_size))"

bash eval_7b_2.sh $summary_path $model_dir $((start + 1 * segment_size)) $((start + 2 * segment_size)) &
echo "eval_7b_2.sh processing range: $((start + 1 * segment_size)) to $((start + 2 * segment_size))"

bash eval_7b_3.sh $summary_path $model_dir $((start + 2 * segment_size)) $((start + 3 * segment_size)) &
echo "eval_7b_3.sh processing range: $((start + 2 * segment_size)) to $((start + 3 * segment_size))"

bash eval_7b_4.sh $summary_path $model_dir $((start + 3 * segment_size)) $((start + 4 * segment_size)) &
echo "eval_7b_4.sh processing range: $((start + 3 * segment_size)) to $((start + 4 * segment_size))"

bash eval_7b_5.sh $summary_path $model_dir $((start + 4 * segment_size)) $((start + 5 * segment_size)) &
echo "eval_7b_5.sh processing range: $((start + 4 * segment_size)) to $((start + 5 * segment_size))"

bash eval_7b_6.sh $summary_path $model_dir $((start + 5 * segment_size)) $((start + 6 * segment_size)) &
echo "eval_7b_6.sh processing range: $((start + 5 * segment_size)) to $((start + 6 * segment_size))"

bash eval_7b_7.sh $summary_path $model_dir $((start + 6 * segment_size)) $((start + 7 * segment_size)) &
echo "eval_7b_7.sh processing range: $((start + 6 * segment_size)) to $((start + 7 * segment_size))"

bash eval_7b_8.sh $summary_path $model_dir $((start + 7 * segment_size)) $end
echo "eval_7b_8.sh processing range: $((start + 7 * segment_size)) to $end"
