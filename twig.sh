#!/bin/bash
model_path=Janus-Pro-7B   # Local path to the pre-trained model weights
output_folder=twig        # Subfolder name for this specific evaluation task
base_output=/path/to/your/project/results     # Root directory to save all generated images and text results
file_name=twig.py         # The evaluation script to execute

n=8                       # Number of concurrent GPUs
threshold=90              # Reflection score threshold to trigger layerwise regeneration

run_parallel_eval() {
    local subset=$1
    local output_path=$base_output/$output_folder/$subset
    echo "Subset: $subset"
    for i in $(seq 0 $((n - 1))); do
        echo "Run $i (index=$i)"
        CUDA_VISIBLE_DEVICES=$i python3 "$file_name" \
            --idx $i --num_workers $n \
            --model_path "$model_path" --output_path "$output_path" \
            --theta $threshold --dataset "dataset/${subset}_val.txt" &
    done
    wait
    echo "Subset $subset generation done."
    echo "----------------------------------------"
    echo "Running post-evaluation."
}

# --- color / shape / texture: BLIP score ---
for subset in color shape texture; do
    run_parallel_eval "$subset"
    output_path=$base_output/$output_folder/$subset
    cd BLIP && bash test.sh "$output_path" && cd ..
    echo "Subset $subset completed!"
    echo "----------------------------------------"
done

# --- non_spatial: CLIP score ---
run_parallel_eval non_spatial
output_path=$base_output/$output_folder/non_spatial
cd CLIP
python3 CLIP_similarity.py --outpath="${output_path}"
cd ..
echo "Subset non_spatial completed!"
echo "----------------------------------------"

# --- spatial / 3d_spatial / numeracy: UniDet ---
for subset in spatial 3d_spatial numeracy; do
    run_parallel_eval "$subset"
    output_path=$base_output/$output_folder/$subset
    case $subset in
        spatial)
            cd UniDet && python3 2D_spatial_eval.py --outpath "$output_path" && cd ..
            ;;
        3d_spatial)
            cd UniDet && python3 3D_spatial_eval.py --outpath "$output_path" && cd ..
            ;;
        numeracy)
            cd UniDet && python3 numeracy_eval.py --outpath "$output_path" && cd ..
            ;;
    esac
    echo "Subset $subset completed!"
    echo "----------------------------------------"
done

# --- complex: 3-in-1 ---
run_parallel_eval complex
output_path=$base_output/$output_folder/complex
echo "output_path: $output_path"
cd UniDet && python3 2D_spatial_eval.py --outpath "$output_path" --complex=True && cd ..
cd CLIP  && python3 CLIP_similarity.py --outpath="${output_path}" --complex=True && cd ..
cd BLIP  && bash test.sh "$output_path" && cd ..
cd 3_in_1_eval && python3 3_in_1.py --outpath="$output_path" && cd ..

echo "Subset complex completed!"
