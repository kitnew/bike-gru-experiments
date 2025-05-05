#!/bin/bash

# Script to clean up PyTorch checkpoint files (.pt), keeping only the latest epoch
# for each model in the experiments directory and its subdirectories

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"

echo "Scanning directory: $ROOT_DIR"

# Recursively find all directories in the experiments folder
find "$ROOT_DIR" -type d | while read -r dir; do
    echo "Processing directory: $dir"
    
    # Check if any .pt files exist in this directory
    pt_files=$(find "$dir" -maxdepth 1 -name "*.pt" | wc -l)
    
    if [ "$pt_files" -gt 0 ]; then
        echo "Found $pt_files checkpoint files in $dir"
        
        # Create temporary directory for our work
        tmp_dir=$(mktemp -d)
        
        echo "Creating temporary directory: $tmp_dir"
        
        # Group files by their base model name (before _epoch)
        for pt_file in $(find "$dir" -maxdepth 1 -name "*.pt"); do
            filename=$(basename "$pt_file")
            echo "Processing file: $filename"
            # Extract parts from filename (model_epoch{NUM}_valloss...)
            if [[ $filename =~ _epoch([0-9]+)_ ]]; then
                epoch_num=${BASH_REMATCH[1]}
                model_prefix=${filename%%_epoch*}
                
                # Keep track of highest epoch for each model prefix
                current_max=0
                if [ -f "$tmp_dir/$model_prefix.epoch" ]; then
                    current_max=$(cat "$tmp_dir/$model_prefix.epoch")
                    echo "Current max for $model_prefix: $current_max"
                fi
                
                if [ $epoch_num -gt $current_max ]; then
                    echo $epoch_num > "$tmp_dir/$model_prefix.epoch"
                    echo "$pt_file" > "$tmp_dir/$model_prefix.file"
                fi
            fi
        done
        
        # Now process each model and keep only the latest epoch
        for model_info in $(find "$tmp_dir" -name "*.epoch" | sort); do
            model_prefix=$(basename "$model_info" .epoch)
            latest_epoch=$(cat "$model_info")
            latest_file=$(cat "$tmp_dir/$model_prefix.file")
            
            echo "Model: $model_prefix - Keeping epoch $latest_epoch: $(basename "$latest_file")"
            
            # Delete all other checkpoints for this model
            for pt_file in $(find "$dir" -maxdepth 1 -name "${model_prefix}_epoch*.pt"); do
                if [ "$pt_file" != "$latest_file" ]; then
                    if [[ $(basename "$pt_file") =~ _epoch([0-9]+)_ ]]; then
                        delete_epoch=${BASH_REMATCH[1]}
                        echo "  Removing epoch $delete_epoch: $(basename "$pt_file")"
                        rm "$pt_file"
                    fi
                fi
            done
        done
        
        # Clean up our temporary directory
        rm -rf "$tmp_dir"
    else
        echo "No checkpoint files found in $dir"
    fi
done

echo "Cleanup complete!"