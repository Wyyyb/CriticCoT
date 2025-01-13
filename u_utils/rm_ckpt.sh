
#!/bin/bash

#if [ $# -ne 1 ]; then
#    echo "Usage: $0 <directory_path>"
#    exit 1
#fi

dir_path="../LLaMA-Factory/output_models_dense_0112/"

if [ ! -d "$dir_path" ]; then
    echo "Error: Directory $dir_path does not exist"
    exit 1
fi

# 首先列出要删除的目录
echo "The following directories will be deleted:"
found_dirs=0
for subdir in "$dir_path"/*; do
    if [ -d "$subdir" ]; then
        for ckpt_dir in "$subdir"/checkpoint-*; do
            if [ -d "$ckpt_dir" ]; then
                num=$(basename "$ckpt_dir" | grep -o '[0-9]\+')
                if [ ! -z "$num" ] && [ "$num" -gt 1200 ]; then
                    echo "$ckpt_dir"
                    found_dirs=1
                fi
            fi
        done
    fi
done

if [ $found_dirs -eq 0 ]; then
    echo "No directories found with checkpoint number > 1200"
    exit 0
fi

# 询问用户确认
read -p "Are you sure you want to delete these directories? (y/N): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Operation cancelled"
    exit 0
fi

# 执行删除操作
echo "Deleting directories..."
for subdir in "$dir_path"/*; do
    if [ -d "$subdir" ]; then
        for ckpt_dir in "$subdir"/checkpoint-*; do
            if [ -d "$ckpt_dir" ]; then
                num=$(basename "$ckpt_dir" | grep -o '[0-9]\+')
                if [ ! -z "$num" ] && [ "$num" -gt 1200 ]; then
                    echo "Removing $ckpt_dir"
                    rm -rf "$ckpt_dir"
                fi
            fi
        done
    fi
done

echo "Deletion completed"