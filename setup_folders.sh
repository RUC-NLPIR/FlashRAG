#!/bin/bash

# Lista delle cartelle da creare
folders=("datasets" "indexes" "models" "results" "tests" "webui_configs")

for folder in "${folders[@]}"; do
    mkdir -p "$folder"
    touch "$folder/.gitkeep"
done

echo "✅ Directory structure created and preserved for Git using .gitkeep files!"