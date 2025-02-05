#!/bin/bash

repo_path=$1
input_dirs=$2

echo repo_path = $(repo_path)

for dirpath in input_dirs; do
  echo processing $dirpath
done

