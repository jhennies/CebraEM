#!/bin/bash

TORCH=true
NAME=cebra-em-env

while getopts :htn: opt; do
  case $opt in
    h)
      echo ""
      echo "Usage: ./install.sh [-h] [-n] [-t] PACKAGE"
      echo ""
      echo "  -h    Help"
      echo "  -t    Do not install pytorch"
      echo "  -n    Environment name"
      echo ""
      echo "  PACKAGE: The name of the package to install: [\"ann\", \"inf\", \"core\", \"all\"]"
      echo ""
      exit 0
      ;;
    t)
      TORCH=false
      ;;
    n)
      NAME=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument"
      exit 1
  esac
done

PACKAGE=${@:$OPTIND:1}
if [ -z "$PACKAGE" ]; then
  echo "No package specified!"
  exit 1
fi
if [ "$PACKAGE" != ann ] && [ "$PACKAGE" != inf ] && [ "$PACKAGE" != core ] && [ "$PACKAGE" != all ]; then
  echo "Invalid package specifier: $PACKAGE"
  exit 1
fi

source activate base

# Always create an environment with python, elf and vigra
conda create -y -n "$NAME" -c conda-forge python=3.9 python-elf vigra || exit 1
conda activate "$NAME" || exit 1
conda list
# And always install the cebra-em-core package
pip install -e ./cebra-em-core/ || exit 1

# Install the cebra-ann or -inf packages only if requested
if [ "$PACKAGE" == ann ] || [ "$PACKAGE" == all ]; then
  pip install -e ./cebra-ann/ || exit 1
fi
if [ "$PACKAGE" == inf ] || [ "$PACKAGE" == all ]; then
  pip install -e ./cebra-inf/ || exit 1
fi

# Install torch by default
if [ "$TORCH" == true ]; then
  install_torch.py || exit 1
fi

echo ""
echo "Installation successful!"
echo "Activate the environment using \"conda activate $NAME\""
