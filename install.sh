#!/bin/bash

TORCH=true
NAME=cebra-em-env
CONDA=conda

while getopts :htmn: opt; do
  case $opt in
    h)
      echo ""
      echo "Usage: ./install.sh [-h] [-t] [-m] [-n NAME] PACKAGE"
      echo ""
      echo "  -h    Help"
      echo "  -t    Do not install pytorch"
      echo "  -m    Use mamba instead of conda"
      echo "  -n    Environment name"
      echo ""
      echo "  PACKAGE: The name of the package to install: [\"ann\", \"em\", \"core\", \"all\"]"
      echo ""
      exit 0
      ;;
    t)
      TORCH=false
      ;;
    m)
      CONDA=mamba
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
"$CONDA" create -y -n "$NAME" -c conda-forge python=3.9 mobie_utils=0.3 python-elf vigra || exit 1
conda activate "$NAME" || exit 1
conda list
# And always install the cebra-em-core package
pip install -e ./cebra-em-core/ || exit 1

# Install the cebra-ann or -inf packages only if requested
if [ "$PACKAGE" == ann ] || [ "$PACKAGE" == all ]; then
  pip install -e ./cebra-ann/ || exit 1
fi
if [ "$PACKAGE" == em ] || [ "$PACKAGE" == all ]; then
  pip install -e ./cebra-em/ || exit 1
fi

# Install torch by default
if [ "$TORCH" == true ]; then
  install_torch.py || exit 1
fi

# Install pybdv
mkdir ext_packages
cd ext_packages || exit 1
git clone https://github.com/jhennies/pybdv.git
cd pybdv || exit 1
git checkout bdv_dataset_with_stiching
python setup.py install

echo ""
echo "Installation successful!"
echo "Activate the environment using \"conda activate $NAME\""
