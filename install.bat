
echo OFF

conda create -y -n cebra-em-env -c conda-forge python=3.9 python-elf vigra
call conda activate cebra-em-env

pip install -e cebra-em-core\

pip install -e cebra-ann\

python cebra-em-core\bin\install_torch.py

echo ""
echo "Installation successful!"
echo "Activate the environment using 'conda activate cebra-em-env'"


