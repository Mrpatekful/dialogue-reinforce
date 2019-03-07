# Installs the required libraries, including ParlAI.

pip install -r .\requirements.txt

pip install https://download.pytorch.org/whl/cu90/torch-1.0.1-cp36-cp36m-win_amd64.whl
pip install torchvision

git clone https://github.com/facebookresearch/ParlAI.git $env:userprofile\ParlAI
python $env:userprofile\ParlAI\setup.py develop
