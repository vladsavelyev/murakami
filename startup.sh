conda create -n python310 python=3.10
git clone https://github.com/vladsavelyev/ohota-na-ovets.git
conda run -n python310 pip3 install --no-cache-dir -r ohota-na-ovets/requirements.txt
conda run -n python310 dvc pull