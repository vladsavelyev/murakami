curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
. /opt/conda/etc/profile.d/conda.sh
conda create -n python310 -y python=3.10
conda activate python310
git clone https://github.com/vladsavelyev/ohota-na-ovets.git
cd ohota-na-ovets
pip3 install --no-cache-dir -r requirements.txt
dvc pull
