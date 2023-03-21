. /opt/conda/etc/profile.d/conda.sh
conda activate python310
export HUB_TOKEN=$(gcloud secrets versions access latest --secret hf-token-murakami --project vlad-saveliev)
export WANDB_API_KEY=$(gcloud secrets versions access latest --secret wandb --project vlad-saveliev)
cd ohota-na-ovets
python main.py