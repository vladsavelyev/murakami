# Ohota-na-ovets

GTP2, fine-tuned to Russian ([rugpt3small_based_on_gpt2](https://huggingface.co/sberbank-ai/rugpt3small_based_on_gpt2)), further fine-tuned to generate Russian translations of H. Murakami books.

## Train

1. Create a GCP instance with a GPU available. The `deeplearning-platform-release` project provides instance images with pre-installed Nvidia drivers and conda, along with common ML tools (though we are interested only in conda, as we will be setting a python 3.10 environment opposed to the pre-installed python 3.7).

```sh
NAME=murakami
gcloud beta compute instances create ${NAME} \
--zone australia-southeast1-c \
--image-project deeplearning-platform-release \
--image-family pytorch-latest-gpu \
--maintenance-policy=TERMINATE \
--accelerator="type=nvidia-tesla-p100,count=1" \
--metadata="install-nvidia-driver=True" \
--preemptible \
--provisioning-model=SPOT \
--max-run-duration=2h \
--scopes cloud-platform
```

Note that this requires a Global GPU quota to be requested.

To finish instance setup, run `setup_instance.sh` on it:

```sh
gcloud beta compute ssh --zone australia-southeast1-c murakami --command="$(cat setup_instance.sh)"
```

The training script will be using the Hugginface Hub and W&B, so create secrets with the Hub adn W&B tokens and give permissions to your compute service account to read those secrets (locally):

```sh
gcloud secrets create wandb-token
printf "${WANDB_TOKEN}" | gcloud secrets versions add wandb-token --data-file -

gcloud secrets create hfhub-token
printf "${HF_HUB_TOKEN}" | gcloud secrets versions add hfhub-token --data-file -

gcloud secrets add-iam-policy-binding wandb --member="serviceAccount:479973615443-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding hf-token-murakami --member="serviceAccount:479973615443-compute@developer.gserviceaccount.com" --role="roles/secretmanager.secretAccessor"
```

You can also optionally create an image from the prepared instance to avoid setting it up again.

Finally, start training:

```sh
gcloud beta compute ssh --zone australia-southeast1-c murakami --command="$(cat run_on_instance.sh)"
```
