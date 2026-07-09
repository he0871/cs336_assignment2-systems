brew install --cask google-cloud-sdk

gcloud init

gcloud storage cp ~/code/assignment1-basics/data/tinystories_train_encoded.txt gs://cs336-data/

uv pip install google-cloud-storage
