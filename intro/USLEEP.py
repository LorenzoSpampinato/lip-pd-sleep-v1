import os
from usleep_api import USleepAPI

# Create an API object with API token stored in environment variable
api = USleepAPI(api_token=os.environ['USLEEP_API_TOKEN'])

# Predict on anonymized PSG and save hypnogram to file
hypnogram, log = api.quick_predict(
    input_file_path="./psg_001.edf",
    output_file_path="./psg_001_hypnogram.tsv",
    anonymize_before_upload=True
)
