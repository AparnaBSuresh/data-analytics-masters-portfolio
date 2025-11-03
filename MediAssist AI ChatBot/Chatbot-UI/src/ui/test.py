from huggingface_hub import HfApi
api = HfApi()
info = api.model_info("AparnaSuresh/MedLlama-3b")  # exact id, exact case
print(info.modelId, info.private, info.pipeline_tag)  # should print and NOT raise
