import transformers
import bentoml

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml.exceptions import NotFound

model_store = BentoMLContainer.model_store.get()
cloud_client = BentoMLContainer.bentocloud_client.get()

def initialize_model_from_huggingface() -> None:
    model= "sshleifer/distilbart-cnn-12-6"
    task = "summarization"

    bentoml.transformers.save_model(
        task,
        transformers.pipeline(task, model=model),
        metadata=dict(model_name=model),
    )

def get_model_tag(config_path: str = "bentofile.yaml") -> str:
    """
    Get the latest model tag from config
    """
    from bentoml._internal.bento.build_config import BentoBuildConfig
    with open(config_path, "r", encoding="utf-8") as f:
        build_config = BentoBuildConfig.from_yaml(f)

    return build_config.models[0].tag

def prepare_model_from_tag(tag: str) -> None:
    """
    Given a model tag, download the model from BentoCloud
    and prepare it for serving. 
    If it exist in local store, do nothing
    """
    try:
        model_store.get(tag)
        print(f"Model {tag} exist in local store")
    except NotFound as e:
        print(e)
        cloud_client.pull_model(tag=tag)

    return None