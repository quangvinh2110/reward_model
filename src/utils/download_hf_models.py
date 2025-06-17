import os
import argparse
import logging
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_model(model_name: str, output_dir: str, cache_dir: str = None):
    """
    Download a Hugging Face model and its tokenizer to the specified directory.

    Args:
        model_name (str): Name of the model on Hugging Face Hub
        output_dir (str): Directory to save the model
        cache_dir (str, optional): Directory to cache the downloaded model
    """
    try:
        logger.info(f"Downloading model {model_name} to {output_dir}")

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Download model and tokenizer
        model = AutoModel.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=False
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, local_files_only=False
        )

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"Successfully downloaded and saved model to {output_dir}")

    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Hugging Face models")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model on Hugging Face Hub",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the model"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the downloaded model",
    )

    args = parser.parse_args()
    download_model(args.model_name, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
