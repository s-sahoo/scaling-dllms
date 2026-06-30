"""Push a checkpoint to a Hugging Face repo."""

import argparse
import os
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push a checkpoint to HuggingFace")
    parser.add_argument("checkpoint", help="Path to the checkpoint file or folder")
    parser.add_argument("repo_id", help="HuggingFace repo ID (e.g. user/my-model)")
    parser.add_argument("--token", required=True, help="HuggingFace API token")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/{args.repo_id}")

    if os.path.isdir(args.checkpoint):
        api.upload_folder(folder_path=args.checkpoint, repo_id=args.repo_id, repo_type="model")
    else:
        api.upload_file(
            path_or_fileobj=args.checkpoint,
            path_in_repo=os.path.basename(args.checkpoint),
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"Done! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
