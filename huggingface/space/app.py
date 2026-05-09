import os
import textwrap

import gradio as gr
from huggingface_hub import HfApi


DEFAULT_MODEL_REPO = os.environ.get("BUFFERX_HF_MODEL_REPO", "")
EXPECTED_FILES = [
    "snapshot/threedmatch/Desc/best.pth",
    "snapshot/threedmatch/Pose/best.pth",
    "snapshot/kitti/Desc/best.pth",
    "snapshot/kitti/Pose/best.pth",
]


def check_model_repo(repo_id):
    repo_id = repo_id.strip()
    if not repo_id:
        return "Enter a Hugging Face model repo id."

    api = HfApi()
    try:
        files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    except Exception as exc:
        return f"Could not read `{repo_id}`: {exc}"

    present = [path for path in EXPECTED_FILES if path in files]
    missing = [path for path in EXPECTED_FILES if path not in files]

    lines = [f"Model repo: `{repo_id}`", ""]
    lines.append(f"Snapshot files found: {len(present)}/{len(EXPECTED_FILES)}")
    if present:
        lines.extend(["", "Found:"])
        lines.extend([f"- `{path}`" for path in present])
    if missing:
        lines.extend(["", "Missing:"])
        lines.extend([f"- `{path}`" for path in missing])
    if not missing:
        lines.extend(["", "The repo layout matches the BUFFER-X downloader."])
    return "\n".join(lines)


def build_commands(repo_id, cuda_channel):
    repo_id = repo_id.strip() or "<model-repo-id>"
    cuda_channel = cuda_channel.strip()
    commands = f"""
    git clone https://github.com/MIT-SPARK/BUFFER-X
    cd BUFFER-X
    ./scripts/install.sh --cuda {cuda_channel} --with-hub
    python scripts/download_pretrained_models.py --source hf --repo-id {repo_id}
    python test.py --dataset 3DMatch --experiment_id threedmatch --verbose
    """
    return textwrap.dedent(commands).strip()


with gr.Blocks(title="BUFFER-X Hub Helper") as demo:
    gr.Markdown("# BUFFER-X Hub Helper")
    with gr.Row():
        repo = gr.Textbox(
            label="Model repo",
            value=DEFAULT_MODEL_REPO,
            placeholder="org-or-user/BUFFER-X",
        )
        cuda = gr.Dropdown(
            label="CUDA channel",
            choices=["cu124", "cu118", "cu111", "cpu"],
            value="cu124",
        )
    with gr.Row():
        check_button = gr.Button("Check model repo", variant="primary")
        commands_button = gr.Button("Generate commands")
    repo_status = gr.Markdown()
    commands = gr.Code(label="Commands", language="bash")

    check_button.click(check_model_repo, inputs=repo, outputs=repo_status)
    commands_button.click(build_commands, inputs=[repo, cuda], outputs=commands)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
