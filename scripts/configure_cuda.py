#!/usr/bin/env python3
"""Configure pyproject.toml for a specific CUDA major version.

Usage: python scripts/configure_cuda.py <12|13>

Rewrites CUDA-version-specific dependencies in pyproject.toml:
  - PyTorch index URL and source references
  - flash-attn prebuilt wheel URL
  - vLLM wheel URL (cu130 GitHub wheel for CUDA 13, PyPI default for CUDA 12)
  - nvidia-cudnn package name (cu12 vs cu13)
  - onnxruntime-gpu index (nightly for CUDA 13, PyPI default for CUDA 12)
  - cuda-python version constraint
"""

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).parent.parent / "pyproject.toml"
OVA_SH = Path(__file__).parent.parent / "ova.sh"

# onnxruntime-gpu nightly index block (only needed for CUDA 13)
ORT_INDEX_BLOCK = """
[[tool.uv.index]]
name = "ort-cuda-13-nightly"
url = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/"
explicit = true
"""

# Per-CUDA-version configuration
CUDA_CONFIGS = {
    "12": {
        "pytorch_index_name": "pytorch-cu128",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu128",
        "flash_attn_wheel": (
            "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels"
            "/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp313-cp313-linux_x86_64.whl"
        ),
        "vllm_source": None,  # Use PyPI default (ships cu12)
        "cudnn_package": "nvidia-cudnn-cu12>=9.13.0",
        "ort_index": None,  # PyPI default is cu12 since onnxruntime-gpu 1.19.0
        "ort_source": None,  # No special source needed
        "cuda_python_req": "cuda-python>=12.0.0",
        "ova_sh_supported": "12",
        "ova_sh_pytorch_index": "cu128",
    },
    "13": {
        "pytorch_index_name": "pytorch-cu130",
        "pytorch_index_url": "https://download.pytorch.org/whl/cu130",
        "flash_attn_wheel": (
            "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels"
            "/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu130torch2.10-cp313-cp313-linux_x86_64.whl"
        ),
        "vllm_source": (
            'vllm = { url = "https://github.com/vllm-project/vllm/releases/download/'
            'v0.17.1/vllm-0.17.1%2Bcu130-cp38-abi3-manylinux_2_35_x86_64.whl" }'
        ),
        "cudnn_package": "nvidia-cudnn-cu13>=9.13.0",
        "ort_index": ORT_INDEX_BLOCK,
        "ort_source": 'onnxruntime-gpu = { index = "ort-cuda-13-nightly" }',
        "cuda_python_req": "cuda-python>=13.0.0",
        "ova_sh_supported": "13",
        "ova_sh_pytorch_index": "cu130",
    },
}


def configure_pyproject(cuda_major: str) -> None:
    cfg = CUDA_CONFIGS[cuda_major]
    content = PYPROJECT.read_text()

    # 1. cuda-python version
    content = re.sub(
        r'"cuda-python>=\d+\.\d+\.\d+"',
        f'"{cfg["cuda_python_req"]}"',
        content,
    )

    # 2. flash-attn wheel URL
    content = re.sub(
        r'"flash-attn @ https://github\.com/mjun0812/flash-attention-prebuild-wheels/releases/download/[^"]*"',
        f'"{cfg["flash_attn_wheel"]}"',
        content,
    )

    # 3. nvidia-cudnn override
    content = re.sub(
        r'"nvidia-cudnn-cu\d+>=[\d.]+"',
        f'"{cfg["cudnn_package"]}"',
        content,
    )

    # 4. PyTorch index name and URL
    content = re.sub(
        r'name = "pytorch-cu\d+"',
        f'name = "{cfg["pytorch_index_name"]}"',
        content,
    )
    content = re.sub(
        r'url = "https://download\.pytorch\.org/whl/cu\d+"',
        f'url = "{cfg["pytorch_index_url"]}"',
        content,
    )

    # 5. torch/torchaudio/torchvision source index references
    content = re.sub(
        r'(\{ index = ")pytorch-cu\d+(" \})',
        rf'\g<1>{cfg["pytorch_index_name"]}\g<2>',
        content,
    )

    # 6. onnxruntime-gpu index — remove existing, add if needed
    # Remove any existing ort index block
    content = re.sub(
        r'\n\[\[tool\.uv\.index\]\]\nname = "ort-cuda-\d+-nightly"\nurl = "https://aiinfra[^"]*"\n(explicit = true\n)?',
        '\n',
        content,
    )
    # Add ort index block if needed (CUDA 13)
    if cfg["ort_index"]:
        # Insert before [tool.uv.sources]
        content = content.replace(
            "\n[tool.uv.sources]",
            cfg["ort_index"] + "\n[tool.uv.sources]",
        )

    # 7. onnxruntime-gpu source — remove or set
    content = re.sub(r'^onnxruntime-gpu = \{[^\n]*\n', '', content, flags=re.MULTILINE)
    if cfg["ort_source"]:
        # Add before qwen-tts or vllm source line
        if "vllm = {" in content:
            content = content.replace("vllm = {", cfg["ort_source"] + "\nvllm = {")
        else:
            content = content.replace("qwen-tts =", cfg["ort_source"] + "\nqwen-tts =")

    # 8. vLLM source — add or remove
    if cfg["vllm_source"]:
        if re.search(r'^vllm = \{', content, re.MULTILINE):
            content = re.sub(
                r'^vllm = \{[^\n]*\n',
                cfg["vllm_source"] + "\n",
                content,
                flags=re.MULTILINE,
            )
        else:
            content = content.replace(
                "qwen-tts =",
                cfg["vllm_source"] + "\nqwen-tts =",
            )
    else:
        content = re.sub(r'^vllm = \{[^\n]*\n', '', content, flags=re.MULTILINE)

    PYPROJECT.write_text(content)
    print(f"Updated {PYPROJECT}")


def configure_ova_sh(cuda_major: str) -> None:
    cfg = CUDA_CONFIGS[cuda_major]
    content = OVA_SH.read_text()

    content = re.sub(
        r'SUPPORTED_CUDA="\d+"',
        f'SUPPORTED_CUDA="{cfg["ova_sh_supported"]}"',
        content,
    )
    content = re.sub(
        r'PYTORCH_INDEX_CUDA="cu\d+"',
        f'PYTORCH_INDEX_CUDA="{cfg["ova_sh_pytorch_index"]}"',
        content,
    )

    OVA_SH.write_text(content)
    print(f"Updated {OVA_SH}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in CUDA_CONFIGS:
        supported = ", ".join(CUDA_CONFIGS.keys())
        print(f"Usage: {sys.argv[0]} <{supported}>")
        sys.exit(1)

    cuda_major = sys.argv[1]
    print(f"Configuring for CUDA {cuda_major}...")
    configure_pyproject(cuda_major)
    configure_ova_sh(cuda_major)


if __name__ == "__main__":
    main()
