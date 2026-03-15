#!/bin/bash
# Patch qwen-asr for vLLM 0.17.x compatibility
# qwen-asr 0.0.6 targets vLLM 0.14.0 and uses APIs removed/moved in 0.16+:
#   1. MMEncoderAttention no longer accepts multimodal_config kwarg
#   2. get_vit_attn_backend no longer accepts attn_backend_override kwarg
#   3. _get_data_parser moved from MultiModalProcessor to ProcessingInfo.get_data_parser

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/../.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Error: .venv not found at $VENV_DIR"
    exit 1
fi

PYTHON_VERSION=$("$VENV_DIR/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
TARGET="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/qwen_asr/core/vllm_backend/qwen3_asr.py"

if [[ ! -f "$TARGET" ]]; then
    echo "Error: qwen3_asr.py not found at $TARGET"
    exit 1
fi

# Check if already patched (look for our marker comment)
if grep -q "vLLM 0.17+: attn_backend_override removed" "$TARGET" 2>/dev/null; then
    echo "Patch already applied"
    exit 0
fi

echo "Patching qwen-asr for vLLM 0.17.x compatibility..."
echo "Target: $TARGET"

# Create backup
cp "$TARGET" "${TARGET}.orig"

# 1. Remove multimodal_config= from MMEncoderAttention constructor call
sed -i '/self\.attn = MMEncoderAttention(/,/)/ {
    /multimodal_config=multimodal_config,/d
}' "$TARGET"

# 2. Replace attn_backend_override block with simplified version
python3 -c "
import re
with open('$TARGET', 'r') as f:
    content = f.read()

# Replace the attn_backend_override block
old = '''        # Get attention backend
        attn_backend_override = (
            multimodal_config.mm_encoder_attn_backend
            if multimodal_config is not None
            else None
        )
        self.attn_backend = get_vit_attn_backend(
            head_size=config.d_model // config.encoder_attention_heads,
            dtype=torch.get_default_dtype(),
            attn_backend_override=attn_backend_override,
        )'''

new = '''        # Get attention backend (vLLM 0.17+: attn_backend_override removed)
        self.attn_backend = get_vit_attn_backend(
            head_size=config.d_model // config.encoder_attention_heads,
            dtype=torch.get_default_dtype(),
        )'''

content = content.replace(old, new)

# 3. Add get_data_parser to Qwen3ASRProcessingInfo
old_info = '''    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {\"audio\": None}


class Qwen3ASRDummyInputsBuilder'''

new_info = '''    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {\"audio\": None}

    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_feature_extractor()
        return Qwen3ASRMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
        )


class Qwen3ASRDummyInputsBuilder'''

content = content.replace(old_info, new_info)

# 4. Remove _get_data_parser from Qwen3ASRMultiModalProcessor
old_proc = '''class Qwen3ASRMultiModalProcessor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
):
    def _get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.info.get_feature_extractor()
        return Qwen3ASRMultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
        )

    def _get_mm_fields_config('''

new_proc = '''class Qwen3ASRMultiModalProcessor(
    Qwen3OmniMoeThinkerMultiModalProcessor,
):
    def _get_mm_fields_config('''

content = content.replace(old_proc, new_proc)

with open('$TARGET', 'w') as f:
    f.write(content)
"

# 5. Fix model registry double-registration warning
# vLLM 0.17+ already ships Qwen3ASRForConditionalGeneration natively.
# Skip qwen-asr's registration to avoid "already registered" warning.
INFERENCE_TARGET="$VENV_DIR/lib/python${PYTHON_VERSION}/site-packages/qwen_asr/inference/qwen3_asr.py"
if [[ -f "$INFERENCE_TARGET" ]]; then
    python3 -c "
with open('$INFERENCE_TARGET', 'r') as f:
    content = f.read()

old_reg = '''try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    ModelRegistry.register_model(\"Qwen3ASRForConditionalGeneration\", Qwen3ASRForConditionalGeneration)
except:
    pass'''

new_reg = '''try:
    from qwen_asr.core.vllm_backend import Qwen3ASRForConditionalGeneration
    from vllm import ModelRegistry
    # vLLM 0.17+ has built-in Qwen3ASR support; only register if missing
    if \"Qwen3ASRForConditionalGeneration\" not in ModelRegistry.models:
        ModelRegistry.register_model(\"Qwen3ASRForConditionalGeneration\", Qwen3ASRForConditionalGeneration)
except:
    pass'''

content = content.replace(old_reg, new_reg)

with open('$INFERENCE_TARGET', 'w') as f:
    f.write(content)
"
fi

echo "Patch applied successfully!"
echo ""
echo "Changes made for vLLM 0.17.x compatibility:"
echo "  - Removed multimodal_config from MMEncoderAttention()"
echo "  - Removed attn_backend_override from get_vit_attn_backend()"
echo "  - Moved _get_data_parser to Qwen3ASRProcessingInfo.get_data_parser()"
echo "  - Skip model registry when vLLM already has Qwen3ASR built-in"
