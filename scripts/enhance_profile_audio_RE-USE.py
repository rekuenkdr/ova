"""
Enhance voice profile reference audio using NVIDIA RE-USE (Universal Speech Enhancement).

Applies AI-powered denoising and optional bandwidth extension to ref_audio.wav
for better voice cloning quality.

Model: https://huggingface.co/nvidia/RE-USE

Usage:
    python scripts/enhance_profile_audio_RE-USE.py <profile_name> [--language <lang>] [--bwe <rate>]

Examples:
    python scripts/enhance_profile_audio_RE-USE.py myvoice                    # Denoise + BWE to 24kHz (default) /profiles/es/myvoice/
    python scripts/enhance_profile_audio_RE-USE.py myvoice --language en      # English profile /profiles/en/myvoice/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

# Ensure torch's bundled CUDA libs are on LD_LIBRARY_PATH (needed by mamba_ssm extensions)
_torch_lib = os.path.join(torch.__path__[0], "lib")
os.environ["LD_LIBRARY_PATH"] = _torch_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import soundfile as sf
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"

RELU = nn.ReLU()


def find_profile(name: str, language: str) -> Path:
    """Find profile directory."""
    profile_dir = PROFILES_DIR / language / name
    if not profile_dir.exists():
        for lang_dir in PROFILES_DIR.iterdir():
            candidate = lang_dir / name
            if candidate.exists():
                return candidate
        print(f"Error: Profile '{name}' not found in {PROFILES_DIR}")
        sys.exit(1)
    return profile_dir


def find_audio(profile_dir: Path) -> Path:
    """Find reference audio file (wav or ogg)."""
    for ext in ["wav", "ogg", "mp3", "m4a", "flac"]:
        audio = profile_dir / f"ref_audio.{ext}"
        if audio.exists():
            return audio
    print(f"Error: No ref_audio.* found in {profile_dir}")
    sys.exit(1)


def download_reuse_model() -> Path:
    """Download the RE-USE model from HuggingFace (uses default HF cache)."""
    from huggingface_hub import snapshot_download
    print("Ensuring NVIDIA RE-USE model is cached...")
    model_dir = Path(snapshot_download(repo_id="nvidia/RE-USE"))
    print(f"RE-USE model at {model_dir}")
    return model_dir


def make_even(value):
    value = int(round(value))
    return value if value % 2 == 0 else value + 1


def enhance_audio(audio: torch.Tensor, sr: int, model_dir: Path, device: torch.device, bwe: int | None = None) -> tuple[torch.Tensor, int]:
    """Run RE-USE speech enhancement on audio tensor.

    Args:
        audio: 1D float tensor of audio samples
        sr: Sample rate of the audio
        model_dir: Path to downloaded RE-USE model directory
        device: Torch device
        bwe: Optional bandwidth extension target sample rate

    Returns:
        (enhanced_audio, sample_rate) tuple
    """
    # Add RE-USE modules to path
    sys.path.insert(0, str(model_dir))
    from models.stfts import mag_phase_stft, mag_phase_istft
    from models.generator_SEMamba_time_d4 import SEMamba
    from utils.util import load_config, pad_or_trim_to_match
    sys.path.pop(0)

    # Load config and model
    config_path = model_dir / "recipes" / "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
    cfg = load_config(str(config_path))
    n_fft = cfg['stft_cfg']['n_fft']
    hop_size = cfg['stft_cfg']['hop_size']
    win_size = cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    sampling_rate = cfg['stft_cfg']['sampling_rate']

    checkpoint_path = model_dir / "exp" / "30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_peak_GAN_tel_mic" / "g_01134000.pth"

    print("  Loading RE-USE model...")
    model = SEMamba(cfg).to(device)
    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict['generator'])
    model.eval()

    # Optional bandwidth extension
    if bwe is not None and bwe != sr:
        import librosa
        print(f"  Bandwidth extension: {sr}Hz -> {bwe}Hz...")
        audio_np = librosa.resample(audio.cpu().numpy(), orig_sr=sr, target_sr=bwe, res_type="kaiser_best")
        audio = torch.FloatTensor(audio_np)
        sr = bwe

    # Prepare input: [1, samples]
    noisy_wav = audio.unsqueeze(0).to(device)

    # Scale STFT params to match input sample rate
    n_fft_scaled = make_even(n_fft * sr // sampling_rate)
    hop_size_scaled = make_even(hop_size * sr // sampling_rate)
    win_size_scaled = make_even(win_size * sr // sampling_rate)

    with torch.no_grad():
        noisy_mag, noisy_pha, noisy_com = mag_phase_stft(
            noisy_wav,
            n_fft=n_fft_scaled,
            hop_size=hop_size_scaled,
            win_size=win_size_scaled,
            compress_factor=compress_factor,
            center=True,
            addeps=False,
        )

        amp_g, pha_g, _ = model(noisy_mag, noisy_pha)

        # Remove sweep artifacts
        mag = torch.expm1(RELU(amp_g))
        zero_portion = torch.sum(mag == 0, 1) / mag.shape[1]
        amp_g[:, :, (zero_portion > 0.5)[0]] = 0

        audio_g = mag_phase_istft(amp_g, pha_g, n_fft_scaled, hop_size_scaled, win_size_scaled, compress_factor)
        audio_g = pad_or_trim_to_match(noisy_wav.detach(), audio_g, pad_value=1e-8)

    return audio_g.squeeze(0).cpu(), sr


def main():
    parser = argparse.ArgumentParser(description="Enhance voice profile reference audio (NVIDIA RE-USE)")
    parser.add_argument("profile", help="Profile name (e.g., eva, martina)")
    parser.add_argument("--language", "-l", default="es", help="Language folder (default: es)")
    parser.add_argument("--bwe", type=int, default=24000,
                        help="Bandwidth extension target sample rate in Hz (default: 24000 for Qwen3-TTS)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    profile_dir = find_profile(args.profile, args.language)
    audio_path = find_audio(profile_dir)
    output_path = profile_dir / "ref_audio.wav"
    backup_path = profile_dir / f"ref_audio_original{audio_path.suffix}"

    print(f"Profile:  {profile_dir}")
    print(f"Input:    {audio_path.name}")
    print(f"Device:   {args.device}")
    print(f"Mode:     {'denoise + BWE to ' + str(args.bwe) + 'Hz' if args.bwe else 'denoise only'}")
    print()

    # Load audio
    print("Loading audio...")
    data, sr = sf.read(str(audio_path), dtype="float32")
    dwav = torch.from_numpy(data)
    if dwav.dim() > 1:
        dwav = dwav.mean(dim=-1)
    dwav = dwav.squeeze()
    print(f"  Sample rate: {sr}Hz, Duration: {len(dwav)/sr:.1f}s, Samples: {len(dwav)}")

    # Download model if needed
    model_dir = download_reuse_model()

    # Enhance (use BWE to target 24kHz for Qwen3-TTS compatibility if needed)
    device = torch.device(args.device)
    bwe = args.bwe if args.bwe != sr else None
    print("\nEnhancing with RE-USE...")
    final_audio, final_sr = enhance_audio(dwav, sr, model_dir, device, bwe=bwe)
    print(f"  Done ({final_sr}Hz)")

    # Backup original
    if audio_path == output_path:
        if not backup_path.exists():
            print(f"\nBacking up original to {backup_path.name}")
            shutil.copy2(audio_path, backup_path)
    else:
        print(f"\nOriginal kept as {audio_path.name}")

    # Save
    audio_np = final_audio.cpu().numpy()
    sf.write(str(output_path), audio_np, final_sr)
    print(f"Saved:    {output_path.name} ({final_sr}Hz, {len(audio_np)/final_sr:.1f}s)")

    # Remove cached voice clone prompts to force regeneration
    removed = []
    for size in ["0.6B", "1.7B"]:
        cache_file = profile_dir / f"voice_clone_prompt_{size}.pt"
        if cache_file.exists():
            cache_file.unlink()
            removed.append(cache_file.name)
    if removed:
        print(f"Removed:  {', '.join(removed)} (will regenerate on next startup)")

    print("\nDone!")


if __name__ == "__main__":
    main()
