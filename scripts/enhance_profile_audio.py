"""
Enhance voice profile reference audio using Resemble Enhance.

Applies AI-powered denoising and optional enhancement to ref_audio.wav
for better voice cloning quality.

Usage:
    python scripts/enhance_profile_audio.py <profile_name> [--language <lang>] [--enhance] [--nfe <n>]

Examples:
    python scripts/enhance_profile_audio.py eva                    # Denoise only (recommended)
    python scripts/enhance_profile_audio.py eva --enhance          # Denoise + enhance (44.1kHz upscale)
    python scripts/enhance_profile_audio.py eva --language en      # English profile
    python scripts/enhance_profile_audio.py eva --enhance --nfe 64 # Higher quality enhancement
"""

import argparse
import shutil
import sys
from pathlib import Path

import soundfile as sf
import torch
import torchaudio


PROJECT_ROOT = Path(__file__).parent.parent
PROFILES_DIR = PROJECT_ROOT / "profiles"


def find_profile(name: str, language: str) -> Path:
    """Find profile directory."""
    profile_dir = PROFILES_DIR / language / name
    if not profile_dir.exists():
        # Search all languages
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


def main():
    parser = argparse.ArgumentParser(description="Enhance voice profile reference audio")
    parser.add_argument("profile", help="Profile name (e.g., eva, martina)")
    parser.add_argument("--language", "-l", default="es", help="Language folder (default: es)")
    parser.add_argument("--enhance", "-e", action="store_true",
                        help="Apply enhancement (upscale to 44.1kHz) in addition to denoising")
    parser.add_argument("--nfe", type=int, default=32,
                        help="Number of function evaluations for enhancement (default: 32, higher = better quality)")
    parser.add_argument("--lambd", type=float, default=0.5,
                        help="Enhancement blending parameter 0-1 (default: 0.5)")
    parser.add_argument("--device", default="cuda:0", help="Device (default: cuda:0)")
    args = parser.parse_args()

    profile_dir = find_profile(args.profile, args.language)
    audio_path = find_audio(profile_dir)
    output_path = profile_dir / "ref_audio.wav"
    backup_path = profile_dir / f"ref_audio_original{audio_path.suffix}"

    print(f"Profile:  {profile_dir}")
    print(f"Input:    {audio_path.name}")
    print(f"Device:   {args.device}")
    print(f"Mode:     {'denoise + enhance' if args.enhance else 'denoise only'}")
    print()

    # Load audio
    print("Loading audio...")
    data, sr = sf.read(str(audio_path), dtype="float32")
    dwav = torch.from_numpy(data)
    # Convert to mono if stereo
    if dwav.dim() > 1:
        dwav = dwav.mean(dim=-1)
    # Ensure 1D tensor
    dwav = dwav.squeeze()
    print(f"  Sample rate: {sr}Hz, Duration: {len(dwav)/sr:.1f}s, Samples: {len(dwav)}")

    # Import resemble-enhance
    from resemble_enhance.enhancer.inference import denoise, enhance

    device = torch.device(args.device)

    # Denoise
    print("\nDenoising...")
    dwav_denoised, new_sr = denoise(dwav, sr, device)
    print(f"  Done ({new_sr}Hz)")

    if args.enhance:
        # Enhance (upscales to 44.1kHz)
        print(f"\nEnhancing (nfe={args.nfe}, lambd={args.lambd})...")
        dwav_enhanced, new_sr = enhance(dwav_denoised, new_sr, device, nfe=args.nfe, lambd=args.lambd)
        final_audio = dwav_enhanced
        final_sr = new_sr
        print(f"  Done ({final_sr}Hz)")
    else:
        final_audio = dwav_denoised
        final_sr = new_sr

    # Resample to 24kHz for Qwen3-TTS compatibility
    if final_sr != 24000:
        print(f"\nResampling {final_sr}Hz -> 24000Hz...")
        final_audio = torchaudio.functional.resample(final_audio, final_sr, 24000)
        final_sr = 24000

    # Backup original
    if audio_path == output_path:
        if not backup_path.exists():
            print(f"\nBacking up original to {backup_path.name}")
            shutil.copy2(audio_path, backup_path)
    else:
        # Original is ogg/mp3/etc, keep it as-is
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
