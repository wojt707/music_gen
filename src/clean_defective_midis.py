#!/usr/bin/env python3
"""
clean_defective_midis.py
========================
Scans generated MIDI directories on $SCRATCH (or local sync), identifies defective
files (silent tracks with 0 notes, truncated tracks < 5s, or collapsed chords),
and deletes them so that `generate.py` can automatically regenerate only the missing ones.

Usage:
  python clean_defective_midis.py [--dir $SCRATCH/generated_midis] [--dry-run]
"""

import argparse
import os
import struct
from pathlib import Path


def inspect_midi(path: Path) -> tuple[bool, str, float, int, int]:
    """
    Checks if a MIDI file is valid and musical.
    Returns: (is_valid, reason, duration_sec, note_count, unique_onsets)
    """
    try:
        if not path.exists() or path.stat().st_size < 40:
            return False, "file too small / empty", 0.0, 0, 0
        data = path.read_bytes()
        if data[:4] != b"MThd":
            return False, "invalid MIDI header", 0.0, 0, 0

        tpb = struct.unpack(">H", data[12:14])[0] if len(data) >= 14 else 480
        if tpb <= 0:
            tpb = 480

        pos = 14
        notes = 0
        max_ticks = 0
        unique_onsets = set()
        tempos = []

        while pos + 8 <= len(data):
            if data[pos : pos + 4] != b"MTrk":
                break
            tlen = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
            trk = data[pos + 8 : pos + 8 + tlen]
            pos += 8 + tlen

            i = 0
            tt = 0
            status = 0
            n_trk = len(trk)
            while i < n_trk:
                v = 0
                while i < n_trk:
                    b = trk[i]
                    i += 1
                    v = (v << 7) | (b & 0x7F)
                    if not (b & 0x80):
                        break
                tt += v
                if i >= n_trk:
                    break
                b = trk[i]
                if b & 0x80:
                    status = b
                    i += 1
                if status == 0xFF:
                    if i >= n_trk:
                        break
                    mtype = trk[i]
                    i += 1
                    mlen = 0
                    while i < n_trk:
                        b2 = trk[i]
                        i += 1
                        mlen = (mlen << 7) | (b2 & 0x7F)
                        if not (b2 & 0x80):
                            break
                    if mtype == 0x51 and mlen == 3 and i + 3 <= n_trk:
                        tempo_val = (trk[i] << 16) | (trk[i + 1] << 8) | trk[i + 2]
                        tempos.append((tt, tempo_val))
                    i += mlen
                elif (status & 0xF0) in (0x80, 0x90, 0xA0, 0xB0, 0xE0):
                    if (status & 0xF0) == 0x90 and i + 1 < n_trk and trk[i + 1] > 0:
                        notes += 1
                        unique_onsets.add(tt)
                    i += 2
                elif (status & 0xF0) in (0xC0, 0xD0):
                    i += 1
                else:
                    i += 1
            if tt > max_ticks:
                max_ticks = tt

        if not tempos:
            duration_sec = (max_ticks / tpb) * 0.5
        else:
            tempos.sort(key=lambda x: x[0])
            duration_sec = 0.0
            cur_tick = 0
            cur_tempo = 500000
            for t_tick, t_tempo in tempos:
                if t_tick > cur_tick:
                    chunk = min(max_ticks, t_tick) - cur_tick
                    duration_sec += (chunk / tpb) * (cur_tempo / 1_000_000.0)
                    cur_tick = min(max_ticks, t_tick)
                    if cur_tick >= max_ticks:
                        break
                cur_tempo = t_tempo
            if max_ticks > cur_tick:
                duration_sec += ((max_ticks - cur_tick) / tpb) * (cur_tempo / 1_000_000.0)

        if notes == 0:
            return False, "0 notes (silent file)", duration_sec, notes, len(unique_onsets)
        if duration_sec < 5.0:
            return False, f"too short ({duration_sec:.1f}s < 5.0s)", duration_sec, notes, len(unique_onsets)
        if notes < 16:
            return False, f"too few notes ({notes} < 16)", duration_sec, notes, len(unique_onsets)
        if len(unique_onsets) < 8:
            return False, f"collapsed chord / too few onsets ({len(unique_onsets)} < 8)", duration_sec, notes, len(unique_onsets)

        return True, "valid", duration_sec, notes, len(unique_onsets)
    except Exception as e:
        return False, f"corrupted: {e}", 0.0, 0, 0


def main():
    parser = argparse.ArgumentParser(description="Find and remove defective MIDI tracks on Ares $SCRATCH.")
    parser.add_argument("--dir", type=str, default=None, help="Root folder to scan. Defaults to $SCRATCH/generated_midis.")
    parser.add_argument("--dry-run", action="store_true", help="Only report defective files without deleting.")
    args = parser.parse_args()

    root_dir = args.dir
    if not root_dir:
        scratch = os.getenv("SCRATCH")
        if scratch:
            root_dir = os.path.join(scratch, "generated_midis")
        else:
            repo_root = Path(__file__).resolve().parent.parent
            root_dir = str(repo_root / "sample_generated_midis")

    root = Path(root_dir)
    if not root.exists():
        print(f"Directory {root} does not exist!")
        return

    print("=" * 80)
    print(f"Scanning MIDI directory: {root}")
    print(f"Mode: {'DRY RUN (no deletions)' if args.dry_run else 'ACTIVE (deleting defective files)'}")
    print("=" * 80)

    total_scanned = 0
    total_valid = 0
    total_deleted = 0

    # Group files by model subfolder (e.g. gt_1000_no_ties/piano/llama)
    for model_dir in sorted(root.glob("*/*/*")):
        if not model_dir.is_dir():
            continue
        mid_files = list(model_dir.glob("**/*.mid"))
        if not mid_files:
            continue

        bad_in_model = []
        for f in mid_files:
            total_scanned += 1
            is_valid, reason, sec, notes, onsets = inspect_midi(f)
            if is_valid:
                total_valid += 1
            else:
                bad_in_model.append((f, reason, sec, notes, onsets))

        if bad_in_model:
            print(f"\n[{model_dir.relative_to(root)}] Total: {len(mid_files)} | Valid: {len(mid_files) - len(bad_in_model)} | Defective: {len(bad_in_model)}")
            sample_reasons = {}
            for f, reason, sec, notes, onsets in bad_in_model:
                sample_reasons[reason] = sample_reasons.get(reason, 0) + 1
                if not args.dry_run:
                    try:
                        f.unlink()
                        total_deleted += 1
                    except OSError as e:
                        print(f"  Error deleting {f.name}: {e}")
            for r, count in sample_reasons.items():
                print(f"   -> {r}: {count} files")
        else:
            print(f"[{model_dir.relative_to(root)}] All {len(mid_files)} files are 100% VALID.")

    print("\n" + "=" * 80)
    print(f"SUMMARY: Scanned={total_scanned} | Valid={total_valid} | Deleted/Defective={total_deleted if not args.dry_run else len(bad_in_model)}")
    print("=" * 80)
    if not args.dry_run:
        print("Now re-run `submit_generate_ares.sh` for the affected models.")
        print("`generate.py` will skip existing valid files and generate ONLY the missing ones!")


if __name__ == "__main__":
    main()

