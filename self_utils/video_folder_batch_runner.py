"""Run one existing video entry point over every video in a folder."""

import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time


VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv', '.webm',
    '.mpeg', '.mpg', '.ts', '.mts', '.m2ts',
}


def _videos(folder, recursive):
    iterator = folder.rglob('*') if recursive else folder.iterdir()
    return sorted(
        (path for path in iterator
         if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def run_folder(source_script, description=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--input', '--input_folder', dest='input_folder', required=True,
        help='folder containing input videos')
    parser.add_argument(
        '--output', default='./output',
        help='parent output directory (default: ./output)')
    parser.add_argument(
        '--eval_after', '--eval-after', action='store_true',
        help='run batch LLM evaluation on this batch output after inference')
    parser.add_argument(
        '--eval_skip_existing', '--eval-skip-existing', action='store_true',
        help='with --eval-after, skip videos whose metrics already exist')
    parser.add_argument(
        '--eval_force_llm', '--eval-force-llm', action='store_true',
        help='with --eval-after, regenerate existing LLM annotations')
    parser.add_argument(
        '--recursive', action='store_true',
        help='also search video files in subdirectories')
    parser.add_argument(
        '--continue_on_error', action=argparse.BooleanOptionalAction,
        default=True,
        help='continue with remaining videos after a failure (default: true)')
    args, forwarded = parser.parse_known_args()

    input_folder = Path(args.input_folder).expanduser().resolve()
    if not input_folder.is_dir():
        parser.error('input folder does not exist: {}'.format(input_folder))

    source_path = Path(source_script).resolve()
    if not source_path.is_file():
        parser.error('source script does not exist: {}'.format(source_path))

    videos = _videos(input_folder, args.recursive)
    if not videos:
        parser.error('no supported video files found in: {}'.format(input_folder))

    # The group directory is named after the inference script, not this
    # *_folder.py wrapper.
    group_dir = Path(args.output).expanduser().resolve() / source_path.stem
    group_dir.mkdir(parents=True, exist_ok=True)
    summary_path = group_dir / 'batch_summary.csv'

    print('=> inference script : {}'.format(source_path.name))
    print('=> input folder     : {}'.format(input_folder))
    print('=> videos found     : {}'.format(len(videos)))
    print('=> group output     : {}'.format(group_dir))

    rows = []
    total_started = time.perf_counter()
    for index, video in enumerate(videos, start=1):
        print('\n' + '=' * 72)
        print('=> [{}/{}] {}'.format(index, len(videos), video))
        print('=' * 72)
        command = [
            sys.executable,
            str(source_path),
            '--input', str(video),
            '--output', str(group_dir),
        ] + forwarded
        started = time.perf_counter()
        result = subprocess.run(command, cwd=str(source_path.parent))
        elapsed = time.perf_counter() - started
        status = 'success' if result.returncode == 0 else 'failed'
        rows.append({
            'index': index,
            'video': str(video),
            'status': status,
            'return_code': result.returncode,
            'elapsed_seconds': '{:.3f}'.format(elapsed),
        })

        # Rewrite after every video so interrupted batches retain progress.
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print('=> [{}] cost: {:.2f}s, return code: {}'.format(
            status, elapsed, result.returncode))
        if result.returncode != 0 and not args.continue_on_error:
            break

    success_count = sum(row['status'] == 'success' for row in rows)
    print('\n=> batch finished: {}/{} successful, total cost: {:.2f}s'.format(
        success_count, len(rows), time.perf_counter() - total_started))
    print('=> batch summary : {}'.format(summary_path))
    print('=> results root  : {}'.format(group_dir))
    eval_script = source_path.parent / 'run_pipeline_batch_llm_eval.py'
    eval_command = [
        sys.executable, str(eval_script), '--input-dir', str(group_dir)]
    if args.eval_skip_existing:
        eval_command.append('--skip-existing')
    if args.eval_force_llm:
        eval_command.append('--force-llm')

    print('=> evaluation cmd: {}'.format(subprocess.list2cmdline(eval_command)))
    eval_return_code = 0
    if args.eval_after:
        if not eval_script.is_file():
            print('=> evaluation script not found: {}'.format(eval_script))
            eval_return_code = 1
        else:
            print('\n=> starting batch LLM evaluation')
            eval_return_code = subprocess.run(
                eval_command, cwd=str(source_path.parent)).returncode

    inference_ok = success_count == len(rows)
    return 0 if inference_ok and eval_return_code == 0 else 1
