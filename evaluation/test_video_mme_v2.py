#!/usr/bin/env python3
"""
Standalone Video-MME-v2 inference & evaluation with HuggingFace transformers.

Usage examples:
    # Basic (no subtitle, no reasoning)
    python test_video_mme_v2.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --nframe 64

    # With subtitle (all text concatenated as one block)
    python test_video_mme_v2.py --model ... --with-subtitle --subtitle-dir /path/to/jsonl/

    # With subtitle (interleaved between frames by timestamp)
    python test_video_mme_v2.py --model ... --with-subtitle --subtitle-interleave --subtitle-dir /path/to/jsonl/

    # Reasoning prompt
    python test_video_mme_v2.py --model ... --reasoning
"""

import argparse
import ast
import json
import os
import os.path as osp
import re
import warnings

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ──────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────
WO_SUB_PROMPT = 'These are the frames of a video.'
WITH_SUB_PROMPT = (
    'These are the frames of a video. '
    "This video's subtitles are listed below:\n{}\n"
)
WITH_SUB_PROMPT_INTERLEAVE = (
    'These are the frames of a video with corresponding subtitles shown between frames. '
    'The subtitles indicate what is being said during the time interval between adjacent frames.'
)
THINK_PROMPT = (
    'Please perform a detailed reasoning based on the provided video frames to answer the following '
    'multiple-choice question selecting the best option from A through H and providing your final response '
    "strictly in the format: 'Final Answer: <letter>."
)
INSTRUCT_PROMPT = (
    'Select the best answer to the following multiple-choice question based on the video. '
    'Respond with only the letter (A, B, C, D, E, F, G, or H) of the correct option.'
)


# ──────────────────────────────────────────────
# Frame extraction  (saves to cache dir, returns file paths)
# ──────────────────────────────────────────────
def extract_frames(video_path, nframe=0, fps=-1, cache_dir=None):
    """
    Extract frames from a video, save to cache, return paths + metadata.
    Returns: (list[str] frame_paths, list[int] indices, dict video_info)
    """
    import decord
    vid = decord.VideoReader(video_path)
    video_info = {
        'fps': vid.get_avg_fps(),
        'n_frames': len(vid),
    }
    if nframe > 0 and fps <= 0:
        step_size = len(vid) / (nframe + 1)
        indices = [int(i * step_size) for i in range(1, nframe + 1)]
    elif fps > 0:
        total_duration = video_info['n_frames'] / video_info['fps']
        required_frames = int(total_duration * fps)
        step_size = video_info['fps'] / fps
        indices = [int(i * step_size) for i in range(required_frames)]
    else:
        raise ValueError('Either nframe or fps must be set to a positive value')

    # Build cache sub-dir per video
    vid_name = osp.splitext(osp.basename(video_path))[0]
    if cache_dir is None:
        cache_dir = '/tmp/videommev2_frames'
    frame_dir = osp.join(cache_dir, vid_name)
    os.makedirs(frame_dir, exist_ok=True)

    tag = f'n{nframe}' if nframe > 0 else f'fps{fps}'
    frame_paths = [osp.join(frame_dir, f'frame_{tag}_{i:04d}.jpg') for i in range(len(indices))]

    # Only extract if not all cached
    if not all(osp.exists(p) for p in frame_paths):
        images = [vid[i].asnumpy() for i in indices]
        for arr, p in zip(images, frame_paths):
            if not osp.exists(p):
                Image.fromarray(arr).save(p)

    return frame_paths, indices, video_info


# ──────────────────────────────────────────────
# Subtitle helpers  (JSONL with word-level timestamps)
# ──────────────────────────────────────────────
def load_subtitle_jsonl(subtitle_path):
    """Load JSONL file. Each line: {"text": str, "start_time": float, "end_time": float}."""
    if not osp.exists(subtitle_path):
        return None
    entries = []
    with open(subtitle_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def subtitle_concat_all(entries):
    """Concatenate all word texts into a single string."""
    if not entries:
        return ''
    return ' '.join(e['text'] for e in entries)


def subtitle_between_timestamps(entries, start_time, end_time):
    """Collect words whose time range overlaps [start_time, end_time)."""
    if not entries:
        return ''
    words = []
    for e in entries:
        if e['end_time'] >= start_time and e['start_time'] < end_time:
            words.append(e['text'])
    return ' '.join(words)


def group_subtitle_segments(entries, gap_threshold=0.5):
    """Group word-level JSONL entries into sentence-level segments.

    A new segment starts when either:
      - The time gap between consecutive words exceeds *gap_threshold* seconds, or
      - The previous word ends with sentence-ending punctuation (. ! ?) **and**
        there is a non-trivial time gap (> 0.1 s).

    Returns a list of dicts: [{"text": str, "start_time": float, "end_time": float}, ...].
    """
    if not entries:
        return []

    segments = []
    current_words = [entries[0]]

    for i in range(1, len(entries)):
        prev = entries[i - 1]
        curr = entries[i]
        time_gap = curr['start_time'] - prev['end_time']
        prev_ends_sentence = prev['text'].rstrip().endswith(('.', '!', '?'))

        if time_gap > gap_threshold or (prev_ends_sentence and time_gap > 0.1):
            segments.append({
                'text': ' '.join(w['text'] for w in current_words),
                'start_time': current_words[0]['start_time'],
                'end_time': current_words[-1]['end_time'],
            })
            current_words = [curr]
        else:
            current_words.append(curr)

    if current_words:
        segments.append({
            'text': ' '.join(w['text'] for w in current_words),
            'start_time': current_words[0]['start_time'],
            'end_time': current_words[-1]['end_time'],
        })

    return segments


def segments_between_timestamps(segments, start_time, end_time):
    """Return subtitle segments whose time range overlaps [start_time, end_time)."""
    result = []
    for seg in segments:
        if seg['end_time'] >= start_time and seg['start_time'] < end_time:
            result.append(seg)
    return result


# ──────────────────────────────────────────────
# Prompt building  (returns transformers chat messages)
# ──────────────────────────────────────────────
def build_chat_messages(row, video_dir, subtitle_dir, nframe, fps,
                        with_subtitle, subtitle_interleave, reasoning,
                        cache_dir=None):
    """
    Build a single chat-template messages list.
    Images are referenced via local file paths.
    Returns: (messages, n_frames)
    """
    video_path = osp.join(video_dir, str(row['video']) + '.mp4')
    frame_paths, indices, video_info = extract_frames(
        video_path, nframe=nframe, fps=fps, cache_dir=cache_dir)

    vid_fps = video_info['fps']
    frame_timestamps = [idx / vid_fps for idx in indices]

    # Load subtitle JSONL if needed
    sub_entries = None
    if with_subtitle:
        sub_path = osp.join(subtitle_dir, str(row['video']) + '.jsonl')
        sub_entries = load_subtitle_jsonl(sub_path)

    content = []

    if with_subtitle and subtitle_interleave:
        # ── Interleave mode: image → subtitle segments → image → ... ──
        # Group word-level entries into sentence-level segments using JSONL timestamps.
        segments = group_subtitle_segments(sub_entries)

        for i, (fp, frame_ts) in enumerate(zip(frame_paths, frame_timestamps)):
            if i < len(frame_timestamps) - 1:
                end_ts = frame_timestamps[i + 1]
            else:
                end_ts = video_info['n_frames'] / vid_fps

            content.append({'type': 'image', 'path': fp})

            matched = segments_between_timestamps(segments, frame_ts, end_ts)
            for seg in matched:
                content.append({
                    'type': 'text',
                    'text': f'[Subtitle {seg["start_time"]:.2f}s - {seg["end_time"]:.2f}s]: {seg["text"]}',
                })

        text_prompt = WITH_SUB_PROMPT_INTERLEAVE

    else:
        # ── Non-interleave: all frames first ──
        for fp in frame_paths:
            content.append({'type': 'image', 'path': fp})

        if with_subtitle:
            full_text = subtitle_concat_all(sub_entries)
            text_prompt = WITH_SUB_PROMPT.format(full_text)
        else:
            text_prompt = WO_SUB_PROMPT

    content.append({'type': 'text', 'text': text_prompt})

    response_prompt = THINK_PROMPT if reasoning else INSTRUCT_PROMPT
    question_text = str(row['question']) + '\n' + str(row['options'])
    content.append({'type': 'text', 'text': f'Question: {question_text}\n{response_prompt}'})

    messages = [{'role': 'user', 'content': content}]
    return messages


# ──────────────────────────────────────────────
# Answer extraction
# ──────────────────────────────────────────────
def extract_characters_regex_v2(s):
    """Extract answer letter A-H from model response."""
    s = s.strip()
    answer_prefixes = [
        'Final Answer:',
        'The best answer is',
        'The correct answer is',
        'The answer is',
        'The answer',
        'The best option is',
        'The correct option is',
        'Best answer:',
        'Best option:',
        'Answer:',
        'Option:',
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, '')
    if len(s.split()) > 10 and not re.search('[A-H]', s):
        return ''
    matches = re.search(r'[A-H]', s)
    if matches is None:
        return ''
    return matches[0]


# ──────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────
def cal_relevance(scores):
    score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    correct_count = sum(scores)
    return score_map.get(correct_count, 0.0), correct_count * 25.0


def cal_logic(scores, group_structure):
    group_structure_list = ast.literal_eval(group_structure)
    last_correct_idx = -1
    for idx, val in enumerate(scores):
        if val:
            last_correct_idx = idx
        else:
            break
    if group_structure_list == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 16, 2: 100.0 * 4 / 16, 3: 100.0 * 9 / 16, 4: 100.0}
    elif group_structure_list == [1, [2, 3], 4]:
        score_map = {0: 0.0, 1: 100.0 / 12, 2: 100.0 * 4 / 12, 3: 100.0 * 7 / 12, 4: 100.0}
        if last_correct_idx == 0 and scores[2]:
            last_correct_idx += 1
    elif group_structure_list == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 100.0 / 10, 2: 100.0 * 2 / 10, 3: 100.0 * 5 / 10, 4: 100.0}
        if last_correct_idx == -1 and scores[1]:
            last_correct_idx += 1
    else:
        raise ValueError(f'Unknown group_structure_list: {group_structure_list}')
    return score_map.get(last_correct_idx + 1, 0.0)


def get_final_rating(data):
    """Compute final rating from scored DataFrame."""
    all_groups = [[] for _ in range((len(data) + 1) // 4)]
    final_rating = {
        'level_1': [], 'level_2': [], 'level_3': [],
        'relevance_score': [], 'relevance_linear_score': [],
        'logic_score': [], 'total': [],
    }
    second_head_rating = {}
    third_head_rating = {}

    for i in range(len(data)):
        level = data.iloc[i]['level']
        group_type = data.iloc[i]['group_type']
        group_structure = data.iloc[i]['group_structure']
        score = data.iloc[i]['score']
        second_head = data.iloc[i]['second_head']
        third_head = data.iloc[i]['third_head']
        all_groups[i // 4].append((level, group_type, group_structure, score, second_head, third_head))

    for group in all_groups:
        level = group[-1][0]
        group_type = group[-1][1]
        group_structure = group[-1][2]
        second_head = group[-1][4]
        third_head = group[-1][5]
        scores = [item[3] for item in group]

        if group_type == 'relevance':
            exp_score, linear_score = cal_relevance(scores)
            final_rating['relevance_score'].append(exp_score)
            final_rating['relevance_linear_score'].append(linear_score)
        elif group_type == 'logic':
            exp_score = cal_logic(scores, group_structure)
            final_rating['logic_score'].append(exp_score)
        else:
            raise ValueError(f'Unknown group_type: {group_type}')

        if level is not None and str(level) != 'None':
            final_rating[f'level_{int(level)}'].append(exp_score)
        final_rating['total'].append(exp_score)

        if second_head not in second_head_rating:
            second_head_rating[second_head] = []
        second_head_rating[second_head].append(exp_score)
        if third_head not in third_head_rating:
            third_head_rating[third_head] = []
        third_head_rating[third_head].append(exp_score)

    for key in final_rating:
        vals = final_rating[key]
        final_rating[key] = sum(vals) / len(vals) if vals else 0.0
    for key in second_head_rating:
        vals = second_head_rating[key]
        second_head_rating[key] = sum(vals) / len(vals) if vals else 0.0
    for key in third_head_rating:
        vals = third_head_rating[key]
        third_head_rating[key] = sum(vals) / len(vals) if vals else 0.0

    return {
        'final_rating': final_rating,
        'second_head_rating': second_head_rating,
        'third_head_rating': third_head_rating,
    }


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(data, output_path):
    """Score predictions and compute final rating."""
    n_total = len(data)
    n_empty = data['prediction'].isna().sum() + (data['prediction'] == '').sum()

    data['score'] = -1
    for i in range(len(data)):
        ans = str(data.iloc[i]['answer'])
        pred = str(data.iloc[i].get('prediction', ''))
        extracted = extract_characters_regex_v2(pred)
        data.at[i, 'score'] = int(extracted == ans) if extracted else -1

    n_failed = (data['score'] == -1).sum()
    print(f'Total: {n_total}, no prediction: {n_empty}, failed to extract: {n_failed - n_empty}')

    valid = data[data['score'] >= 0]
    if len(valid) > 0:
        simple_acc = valid['score'].mean() * 100
        print(f'Simple accuracy (valid only): {simple_acc:.2f}%')

    rating = get_final_rating(data)

    fr = rating['final_rating']
    print(f"\n{'Metric':<30} {'Score':>8}")
    print('-' * 40)
    for k, v in fr.items():
        print(f'{k:<30} {v:>8.2f}')

    sh = rating['second_head_rating']
    if any(str(v) != 'None' for v in sh.keys()):
        print(f"\n{'Second Head':<30} {'Score':>8}")
        print('-' * 40)
        for k, v in sh.items():
            print(f'{str(k):<30} {v:>8.2f}')

    th = rating['third_head_rating']
    if any(str(v) != 'None' for v in th.keys()):
        print(f"\n{'Third Head':<30} {'Score':>8}")
        print('-' * 40)
        for k, v in th.items():
            print(f'{str(k):<30} {v:>8.2f}')

    rating_path = output_path.replace('.tsv', '_rating.json')
    with open(rating_path, 'w') as f:
        json.dump(rating, f, indent=2, default=str)
    print(f'\nRating saved to {rating_path}')

    score_path = output_path.replace('.tsv', '_scored.tsv')
    data.to_csv(score_path, sep='\t', index=False)
    print(f'Scored data saved to {score_path}')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def _save_results(df, predictions, output_path):
    out = df.copy()
    out['prediction'] = out['index'].apply(lambda idx: predictions.get(int(idx), ''))
    out.to_csv(output_path, sep='\t', index=False)


def main():
    parser = argparse.ArgumentParser(description='Video-MME-v2 inference & eval with transformers')
    # Data
    parser.add_argument('--parquet', type=str, default='/home/dongyuhao/test.parquet')
    parser.add_argument('--video-dir', type=str, default='/xiaoxinghu/video_mme_final/videos_original')
    parser.add_argument('--subtitle-dir', type=str, default=None,
                        help='Directory containing per-video JSONL subtitle files (e.g. 001.jsonl). '
                             'Required when --with-subtitle is set.')
    parser.add_argument('--frame-cache-dir', type=str, default='/tmp/videommev2_frames',
                        help='Directory to cache extracted frames')
    parser.add_argument('--output', type=str, default=None)
    # Sampling
    parser.add_argument('--nframe', type=int, default=64)
    parser.add_argument('--fps', type=float, default=-1)
    # Switches
    parser.add_argument('--with-subtitle', action='store_true')
    parser.add_argument('--subtitle-interleave', action='store_true')
    parser.add_argument('--reasoning', action='store_true')
    # Model
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or local path, e.g. Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--attn-impl', type=str, default='sdpa',
                        choices=['sdpa', 'flash_attention_2', 'eager'])
    parser.add_argument('--min-pixels', type=int, default=None,
                        help='Min pixels for image resizing (e.g. 200704 = 256*28*28)')
    parser.add_argument('--max-pixels', type=int, default=None,
                        help='Max pixels for image resizing (e.g. 1003520 = 1280*28*28)')
    # Generation
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=None,
                        help='Top-p (nucleus) sampling threshold. Only used when temperature > 0.')
    parser.add_argument('--repetition-penalty', type=float, default=None,
                        help='Repetition penalty (1.0 = no penalty). Values > 1.0 discourage repetition.')
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    args = parser.parse_args()

    if args.subtitle_interleave and not args.with_subtitle:
        parser.error('--subtitle-interleave requires --with-subtitle')
    if args.with_subtitle and args.subtitle_dir is None:
        parser.error('--subtitle-dir is required when --with-subtitle is set')

    # ── Load data ──
    df = pd.read_parquet(args.parquet)
    df = df.assign(index=range(len(df)))
    df['video'] = df['video_id'].apply(str)

    # ── Output path ──
    if args.output is None:
        tag = osp.basename(args.model.rstrip('/'))
        suffix_parts = [f'{args.nframe}frame' if args.nframe > 0 else f'{args.fps}fps']
        if args.with_subtitle:
            suffix_parts.append('subs_interleave' if args.subtitle_interleave else 'subs')
        if args.reasoning:
            suffix_parts.append('reasoning')
        args.output = f'Video-MME-v2_{tag}_{"_".join(suffix_parts)}.tsv'

    # ── Resume support ──
    predictions = {}
    if osp.exists(args.output):
        existing = pd.read_csv(args.output, sep='\t')
        if 'prediction' in existing.columns:
            for _, r in existing.iterrows():
                if pd.notna(r.get('prediction', None)) and str(r['prediction']).strip():
                    predictions[int(r['index'])] = str(r['prediction'])
        print(f'Resuming: loaded {len(predictions)} existing predictions from {args.output}')

    todo_indices = [i for i in range(len(df)) if int(df.iloc[i]['index']) not in predictions]
    if not todo_indices:
        print('All predictions already done, skipping to evaluation.')
    else:
        # ── Load model & processor ──
        from transformers import AutoProcessor, AutoModelForImageTextToText

        dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
        torch_dtype = dtype_map[args.dtype]

        print(f'Loading model: {args.model}  dtype={args.dtype}  attn={args.attn_impl}')
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map='auto',
            attn_implementation=args.attn_impl,
            trust_remote_code=True,
        )
        model.eval()

        processor_kwargs = {}
        if args.min_pixels is not None:
            processor_kwargs['min_pixels'] = args.min_pixels
        if args.max_pixels is not None:
            processor_kwargs['max_pixels'] = args.max_pixels
        processor = AutoProcessor.from_pretrained(
            args.model, trust_remote_code=True, **processor_kwargs)

        print(f'Running inference on {len(todo_indices)} / {len(df)} questions ...')
        print(f'  Frames     : nframe={args.nframe}, fps={args.fps}')
        print(f'  Subtitle   : {args.with_subtitle} (interleave={args.subtitle_interleave})')
        if args.with_subtitle:
            print(f'  Subtitle dir: {args.subtitle_dir}')
        print(f'  Reasoning  : {args.reasoning}')
        print(f'  Output     : {args.output}')
        print()

        # ── Inference loop (one at a time) ──
        for cnt, i in enumerate(tqdm(todo_indices, desc='Inference')):
            row = df.iloc[i]
            data_idx = int(row['index'])

            try:
                messages = build_chat_messages(
                    row, args.video_dir, args.subtitle_dir,
                    nframe=args.nframe, fps=args.fps,
                    with_subtitle=args.with_subtitle,
                    subtitle_interleave=args.subtitle_interleave,
                    reasoning=args.reasoning,
                    cache_dir=args.frame_cache_dir,
                )

                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors='pt',
                ).to(model.device)

                with torch.inference_mode():
                    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)
                    if args.temperature > 0:
                        gen_kwargs['temperature'] = args.temperature
                        gen_kwargs['do_sample'] = True
                        if args.top_p is not None:
                            gen_kwargs['top_p'] = args.top_p
                    else:
                        gen_kwargs['do_sample'] = False
                    if args.repetition_penalty is not None:
                        gen_kwargs['repetition_penalty'] = args.repetition_penalty

                    output_ids = model.generate(**inputs, **gen_kwargs)

                # Trim input tokens from output
                generated_ids = output_ids[0, inputs['input_ids'].shape[1]:]
                reply = processor.decode(generated_ids, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=False).strip()

            except Exception as e:
                print(f'  [idx={data_idx}] Error: {e}')
                reply = ''

            predictions[data_idx] = reply

            # Periodic save every 50 questions
            if (cnt + 1) % 50 == 0:
                _save_results(df, predictions, args.output)

        _save_results(df, predictions, args.output)
        print(f'\nAll predictions saved to {args.output}')

    # ── Evaluate ──
    print('\n── Evaluation ──')
    result_df = pd.read_csv(args.output, sep='\t')
    evaluate(result_df, args.output)


if __name__ == '__main__':
    main()
