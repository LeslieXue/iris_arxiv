import argparse
import json
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from iris_llm import CandidateForScoring, TeacherScore


def parse_args():
    default_config = Path(__file__).with_name("config.toml")

    parser = argparse.ArgumentParser(
        description="Score exported IRIS chunks with a teacher LLM"
    )
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to the LLM reranking experiment configuration",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Teacher-input JSONL; defaults to output_dir/teacher_inputs.jsonl",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Score JSONL; defaults to output_dir/qwen_scores.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N unprocessed chunks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing score, error, and metadata files",
    )
    return parser.parse_args()


def load_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return repo_root / path


def record_key(record) -> tuple[int, int, int]:
    return (
        int(record.document_id),
        int(record.query_id),
        int(record.chunk_id),
    )


def output_key(payload: dict) -> tuple[int, int, int]:
    return (
        int(payload["document_id"]),
        int(payload["query_id"]),
        int(payload["chunk_id"]),
    )


def read_completed_keys(path: Path) -> set[tuple[int, int, int]]:
    if not path.exists():
        return set()

    completed = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
                completed.add(output_key(payload))
            except Exception as exc:
                raise ValueError(
                    f"Could not read existing score at "
                    f"{path}:{line_number}: {exc}"
                ) from exc

    return completed


def load_records(
    path: Path,
    completed: set[tuple[int, int, int]],
    limit: int | None,
) -> list[CandidateForScoring]:
    records = []

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                record = CandidateForScoring.model_validate_json(line)
            except Exception as exc:
                raise ValueError(
                    f"Invalid teacher input at {path}:{line_number}: {exc}"
                ) from exc

            if record_key(record) in completed:
                continue

            records.append(record)

            if limit is not None and len(records) >= limit:
                break

    return records


def batches(
    records: list[CandidateForScoring],
    batch_size: int,
) -> Iterable[list[CandidateForScoring]]:
    for start in range(0, len(records), batch_size):
        yield records[start : start + batch_size]


def build_prompt(template: str, chunk_text: str) -> str:
    placeholder = "{{CHUNK_TEXT}}"

    if template.count(placeholder) != 1:
        raise ValueError(
            f"Prompt must contain exactly one {placeholder} placeholder"
        )

    return template.replace(placeholder, chunk_text)


def parse_teacher_output(
    raw_output: str,
    record: CandidateForScoring,
) -> TeacherScore:
    cleaned = raw_output.strip()

    if not cleaned.startswith("{") or not cleaned.endswith("}"):
        raise ValueError("Teacher output is not exactly one JSON object")

    payload = json.loads(cleaned)

    if not isinstance(payload, dict):
        raise TypeError("Teacher output must decode to a JSON object")

    payload.update(
        {
            "document_id": record.document_id,
            "query_id": record.query_id,
            "chunk_id": record.chunk_id,
        }
    )

    score = TeacherScore.model_validate(payload)

    if not score.brief_reason.strip():
        raise ValueError("brief_reason must not be empty")

    if len(score.brief_reason.split()) > 40:
        raise ValueError("brief_reason contains more than 40 words")

    return score


def write_jsonl(handle, payload: dict) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def main():
    args = parse_args()

    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[2]
    config_path = resolve_repo_path(repo_root, str(args.config))
    experiment_cfg = load_toml(config_path)

    output_dir = resolve_repo_path(
        repo_root,
        experiment_cfg["paths"]["output_dir"],
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = (
        resolve_repo_path(repo_root, args.input)
        if args.input
        else output_dir / "teacher_inputs.jsonl"
    )
    output_path = (
        resolve_repo_path(repo_root, args.output)
        if args.output
        else output_dir / "qwen_scores.jsonl"
    )
    error_path = output_path.with_suffix(".errors.jsonl")
    metadata_path = output_path.with_suffix(".metadata.json")

    teacher_cfg = experiment_cfg["teacher"]
    prompt_path = (
        config_path.parent
        / "prompts"
        / f"{teacher_cfg['prompt_version']}.txt"
    )
    prompt_template = prompt_path.read_text(encoding="utf-8")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")

    if args.overwrite:
        for path in (output_path, error_path, metadata_path):
            path.unlink(missing_ok=True)

    completed = read_completed_keys(output_path)
    records = load_records(
        path=input_path,
        completed=completed,
        limit=args.limit,
    )

    print(f"Model: {teacher_cfg['model_id']}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Already completed: {len(completed)}")
    print(f"Chunks to score: {len(records)}")

    if not records:
        print("No unprocessed chunks remain.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        teacher_cfg["model_id"],
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        teacher_cfg["model_id"],
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    model_device = next(model.parameters()).device
    print(f"Model device: {model_device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    generation_kwargs = {
        "max_new_tokens": int(teacher_cfg["max_new_tokens"]),
        "do_sample": bool(teacher_cfg["do_sample"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(
            teacher_cfg["temperature"]
        )

    success_count = 0
    failure_count = 0

    with (
        output_path.open("a", encoding="utf-8") as score_handle,
        error_path.open("a", encoding="utf-8") as error_handle,
        torch.inference_mode(),
    ):
        for batch_number, batch in enumerate(
            batches(records, int(teacher_cfg["batch_size"])),
            start=1,
        ):
            prompts = [
                build_prompt(prompt_template, record.chunk_text)
                for record in batch
            ]

            chat_messages = [
                [{"role": "user", "content": prompt}]
                for prompt in prompts
            ]

            rendered_prompts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(
                        teacher_cfg["enable_thinking"]
                    ),
                )
                for messages in chat_messages
            ]

            inputs = tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {
                name: tensor.to(model_device)
                for name, tensor in inputs.items()
            }

            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )

            prompt_width = inputs["input_ids"].shape[1]
            generated_tokens = generated[:, prompt_width:]

            raw_outputs = tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
            )

            for record, raw_output in zip(
                batch,
                raw_outputs,
                strict=True,
            ):
                try:
                    score = parse_teacher_output(
                        raw_output=raw_output,
                        record=record,
                    )

                    write_jsonl(
                        score_handle,
                        {
                            **score.model_dump(),
                            "raw_output": raw_output.strip(),
                        },
                    )
                    success_count += 1

                except Exception as exc:
                    write_jsonl(
                        error_handle,
                        {
                            "document_id": record.document_id,
                            "query_id": record.query_id,
                            "chunk_id": record.chunk_id,
                            "error": str(exc),
                            "raw_output": raw_output,
                        },
                    )
                    failure_count += 1

            processed = success_count + failure_count
            print(
                f"Batch {batch_number}: "
                f"processed={processed}/{len(records)}, "
                f"succeeded={success_count}, "
                f"failed={failure_count}"
            )

    model_commit = getattr(model.config, "_commit_hash", None)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": teacher_cfg["model_id"],
        "model_commit": model_commit,
        "prompt_version": teacher_cfg["prompt_version"],
        "enable_thinking": bool(teacher_cfg["enable_thinking"]),
        "do_sample": bool(teacher_cfg["do_sample"]),
        "max_new_tokens": int(teacher_cfg["max_new_tokens"]),
        "batch_size": int(teacher_cfg["batch_size"]),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "requested_records": len(records),
        "successful_records": success_count,
        "failed_records": failure_count,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(model_device),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Scores: {output_path}")
    print(f"Errors: {error_path}")
    print(f"Metadata: {metadata_path}")

    if failure_count:
        raise RuntimeError(
            f"{failure_count} teacher outputs failed validation; "
            f"inspect {error_path}"
        )

    print("Teacher scoring completed successfully.")


if __name__ == "__main__":
    main()