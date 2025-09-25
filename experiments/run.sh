#!/usr/bin/env bash

uv run --extra cuda modality-llm generate csv modal_verbs.jsonl dataset-v1-gen.csv --gen-include-alternatives --format csv --removal-backend api --removal-model openai/gpt-oss-20b --require-consensus both --removal-concurrency 64
uv run --extra cuda modality-llm generate judge dataset-v1-gen.csv dataset-v1-judge.csv --judge-model openai/gpt-oss-20b --judge-concurrency 64
uv run --extra cuda modality-llm generate repair dataset-v1-judge.csv dataset-v1-repair.csv --repair-model openai/gpt-oss-20b --repair-concurrency 64
uv run --extra cuda modality-llm generate judge dataset-v1-repair.csv dataset-v1-repair-rejudged.csv --judge-model openai/gpt-oss-20b --judge-concurrency 64
uv run --extra cuda modality-llm generate repair dataset-v1-repair-rejudged.csv dataset-v1-repair-rejudged-repaired.csv --repair-model openai/gpt-oss-20b --repair-concurrency 64
