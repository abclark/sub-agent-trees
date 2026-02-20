#!/usr/bin/env python3
"""
Sub-Agent Trees: Hierarchical task decomposition for AI.

Define tasks as a DAG. Run leaves in parallel across providers.
Shared context syncs between rounds. Merge agent resolves conflicts.

Usage:
    python orchestrator.py plan.yaml
    python orchestrator.py plan.yaml --dry-run
    python orchestrator.py plan.yaml --output-dir ./results
"""

import argparse
import asyncio
import json
import os
import sys
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# --- Provider Clients ---

class Provider:
    """Base class for AI providers."""
    async def complete(self, system: str, prompt: str, model: str) -> str:
        raise NotImplementedError


class AnthropicProvider(Provider):
    def __init__(self):
        import anthropic
        self.client = anthropic.AsyncAnthropic()

    async def complete(self, system: str, prompt: str, model: str) -> str:
        response = await self.client.messages.create(
            model=model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class OpenAIProvider(Provider):
    def __init__(self):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()

    async def complete(self, system: str, prompt: str, model: str) -> str:
        response = await self.client.chat.completions.create(
            model=model,
            max_tokens=8192,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class GoogleProvider(Provider):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
        self._genai = genai

    async def complete(self, system: str, prompt: str, model: str) -> str:
        gmodel = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=system,
        )
        response = await asyncio.to_thread(
            gmodel.generate_content, prompt
        )
        return response.text


PROVIDERS = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}

def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()


# --- Data Structures ---

@dataclass
class Task:
    id: str
    prompt: str
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-6"
    depends_on: list[str] = field(default_factory=list)
    read_files: list[str] = field(default_factory=list)
    output_file: Optional[str] = None
    # Populated at runtime
    result: Optional[str] = None
    elapsed: float = 0.0
    error: Optional[str] = None


@dataclass
class Plan:
    name: str
    context_file: Optional[str] = None
    output_dir: str = "./output"
    tasks: list[Task] = field(default_factory=list)
    merge_prompt: Optional[str] = None
    merge_provider: str = "anthropic"
    merge_model: str = "claude-sonnet-4-6"


# --- Core Engine ---

class Orchestrator:
    def __init__(self, plan: Plan, dry_run: bool = False):
        self.plan = plan
        self.dry_run = dry_run
        self.providers: dict[str, Provider] = {}
        self.context = ""
        self.task_map: dict[str, Task] = {t.id: t for t in plan.tasks}
        self.output_dir = Path(plan.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_provider(self, name: str) -> Provider:
        if name not in self.providers:
            self.providers[name] = get_provider(name)
        return self.providers[name]

    def _build_layers(self) -> list[list[Task]]:
        """Topological sort into parallelizable layers."""
        completed = set()
        remaining = list(self.plan.tasks)
        layers = []

        while remaining:
            # Find tasks whose dependencies are all completed
            ready = [t for t in remaining if all(d in completed for d in t.depends_on)]
            if not ready:
                unresolved = [t.id for t in remaining]
                raise ValueError(f"Circular dependency or missing tasks: {unresolved}")
            layers.append(ready)
            for t in ready:
                completed.add(t.id)
            remaining = [t for t in remaining if t.id not in completed]

        return layers

    def _load_context(self) -> str:
        if self.plan.context_file and os.path.exists(self.plan.context_file):
            return Path(self.plan.context_file).read_text()
        return ""

    def _read_files(self, paths: list[str]) -> str:
        parts = []
        for p in paths:
            p = os.path.expanduser(p)
            if os.path.exists(p):
                content = Path(p).read_text()
                parts.append(f"--- {p} ---\n{content}")
            else:
                parts.append(f"--- {p} --- [FILE NOT FOUND]")
        return "\n\n".join(parts)

    def _build_prompt(self, task: Task) -> str:
        parts = []

        # Shared context
        if self.context:
            parts.append(f"## Shared Context\n\n{self.context}")

        # Dependency outputs
        for dep_id in task.depends_on:
            dep = self.task_map[dep_id]
            if dep.result:
                parts.append(f"## Output from '{dep_id}'\n\n{dep.result}")

        # File contents
        if task.read_files:
            parts.append(f"## Reference Files\n\n{self._read_files(task.read_files)}")

        # Task prompt
        parts.append(f"## Your Task\n\n{task.prompt}")

        return "\n\n".join(parts)

    async def _run_task(self, task: Task) -> None:
        prompt = self._build_prompt(task)
        system = (
            "You are an expert AI agent completing a specific subtask. "
            "Be thorough and concrete. Produce actionable output, not vague suggestions."
        )

        if self.dry_run:
            print(f"  [DRY RUN] {task.id} ({task.provider}/{task.model})")
            print(f"    Depends on: {task.depends_on or 'none'}")
            print(f"    Read files: {task.read_files or 'none'}")
            print(f"    Prompt length: {len(prompt)} chars")
            task.result = f"[DRY RUN output for {task.id}]"
            return

        print(f"  ▶ {task.id} ({task.provider}/{task.model})...", end="", flush=True)
        start = time.time()

        try:
            provider = self._get_provider(task.provider)
            task.result = await provider.complete(system, prompt, task.model)
            task.elapsed = time.time() - start
            print(f" ✅ {task.elapsed:.1f}s ({len(task.result)} chars)")
        except Exception as e:
            task.elapsed = time.time() - start
            task.error = str(e)
            task.result = f"[ERROR: {e}]"
            print(f" ❌ {task.elapsed:.1f}s ({e})")

        # Save output
        if task.output_file:
            out_path = self.output_dir / task.output_file
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(task.result)

    async def _run_layer(self, layer: list[Task]) -> None:
        await asyncio.gather(*[self._run_task(t) for t in layer])

    def _update_context(self, layer: list[Task]) -> None:
        """Append layer results to shared context."""
        additions = []
        for t in layer:
            if t.result and not t.error:
                # Truncate to avoid context explosion
                summary = t.result[:2000]
                if len(t.result) > 2000:
                    summary += f"\n[... truncated, full output: {len(t.result)} chars]"
                additions.append(f"### {t.id}\n{summary}")

        if additions:
            self.context += "\n\n## Round Results\n\n" + "\n\n".join(additions)

        # Write context file
        if self.plan.context_file:
            Path(self.plan.context_file).write_text(self.context)

    async def _run_merge(self) -> Optional[str]:
        """Run merge agent on all task outputs."""
        if not self.plan.merge_prompt:
            return None

        all_outputs = []
        for t in self.plan.tasks:
            if t.result and not t.error:
                all_outputs.append(f"## {t.id}\n\n{t.result}")

        prompt = (
            f"## All Agent Outputs\n\n{''.join(all_outputs)}\n\n"
            f"## Merge Task\n\n{self.plan.merge_prompt}"
        )
        system = (
            "You are a synthesis agent. Read all outputs from parallel agents "
            "and produce a single unified result. Resolve conflicts. Be concrete."
        )

        if self.dry_run:
            print(f"  [DRY RUN] merge ({self.plan.merge_provider}/{self.plan.merge_model})")
            return "[DRY RUN merge output]"

        print(f"  ▶ merge ({self.plan.merge_provider}/{self.plan.merge_model})...", end="", flush=True)
        start = time.time()

        try:
            provider = self._get_provider(self.plan.merge_provider)
            result = await provider.complete(system, prompt, self.plan.merge_model)
            elapsed = time.time() - start
            print(f" ✅ {elapsed:.1f}s ({len(result)} chars)")

            merge_path = self.output_dir / "MERGED.md"
            merge_path.write_text(result)
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f" ❌ {elapsed:.1f}s ({e})")
            return None

    async def run(self) -> dict:
        print(f"\n🌳 {self.plan.name}")
        print(f"   {len(self.plan.tasks)} tasks, output → {self.plan.output_dir}\n")

        # Load initial context
        self.context = self._load_context()

        # Build layers
        layers = self._build_layers()
        print(f"   {len(layers)} layers:")
        for i, layer in enumerate(layers):
            ids = [t.id for t in layer]
            print(f"     Layer {i}: {ids}")
        print()

        # Execute layers
        total_start = time.time()
        for i, layer in enumerate(layers):
            print(f"  Layer {i} ({len(layer)} tasks in parallel):")
            await self._run_layer(layer)
            self._update_context(layer)
            print()

        # Merge
        merge_result = None
        if self.plan.merge_prompt:
            print("  Merge:")
            merge_result = await self._run_merge()
            print()

        total_elapsed = time.time() - total_start

        # Summary
        print(f"  ── Summary ──")
        print(f"  Total time: {total_elapsed:.1f}s")
        for t in self.plan.tasks:
            status = "✅" if not t.error else "❌"
            print(f"    {status} {t.id}: {t.elapsed:.1f}s, {len(t.result or '')} chars")
        if merge_result:
            print(f"    ✅ merge: {len(merge_result)} chars")
        print(f"  Output: {self.output_dir}/")
        print()

        # Save manifest
        manifest = {
            "name": self.plan.name,
            "total_seconds": total_elapsed,
            "tasks": [
                {
                    "id": t.id,
                    "provider": t.provider,
                    "model": t.model,
                    "elapsed": t.elapsed,
                    "chars": len(t.result or ""),
                    "error": t.error,
                    "output_file": t.output_file,
                }
                for t in self.plan.tasks
            ],
        }
        (self.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        return manifest


# --- Plan Loader ---

def load_plan(path: str, output_dir: Optional[str] = None) -> Plan:
    with open(path) as f:
        raw = yaml.safe_load(f)

    tasks = []
    for t in raw.get("tasks", []):
        tasks.append(Task(
            id=t["id"],
            prompt=t["prompt"],
            provider=t.get("provider", raw.get("default_provider", "anthropic")),
            model=t.get("model", raw.get("default_model", "claude-sonnet-4-6")),
            depends_on=t.get("depends_on", []),
            read_files=t.get("read_files", []),
            output_file=t.get("output_file"),
        ))

    return Plan(
        name=raw.get("name", "Unnamed Plan"),
        context_file=raw.get("context_file"),
        output_dir=output_dir or raw.get("output_dir", "./output"),
        tasks=tasks,
        merge_prompt=raw.get("merge_prompt"),
        merge_provider=raw.get("merge_provider", raw.get("default_provider", "anthropic")),
        merge_model=raw.get("merge_model", raw.get("default_model", "claude-sonnet-4-6")),
    )


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(description="Sub-Agent Trees: Hierarchical AI task decomposition")
    parser.add_argument("plan", help="Path to YAML plan file")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    plan = load_plan(args.plan, args.output_dir)
    orchestrator = Orchestrator(plan, dry_run=args.dry_run)
    asyncio.run(orchestrator.run())


if __name__ == "__main__":
    main()
