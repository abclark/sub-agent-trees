# Sub-Agent Trees

Hierarchical task decomposition for AI. Break complex problems into independent subtasks, run them in parallel across multiple AI providers, and synthesize results bottom-up.

## How it works

1. Define tasks as a DAG in a YAML file
2. Tasks with no dependencies run in parallel (Layer 0)
3. Tasks that depend on others wait for their inputs (Layer 1, 2, ...)
4. A shared context file syncs between layers
5. An optional merge agent produces the final output

```
          [Merge]
          /      \
   [Synthesis A] [Synthesis B]
    /    |    \    /    |    \
 [T1]  [T2]  [T3] [T4] [T5] [T6]
```

## Quick start

```bash
pip install anthropic openai google-generativeai pyyaml

# Set API keys
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...

# Run a plan
python orchestrator.py examples/open-plan-redesign.yaml

# Dry run (show structure without executing)
python orchestrator.py examples/open-plan-redesign.yaml --dry-run
```

## Plan format

```yaml
name: "My Project"
context_file: "./output/CONTEXT.md"   # Shared context, syncs between layers
output_dir: "./output"
default_provider: anthropic
default_model: claude-sonnet-4-6

tasks:
  - id: research
    prompt: "Research X and produce a summary."
    provider: openai        # Override provider per task
    model: gpt-5.2
    read_files:             # Files to include in prompt
      - ./docs/reference.md
    output_file: research.md

  - id: synthesis
    prompt: "Combine the research into a plan."
    depends_on: [research]  # Waits for research to complete
    output_file: plan.md

merge_prompt: "Produce the final deliverable from all outputs."
merge_provider: anthropic
merge_model: claude-sonnet-4-6
```

## Features

- **Multi-provider**: Anthropic, OpenAI, Google. Different models per task.
- **Automatic layering**: Topological sort into parallelizable layers.
- **Shared context**: Results from each layer append to a context file that subsequent layers read.
- **Dependency outputs**: Tasks automatically receive the full output of their dependencies.
- **File reading**: Tasks can read local files as reference material.
- **Merge agent**: Optional final synthesis across all outputs.
- **Dry run**: Preview the execution plan without API calls.
- **Manifest**: JSON summary of timing, costs, and outputs.

## Multi-model arbitrage

Different models excel at different tasks:

| Model | Strengths | Cost (per M tokens) |
|-------|-----------|-------------------|
| Claude Opus | Nuanced reasoning, long-form writing | $5/$15 |
| Claude Sonnet | Good balance of speed and quality | $3/$15 |
| GPT-5.2 | Structured output, schemas, tables | ~$5/$15 |
| GPT-5 nano | Ultra-cheap for simple tasks | $0.15/$0.40 |
| Gemini 3 Pro | Fast synthesis, good at conflicts | $2/$12 |
| Gemini Flash | Cheapest capable model | $0.50/$3 |

Assign expensive models to hard tasks, cheap models to straightforward ones.

## License

MIT
