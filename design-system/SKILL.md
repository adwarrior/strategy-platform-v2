---
name: strategy-platform-design
description: Use this skill to generate well-branded interfaces and assets for the Strategy Platform — a dark-themed quantitative trading backtesting and optimization dashboard. Contains essential design guidelines, colors, typography, spacing tokens, and a full dashboard UI kit for prototyping or production work.
user-invocable: true
---

Read the README.md file within this skill, and explore the other available files.

If creating visual artifacts (slides, mocks, throwaway prototypes, etc), copy assets out and create static HTML files for the user to view. If working on production code, you can copy assets and read the rules here to become an expert in designing with this brand.

If the user invokes this skill without any other guidance, ask them what they want to build or design, ask some questions, and act as an expert designer who outputs HTML artifacts _or_ production code, depending on the need.

Key design principles for Strategy Platform:
- Dark terminal aesthetic: charcoal/near-black backgrounds (not pure black), amber (#f0a830 equiv) primary accent, teal secondary
- Monospace (JetBrains Mono) for all numeric data, metric values, timestamps, parameter values
- No gradients, no illustrations, no emoji decoration — functional and precise
- Profit = green (oklch 68% 0.19 145), Loss = red (oklch 60% 0.22 25), neutral = white
- Run type prefixes: OPT (optimization), BAC (backtest), BAC-OOS (backtest from OOS), AR (autoresearch)
- Tone: direct, technical, imperative labels, no marketing fluff
