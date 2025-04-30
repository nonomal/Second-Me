# 📚 Wiki Entity & Relation Extraction Pipeline

This repository contains a modular pipeline for **entity extraction**, **entity scoring**, and **relation building** using large language models (LLMs), aiming to replace GraphRAG for knowledge graph construction from unstructured text.

The system is designed with extensibility in mind, allowing users to customize prompts and logic for richer entity and relationship discovery.

---

## 🧩 Overview

This pipeline provides the following core functionalities:

- **Entity Extraction**: Identify key entities from input text.
- **Entity Scoring**: Evaluate and filter extracted entities based on relevance or importance.
- **Wiki Generation**: Generate concise wiki-style summaries for each entity.
- **Relation Building**: Construct relationships between identified entities.

---

## 📁 Directory Structure

```
wiki/
├── base.py               # Core classes and workflow orchestration
├── build_relation.py     # Logic for constructing relationships between entities
├── prompt.py             # Prompt templates used across the pipeline
├── utils.py              # Utility functions shared among modules
└── README.md             # Document for wiki pipeline

```

---

## 📄 Script Descriptions

### 1. `base.py`

Implements core components of the pipeline as classes:

- **Entity Extractor**: Uses LLMs to extract candidate entities.
- **Entity Scorer**: Filters and ranks entities based on criteria like relevance or frequency.
- **Wiki Generator**: Creates structured wiki content for each selected entity.

> This file serves as the central orchestrator connecting all stages of the pipeline.

---

### 2. `build_relation.py`

Constructs relationships between the extracted entities based on context and semantic connections.

- Supports customizable logic for relation types.
- Allows users to define domain-specific relationship templates.
- Can be extended to integrate external knowledge sources.

> This script assumes that entities have been successfully extracted and scored.

---

### 3. `prompt.py`

Contains all prompt templates used throughout the pipeline:

- Entity extraction prompts
- Entity scoring instructions
- Wiki generation templates
- Relation construction queries

> Users are encouraged to modify these prompts to improve entity richness or introduce new relation types. Changes should be reflected in corresponding processing logic.

---

### 4. `utils.py`

Provides reusable utility functions used across the pipeline:

- Text preprocessing
- JSON handling
- Logging helpers
- Prompt formatting utilities

> Designed to keep core scripts clean and focused on business logic.

---

## 🔧 Customization Guide

Since this pipeline heavily relies on LLM-based reasoning:

- **To extract more entities**: Revise prompts in `prompt.py` and adjust filtering thresholds in `base.py`.
- **To discover richer relations**: Extend `prompt.py` with new relation templates and update logic in `build_relation.py`.
- **To support new domains**: Customize entity scoring logic and add domain-specific prompt variations.

---

## 🧠 Tips for Extending

- Always validate changes in prompts by testing against a representative dataset.
- Consider adding caching mechanisms when experimenting with expensive LLM calls.
- For better performance, you may integrate vector similarity checks for entity deduplication or relation validation.

---

## 📌 License & Attribution

Please refer to the main project root for license information and third-party attributions.

--- 