# RAG Input to EPIC Agent - Complete Guide

## Overview

The RAG (Retrieval Augmented Generation) system processes uploaded PRD documents and additional documentation, then supplies a carefully curated and intelligently structured input to the EPIC Agent. This document explains exactly what the EPIC Agent receives as input.

## RAG Processing Pipeline

### 1. Document Processing
- **DOCX Files**: Automatically extracted using python-docx library
- **Text Files**: Processed with multiple encoding detection (UTF-8, Latin-1, etc.)
- **Binary Content**: Fallback extraction of printable characters
- **Vector Storage**: Documents are chunked and stored in ChromaDB with embeddings

### 2. RAG Summary Generation

The RAG system creates intelligent summaries by:

#### A. Semantic Queries
The system queries the vector database with these specific topics:
- "requirements and functional specifications"
- "user stories and acceptance criteria"
- "business objectives and goals"
- "system constraints and dependencies"
- "key features and capabilities"
- "technical architecture and design"
- "data flow and integration requirements"
- "security and compliance requirements"
- "performance and scalability requirements"
- "user interface and user experience requirements"

#### B. Content Retrieval
- For each query, retrieves top 2 most relevant document chunks
- Uses semantic similarity with SentenceTransformers embeddings
- Prioritizes content based on relevance scores

#### C. Summary Assembly
- Combines retrieved chunks into coherent sections
- Orders sections by importance (user stories and requirements get higher priority)
- Truncates to optimal length (15,000 characters for PRD, 8,000 for additional docs)

## Final Input Structure to EPIC Agent

The EPIC Agent receives this exact structured input:

```
{user_context}

RAG-Enhanced PRD Analysis:
{prd_summary}

Additional Context:
{docs_summary}

Instructions: The above content has been intelligently extracted and summarized using RAG (Retrieval Augmented Generation). 
It contains the most relevant requirements, user stories, and business objectives from the original documents.
Use this curated information to generate comprehensive epics and user stories.
```

## Example of RAG-Enhanced PRD Analysis Section

Based on the semantic queries, the PRD Analysis typically contains:

### [Requirements And Functional Specifications]
```
[Extracted content from the most relevant chunks about requirements]
- Specific functional requirements
- System behaviors and rules
- Input/output specifications
```

### [User Stories And Acceptance Criteria]
```
[Most relevant user story content from the document]
- As a [user type], I want [functionality] so that [benefit]
- Acceptance criteria with clear success metrics
- Edge cases and validation rules
```

### [Business Objectives And Goals]
```
[Key business drivers and success metrics]
- Primary business goals
- Success criteria and KPIs
- Market objectives and competitive advantages
```

### [Key Features And Capabilities]
```
[Core feature descriptions and capabilities]
- Main feature sets
- Technical capabilities
- Integration points
```

### [System Constraints And Dependencies]
```
[Technical and business constraints]
- Technology limitations
- Resource constraints
- External dependencies
- Compliance requirements
```

## RAG Advantages over Traditional PRD Parser

### 1. **Intelligent Content Extraction**
- Semantic understanding instead of simple text processing
- Relevant content prioritization
- Automatic topic categorization

### 2. **Performance Optimization**
- 50%+ faster processing (single agent vs. two agents)
- Reduced token usage and costs
- Smart content chunking and retrieval

### 3. **Quality Enhancement**
- Context-aware content selection
- Elimination of irrelevant information
- Structured topic organization

## Token and Length Management

### Input Optimization
- **Maximum Context**: 128k tokens (GPT-4o limit)
- **PRD Summary**: Up to 15,000 characters
- **Additional Docs**: Up to 8,000 characters
- **Total Enhanced Context**: Usually 20,000-30,000 characters
- **Token Count**: Typically 5,000-8,000 tokens (well within limits)

### Fallback Mechanisms
- If vector DB fails: Intelligent keyword-based extraction
- If content too large: Smart truncation with content preservation
- If token limit approached: Progressive content optimization

## Content Quality Assurance

### Valid Content Detection
- Minimum 50 characters for processing
- Error message filtering (encoding issues, corrupt files)
- Content type validation (text vs. binary)

### RAG Query Results
- Each semantic query returns top 2 most relevant chunks
- Chunks ranked by cosine similarity scores
- Content deduplicated and organized by topic

## Real Processing Example

When you upload a PRD document, here's what happens:

1. **Document Upload**: DOCX file detected and text extracted
2. **Vector Storage**: Content split into ~1000 character chunks with 200 character overlap
3. **Semantic Processing**: 10 different queries extract relevant content
4. **Summary Generation**: Most relevant chunks combined into structured summary
5. **Epic Agent Input**: Complete enhanced context with instructions

## Debugging and Monitoring

The system provides extensive logging to track:
- Document processing success/failure
- RAG summary generation details
- Token usage and optimization
- Content quality metrics
- Processing time and performance stats

## Integration with Epic Generator

The Epic Agent receives this curated input and uses it to generate:
- Comprehensive epics with clear objectives
- Detailed user stories with acceptance criteria
- Technical requirements and constraints
- Implementation priorities and dependencies

This RAG-enhanced approach ensures the Epic Agent gets the most relevant, well-structured, and actionable content from your PRD documents, resulting in higher quality epic and user story generation.
