# RAG to EPIC Agent Input Flow - Visual Diagram

```
┌─────────────────────┐
│   Document Upload   │
│   (.docx, .txt)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Document Processing │
│  • DOCX extraction  │
│  • Encoding detect  │
│  • Content validate │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Vector Storage    │
│  • Chunk content    │
│  • Generate embeds  │
│  • Store in ChromaDB│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Semantic Queries   │
│  • Requirements     │
│  • User Stories     │
│  • Business Goals   │
│  • Features         │
│  • Architecture     │
│  • Constraints      │
│  • Security         │
│  • Performance      │
│  • UI/UX            │
│  • Integration      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Content Retrieval  │
│  • Top 2 chunks     │
│  • Per query topic  │
│  • Relevance ranked │
│  • Cosine similarity│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Summary Assembly   │
│  • Topic sections   │
│  • Priority ordering│
│  • Length optimize  │
│  • Structure format │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Enhanced Context  │
│                     │
│ {user_context}      │
│                     │
│ RAG-Enhanced PRD:   │
│ [Requirements...]   │
│ [User Stories...]   │
│ [Business Goals...] │
│ [Features...]       │
│ [Architecture...]   │
│ [Constraints...]    │
│                     │
│ Additional Context: │
│ {docs_summary}      │
│                     │
│ Instructions:       │
│ Use this curated... │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    EPIC AGENT       │
│  • Receives struct │
│  • Processes RAG    │
│  • Generates epics  │
│  • Creates stories  │
└─────────────────────┘
```

## Input Structure Details

### User Context Section
```
{user_context}
```
- Any additional context provided by the user
- Business background, constraints, special requirements
- Usually empty or contains specific project context

### RAG-Enhanced PRD Analysis
```
[Requirements And Functional Specifications]
• Extracted functional requirements from the document
• System behaviors and business rules
• Input/output specifications and data handling

[User Stories And Acceptance Criteria]
• As a [role], I want [feature] so that [benefit]
• Detailed acceptance criteria with success metrics
• Edge cases and validation requirements

[Business Objectives And Goals]
• Primary business drivers and success criteria
• Market positioning and competitive advantages
• KPIs and measurable outcomes

[Key Features And Capabilities]
• Core feature descriptions and technical capabilities
• Integration points and system interfaces
• Feature priorities and dependencies

[System Constraints And Dependencies]
• Technical limitations and platform constraints
• Resource requirements and scalability needs
• External system dependencies

[Technical Architecture And Design]
• System architecture patterns and components
• Technology stack recommendations
• Design principles and patterns

[Security And Compliance Requirements]
• Security policies and access controls
• Compliance standards and regulations
• Data protection and privacy requirements

[Performance And Scalability Requirements]
• Performance benchmarks and SLAs
• Scalability requirements and load handling
• Resource optimization strategies

[User Interface And User Experience Requirements]
• UI/UX design guidelines and principles
• User interaction patterns and workflows
• Accessibility and usability requirements

[Data Flow And Integration Requirements]
• Data models and database requirements
• API specifications and integration patterns
• Third-party service integrations
```

### Additional Context
```
{docs_summary}
```
- Summary of additional uploaded documents
- Supporting documentation insights
- Technical specifications or reference materials

### Instructions
```
Instructions: The above content has been intelligently extracted and summarized using RAG (Retrieval Augmented Generation). 
It contains the most relevant requirements, user stories, and business objectives from the original documents.
Use this curated information to generate comprehensive epics and user stories.
```

## Token Management

- **Total Context**: Usually 5,000-8,000 tokens
- **Safe Limit**: 128,000 tokens (GPT-4o)
- **Optimization**: Smart truncation if approaching limits
- **Efficiency**: 50%+ faster than traditional two-agent approach

## Quality Assurance

- **Content Validation**: Minimum length, error detection
- **Semantic Relevance**: Cosine similarity scoring
- **Topic Coverage**: 10 different semantic query areas
- **Structure**: Clear section headers and logical flow
