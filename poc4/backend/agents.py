

# POC4 Agents and System Instructions

class ChatbotAgent:
    """
    Provides conversational support, answers migration questions, explains errors, and suggests fixes.
    """
    SYSTEM_PROMPT = (
        "You are a helpful assistant for data migration. Answer user questions, explain errors, and suggest fixes."
    )
    def handle_message(self, message, session):
        # Implement LLM or rule-based response logic here
        return f"ChatbotAgent response to: {message}"

class SchemaMappingAgent:
    """
    Auto-maps fields, suggests mapping logic, and learns from corrections.
    """
    SYSTEM_PROMPT = (
        "You are an expert in schema mapping. Auto-map fields between source and target schemas, suggest mapping logic, and learn from user corrections."
    )
    def map_schema(self, source_schema, target_schema):
        # Build lists with table context
        import difflib
        source_fields = []
        for table in source_schema.get('tables', []):
            for field in table.get('fields', []):
                source_fields.append({
                    'name': field['name'],
                    'type': field['type'],
                    'table': table['name']
                })
        target_fields = []
        for table in target_schema.get('tables', []):
            for field in table.get('fields', []):
                target_fields.append({
                    'name': field['name'],
                    'type': field['type'],
                    'table': table['name']
                })
        # Static mapping dictionary for known pairs
        # No static mapping, use dynamic semantic matching only
        mapping = []
        matched_targets = set()
        def normalize(name):
            return name.replace('_', '').replace('-', '').replace(' ', '').lower()
        def describe_field(field):
            # Generate a verbose description based on name, type, and table
            name = field['name'].replace('_', ' ').replace('-', ' ')
            desc = f"Field '{field['name']}' in table '{field['table']}' of type '{field['type']}'. "
            # Add heuristics for banking terms and context
            if 'customer' in name.lower():
                desc += "This field relates to customer information. "
            if 'account' in name.lower():
                desc += "This field relates to account details. "
            if 'transaction' in name.lower():
                desc += "This field records transaction data. "
            if 'product' in name.lower():
                desc += "This field describes a product. "
            if 'email' in name.lower():
                desc += "Contains an email address. "
            if 'phone' in name.lower():
                desc += "Contains a phone number. "
            if 'address' in name.lower():
                desc += "Contains a postal address. "
            if 'date' in name.lower() or 'at' in name.lower():
                desc += "Contains a date or timestamp. "
            if 'balance' in name.lower():
                desc += "Contains a monetary balance. "
            if 'rate' in name.lower():
                desc += "Contains an interest rate. "
            if 'id' in name.lower() or 'reference' in name.lower():
                desc += "Acts as a unique identifier or reference. "
            if 'name' in name.lower():
                desc += "Represents a name or label. "
            return desc.strip()

        def semantic_similarity(desc1, desc2):
            # Use difflib for simple semantic similarity
            import difflib
            return difflib.SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()

        def fuzzy_match(src, tgt_list):
            src_desc = describe_field(src)
            src_norm = normalize(src['name'])
            def tokenize(name):
                import re
                # Split camelCase, snake_case, kebab-case, and spaces
                tokens = re.sub('([a-z])([A-Z])', r'\1 \2', name)
                tokens = re.sub(r'[_\-]', ' ', name)
                tokens = tokens.lower().split()
                return set(tokens)
            src_tokens = tokenize(src['name'])
            # Guarantee exact normalized name match (case-insensitive, ignore underscores, dashes, spaces)
            for tgt in tgt_list:
                tgt_norm = normalize(tgt['name'])
                if src_norm == tgt_norm:
                    print(f"[SchemaMappingAgent] Exact match: {src['name']} -> {tgt['name']}")
                    # Always map if names match, regardless of type
                    similarity = 1.0
                    return tgt, similarity, src_desc, describe_field(tgt)
            # Also check for alternate spellings (e.g., first_name vs firstname)
            for tgt in tgt_list:
                tgt_norm = normalize(tgt['name'])
                if src_norm.replace('name', '') == tgt_norm.replace('name', ''):
                    print(f"[SchemaMappingAgent] Alt spelling match: {src['name']} -> {tgt['name']}")
                    return tgt, 0.95, src_desc, describe_field(tgt)
            # Fallback to fuzzy logic
            best = None
            best_score = 0.0
            best_type_penalty = 0.0
            for tgt in tgt_list:
                tgt_desc = describe_field(tgt)
                tgt_norm = normalize(tgt['name'])
                tgt_tokens = tokenize(tgt['name'])
                desc_score = semantic_similarity(src_desc, tgt_desc)
                name_score = difflib.SequenceMatcher(None, src_norm, tgt_norm).ratio()
                token_score = len(src_tokens & tgt_tokens) / max(len(src_tokens | tgt_tokens), 1)
                # Boost for special token pairs
                boost = 0.0
                if ('first' in src_tokens and 'given' in tgt_tokens) or ('given' in src_tokens and 'first' in tgt_tokens):
                    boost += 0.3
                if ('last' in src_tokens and 'family' in tgt_tokens) or ('family' in src_tokens and 'last' in tgt_tokens):
                    boost += 0.3
                # Penalize for type/length differences
                src_type = src['type'].split('(')[0].lower()
                tgt_type = tgt['type'].split('(')[0].lower()
                type_penalty = 0.0
                if src_type != tgt_type:
                    type_penalty = 0.4  # strong penalty for type mismatch
                else:
                    # If types match but length/precision differs, mild penalty
                    import re
                    src_len = re.findall(r'\((.*?)\)', src['type'])
                    tgt_len = re.findall(r'\((.*?)\)', tgt['type'])
                    if src_len and tgt_len and src_len[0] != tgt_len[0]:
                        type_penalty = 0.2
                score = max(desc_score, name_score, token_score) + boost - type_penalty
                if score > best_score:
                    best_score = score
                    best = tgt
                    best_type_penalty = type_penalty
            if best_score >= 0.4 or best is not None:
                if best_type_penalty == 0.4 and best_score > 0.6:
                    best_score = 0.6
                return best, best_score, src_desc, describe_field(best) if best else ''
            return None, best_score, src_desc, describe_field(best) if best else ''
            # Token-based similarity (for BIAN, camelCase, snake_case, etc.)
            def tokenize(name):
                import re
                tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+', name)
                tokens += name.replace('_', ' ').replace('-', ' ').lower().split()
                return set([t.lower() for t in tokens if t])
            src_tokens = tokenize(src['name'])
            best = None
            best_score = 0.0
            for tgt in tgt_list:
                tgt_tokens = tokenize(tgt['name'])
                overlap = src_tokens & tgt_tokens
                score = len(overlap) / max(len(src_tokens), len(tgt_tokens), 1)
                banking_keywords = ['customer', 'account', 'transaction', 'product', 'balance', 'reference', 'name', 'type', 'date', 'email', 'phone', 'address', 'rate', 'interest', 'batch', 'id', 'code', 'number']
                src_banking = src_tokens & set(banking_keywords)
                tgt_banking = tgt_tokens & set(banking_keywords)
                banking_score = len(src_banking & tgt_banking) / max(len(src_banking | tgt_banking), 1)
                type_score = 1.0 if src['type'].split('(')[0].lower() in tgt['type'].lower() else 0.0
                total_score = score + banking_score * 0.5 + type_score * 0.2
                seq_score = difflib.SequenceMatcher(None, normalize(src['name']), normalize(tgt['name'])).ratio()
                total_score += seq_score * 0.2
                if total_score > best_score:
                    best_score = total_score
                    best = tgt
            if best_score > 0.55:
                return best
            return None
        # Map source fields to target fields using two-way fuzzy/semantic match
        mapping = []
        mapped_targets = set()
        for s_field in source_fields:
            match, confidence, src_desc, tgt_desc = fuzzy_match(s_field, target_fields)
            # Guarantee exact match mapping and update mapped_targets for all matches
            if match:
                mapped_targets.add(normalize(match['name']))
            mapping.append({
                'source': s_field['name'],
                'source_type': s_field['type'],
                'source_table': s_field['table'],
                'source_description': src_desc,
                'target': match['name'] if match else '',
                'target_type': match['type'] if match else '',
                'target_table': match['table'] if match else '',
                'target_description': tgt_desc,
                'auto_mapped': bool(match),
                'similarity': round(confidence, 2),
                'low_confidence': confidence < 0.6
            })
        # Also map target fields that were not matched to any source
        mapped_sources = [m['source'].lower() for m in mapping]
        for t_field in target_fields:
            if t_field['name'].lower() not in mapped_targets:
                mapping.append({
                    'source': '',
                    'source_type': '',
                    'source_table': '',
                    'source_description': '',
                    'target': t_field['name'],
                    'target_type': t_field['type'],
                    'target_table': t_field['table'],
                    'target_description': describe_field(t_field),
                    'auto_mapped': False,
                    'similarity': 0.0
                })
        return mapping

class TransformationRuleAgent:
    """
    Proposes and validates transformation rules for data cleansing and conversion.
    """
    SYSTEM_PROMPT = (
        "You are a transformation rule expert. Propose and validate rules for data type conversions, value mappings, and cleansing."
    )
    def suggest_rules(self, mapping):
        """
        Generate transformation rules with variations and concrete examples.
        Heuristics:
        - Multiple sources -> concatenate/compose with separators.
        - Type mismatch -> cast to target type.
        - Field-intent hints (email/phone/date/name/address/id/amount/rate) -> normalization rules.
        - Always include at least one example. Provide optional 'variants'.
        Note: Groups by target only; the caller may further split by origin.
        """
        def base_type(t):
            return (t or '').split('(')[0].strip().lower()
        def looks_like(name, token):
            return token in (name or '').replace('_', ' ').lower()
        rules = []
        by_target = {}
        for m in mapping:
            if m.get('target'):
                by_target.setdefault(m['target'], []).append(m)
        for tgt, entries in by_target.items():
            srcs = [e for e in entries if e.get('source')]
            src_names = [e.get('source') for e in srcs]
            primary = srcs[0] if srcs else entries[0]
            src_bt = base_type(primary.get('source_type'))
            tgt_bt = base_type(primary.get('target_type'))
            t_name = (tgt or '').lower()
            rule_txt = 'Direct mapping'
            example = f"Example: Map {src_names[0]} to {tgt}" if src_names else f"Example: Populate {tgt}"
            variants = []
            # Multiple sources -> compose
            if len(src_names) >= 2:
                sep = ' '
                rule_txt = f"Concatenate {', '.join(src_names[:-1])} + '{sep}' + {src_names[-1]}"
                # Example values
                if all(looks_like(s, 'name') for s in src_names) or looks_like(t_name, 'name'):
                    example = "Example: 'John' + ' ' + 'Doe' => 'John Doe'"
                elif any(looks_like(s, 'address') for s in src_names) or looks_like(t_name, 'address'):
                    example = "Example: '123 Main St' + ', ' + 'SF' + ' ' + '94107' => '123 Main St, SF 94107'"
                else:
                    example = "Example: 'A' + ' ' + 'B' => 'A B'"
                variants.append("Trim and collapse multiple spaces after concatenation")
            # Type conversion
            if src_bt and tgt_bt and src_bt != tgt_bt:
                cast_rule = f"CAST({src_names[0]} AS {tgt_bt.upper()})" if src_names else f"CAST(NULL AS {tgt_bt.upper()})"
                variants.append(f"Type conversion: {cast_rule}")
                if not src_names:
                    example = f"Example: Default to empty {tgt_bt} for {tgt}"
                else:
                    example = example or f"Example: Cast {src_names[0]} from {src_bt} to {tgt_bt}"
            # Intent-specific normalizations
            if looks_like(t_name, 'email') or any(looks_like(s, 'email') for s in src_names):
                variants.append("Normalize email: LOWER(TRIM(src))")
                example = "Example: ' Alice@Example.Com ' => 'alice@example.com'"
            if looks_like(t_name, 'phone') or any(looks_like(s, 'phone') for s in src_names):
                variants.append("Normalize phone: keep digits only; format E.164 if possible")
                example = "Example: '+1 (415) 555-1234' => '14155551234'"
            if looks_like(t_name, 'date') or looks_like(t_name, 'time'):
                variants.append("Parse and reformat date: e.g., YYYYMMDD -> YYYY-MM-DD")
                example = "Example: '20250107' => '2025-01-07'"
            if looks_like(t_name, 'name') and len(src_names) == 1:
                variants.append("Title-case the name; strip extra spaces")
                example = "Example: '  aLiCe  ' => 'Alice'"
            if looks_like(t_name, 'address'):
                variants.append("Normalize address abbreviations (St, Ave); remove double spaces")
            if looks_like(t_name, 'id') or looks_like(t_name, 'uuid'):
                variants.append("Ensure non-null ID; generate UUID if missing")
            if looks_like(t_name, 'amount') or looks_like(t_name, 'balance') or looks_like(t_name, 'rate'):
                variants.append("Round/format numeric precision appropriately (e.g., 2 decimals)")
            # Null/default handling
            if src_names:
                variants.append(f"Default when null: COALESCE({src_names[0]}, '')")
            else:
                rule_txt = "Default/static value"
                example = f"Example: Set {tgt} to '' (empty)"
            rules.append({
                'field': tgt,
                'sources': src_names,
                'rule': rule_txt,
                'example': example,
                'variants': variants
            })
        return rules

class ValidationAgent:
    """
    Checks for schema mismatches, missing data, and simulates migration outcomes.
    """
    SYSTEM_PROMPT = (
        "You are a validation expert. Check for schema mismatches, missing fields, and simulate migration outcomes."
    )
    def validate(self, mapping, rules):
        # Implement validation logic here
        return {"issues": ["Missing field: signup_date", "Type mismatch: total"]}

class MigrationExecutionAgent:
    """
    Handles actual data transfer, monitors progress, and manages rollbacks.
    """
    SYSTEM_PROMPT = (
        "You are a migration execution expert. Transfer data, monitor progress, and handle rollbacks if needed."
    )
    def run(self, mapping):
        # Implement migration logic here
        return {"status": "Migration completed", "logs": ["Transferred 1000 records"]}

class ReconciliationAgent:
    """
    Compares pre- and post-migration data, highlights discrepancies, and generates reports.
    """
    SYSTEM_PROMPT = (
        "You are a reconciliation expert. Compare source and target data post-migration, highlight discrepancies, and generate reconciliation reports."
    )
    def approve(self, migration_result):
        # Implement reconciliation logic here
        return {"matched": 950, "unmatched": 50, "report": "Sample reconciliation report"}

# Instantiate agents for import in routes.py
schema_mapping_agent = SchemaMappingAgent()
transformation_rule_agent = TransformationRuleAgent()
validation_agent = ValidationAgent()
migration_execution_agent = MigrationExecutionAgent()
reconciliation_agent = ReconciliationAgent()
chatbot_agent = ChatbotAgent()
