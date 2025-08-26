

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

import os, sqlite3, json, re, difflib
from typing import Dict, List, Tuple, Any, Optional

class MigrationExecutionAgent:
    """
    Handles actual data transfer, monitors progress, and manages rollbacks.
    Adds entity-based merging to keep data integrity across tables: the same logical
    record found in multiple origin tables (source/legacy) is merged and inserted once
    into the target, avoiding duplicates. Child tables reuse the same parent/entity key.
    """
    SYSTEM_PROMPT = (
        "You are a migration execution expert. Transfer data, monitor progress, enforce entity-level deduplication across source and legacy, and handle rollbacks if needed."
    )

    def _schemas_dir(self) -> str:
        return os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static', 'schemas')

    def _open_dbs(self) -> Tuple[sqlite3.Connection, sqlite3.Connection, sqlite3.Connection]:
        base = self._schemas_dir()
        src = sqlite3.connect(os.path.join(base, 'source.db'))
        leg = sqlite3.connect(os.path.join(base, 'legacy.db'))
        tgt = sqlite3.connect(os.path.join(base, 'target.db'))
        # Row factory for dict-like access
        src.row_factory = sqlite3.Row
        leg.row_factory = sqlite3.Row
        tgt.row_factory = sqlite3.Row
        return src, leg, tgt

    def _open_ds3_db(self) -> Optional[sqlite3.Connection]:
        """Open the DS3 (cards) DB if present."""
        try:
            base = self._schemas_dir()
            p = os.path.join(base, 'cards_ds3.db')
            if not os.path.exists(p):
                return None
            conn = sqlite3.connect(p)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:
            return None

    # --- Origin label helpers ---
    def _origin_label(self, origin_code: str) -> str:
        """Convert internal origin code to a friendly data_origin label.
        source -> Data Source 1, legacy -> Data Source 2, dsN -> Data Source N
        """
        oc = (origin_code or '').strip().lower()
        if oc == 'source':
            return 'Data Source 1'
        if oc == 'legacy':
            return 'Data Source 2'
        if oc.startswith('ds'):
            # extract number after 'ds'
            try:
                n = int(oc[2:])
                return f'Data Source {n}'
            except Exception:
                return 'Data Source'
        # passthrough for existing/unknown
        return origin_code

    def _origin_code_from_label(self, label: str) -> str:
        """Convert a friendly data_origin label back to an internal code.
        Data Source 1 -> source, Data Source 2 -> legacy, Data Source N -> dsN
        """
        if not label:
            return ''
        lab = label.strip()
        if lab.lower() == 'existing':
            return 'existing'
        # Normalize common forms
        if lab.lower() == 'data source 1':
            return 'source'
        if lab.lower() == 'data source 2':
            return 'legacy'
        if lab.lower().startswith('data source '):
            try:
                n = int(lab.split()[-1])
                return f'ds{n}'
            except Exception:
                return lab.lower()
        return lab.lower()

    def _group_mapping_by_target_table(self, mapping: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        by_tbl: Dict[str, List[Dict[str, Any]]] = {}
        for m in mapping or []:
            tgt_tbl = (m.get('target_table') or '').strip()
            tgt_col = (m.get('target') or '').strip()
            src_tbl = (m.get('source_table') or '').strip()
            src_col = (m.get('source') or '').strip()
            if not tgt_tbl or not tgt_col:
                # skip incomplete mapping rows
                continue
            by_tbl.setdefault(tgt_tbl, []).append({
                'origin': (m.get('origin') or 'source').lower(),
                'source_table': src_tbl,
                'source': src_col,
                'target_table': tgt_tbl,
                'target': tgt_col,
                'target_type': m.get('target_type')
            })
        return by_tbl

    def _choose_entity_keys(self, target_table: str, target_columns: List[str]) -> List[str]:
        """Pick a stable business/entity key for deduplication.
        Preference order: explicit id columns, common domain keys, else all mapped target columns.
        """
        # Explicit preference: for flat OLAP table, dedupe on customer_id when available
        tl = target_table.lower() if target_table else ''
        if 'customer_id' in [c.lower() for c in target_columns]:
            if tl == 'banking_olap_flat_exact':
                return ['customer_id']
        # Prefer customer-level identity over email (emails may be missing for DS3)
        cand = [
            'customer_id', 'email', f'{target_table}_id', 'reference', 'external_id', 'uuid',
            'account_id', 'transaction_id', 'id'
        ]
        tl_cols = [c.lower() for c in target_columns]
        for k in cand:
            if k.lower() in tl_cols:
                return [k]
        # Composite: prefer name+date style if present
        composites = [
            ['first_name', 'last_name', 'date_of_birth'],
            ['name', 'created_at'],
            ['email'],
        ]
        for comp in composites:
            if all(c in tl_cols for c in comp):
                return comp
        # Fallback: all columns (will hash to produce a key)
        return target_columns[:]

    def _entity_key_value(self, row: Dict[str, Any], keys: List[str]) -> str:
        vals = []
        for k in keys:
            v = row.get(k)
            if v is None:
                v = ''
            vals.append(str(v).strip().lower())
        return '|'.join(vals)

    def _origin_business_key(self, o_row: Dict[str, Any]) -> Optional[str]:
        """Derive a stable business key from an origin row when target key cols are missing.
        Priority order:
          1) customer_id-like (customer_id, cust_id, customerid, custid)
          2) email-like (email, cust_email_addr, email_address)
          3) composite name (first_name/cust_first_nm + last_name/cust_last_nm)
        Also supports DS3 hints (cardholder_id, email_addr).
        Returns a normalized string or None if no candidate.
        """
        # Normalize keys for case-insensitive lookup
        lower_map = {k.lower(): k for k in o_row.keys()}

        def first_present(names: List[str]) -> Optional[str]:
            for nm in names:
                k = lower_map.get(nm)
                if k and o_row.get(k) not in (None, ''):
                    return str(o_row.get(k)).strip().lower()
            return None

        # 1) Customer ID like (avoid generic 'id' which could be account/transaction id)
        id_val = first_present(['customer_id', 'cust_id', 'customerid', 'custid', 'cardholder_id'])
        if id_val:
            return f"cust:{id_val}"
        # 2) Email like
        email_val = first_present(['email', 'cust_email_addr', 'email_address', 'email_addr'])
        if email_val:
            return f"email:{email_val}"
        # 3) Composite name
        first = first_present(['first_name', 'cust_first_nm'])
        last = first_present(['last_name', 'cust_last_nm'])
        if first or last:
            return f"name:{(first or '').strip()}_{(last or '').strip()}"
        return None

    def _ds3_link_maps(self, ds3: Optional[sqlite3.Connection]) -> Dict[str, Any]:
        """Preload relationship maps for DS3 to compute a consistent entity anchor across tables."""
        maps = {
            'card_to_acct': {},
            'acct_to_holder': {},
            'txn_to_card': {},
            'auth_to_card': {},
            'case_to_txn': {},
        }
        if ds3 is None:
            return maps
        try:
            for r in ds3.execute('SELECT card_id, card_account_id FROM "card"'):
                maps['card_to_acct'][r[0]] = r[1]
        except Exception:
            pass
        try:
            for r in ds3.execute('SELECT card_account_id, cardholder_id FROM "card_account"'):
                maps['acct_to_holder'][r[0]] = r[1]
        except Exception:
            pass
        try:
            for r in ds3.execute('SELECT auth_id, card_id FROM "card_auth"'):
                maps['auth_to_card'][r[0]] = r[1]
        except Exception:
            pass
        try:
            for r in ds3.execute('SELECT txn_id, card_id FROM "card_txn"'):
                maps['txn_to_card'][r[0]] = r[1]
        except Exception:
            pass
        try:
            for r in ds3.execute('SELECT case_id, txn_id FROM "dispute_case"'):
                maps['case_to_txn'][r[0]] = r[1]
        except Exception:
            pass
        return maps

    def _ds3_anchor_for_row(self, table: str, row: Dict[str, Any], maps: Dict[str, Any]) -> Optional[str]:
        """Return a stable DS3 entity anchor string like 'ds3:holder:123' based on relationship chains."""
        tl = (table or '').lower()
        lk = {k.lower(): k for k in row.keys()}
        def _get(name: str):
            key = lk.get(name.lower())
            return row.get(key) if key else None
        # cardholder
        if tl == 'cardholder' or _get('cardholder_id') is not None:
            ch = _get('cardholder_id')
            if ch is not None:
                return f"ds3:holder:{ch}"
        # via card_account
        if tl == 'card_account' or _get('card_account_id') is not None:
            acct = _get('card_account_id')
            if acct is not None:
                holder = maps.get('acct_to_holder', {}).get(acct)
                return f"ds3:holder:{holder}" if holder is not None else f"ds3:acct:{acct}"
        # via card
        if tl == 'card' or _get('card_id') is not None:
            card = _get('card_id')
            if card is not None:
                acct = maps.get('card_to_acct', {}).get(card)
                holder = maps.get('acct_to_holder', {}).get(acct) if acct is not None else None
                return f"ds3:holder:{holder}" if holder is not None else f"ds3:card:{card}"
        # via card_txn
        if tl == 'card_txn' or _get('txn_id') is not None:
            txn = _get('txn_id')
            if txn is not None:
                card = maps.get('txn_to_card', {}).get(txn)
                if card is not None:
                    acct = maps.get('card_to_acct', {}).get(card)
                    holder = maps.get('acct_to_holder', {}).get(acct) if acct is not None else None
                    if holder is not None:
                        return f"ds3:holder:{holder}"
                return f"ds3:txn:{txn}"
        # via dispute_case -> card_txn chain
        if tl == 'dispute_case' or _get('case_id') is not None:
            case_id = _get('case_id')
            txn = _get('txn_id')
            if txn is None and case_id is not None:
                txn = maps.get('case_to_txn', {}).get(case_id)
            if txn is not None:
                card = maps.get('txn_to_card', {}).get(txn)
                if card is not None:
                    acct = maps.get('card_to_acct', {}).get(card)
                    holder = maps.get('acct_to_holder', {}).get(acct) if acct is not None else None
                    if holder is not None:
                        return f"ds3:holder:{holder}"
                return f"ds3:case:{case_id or ''}"
        # via card_auth
        if tl == 'card_auth' or _get('auth_id') is not None:
            auth = _get('auth_id')
            if auth is not None:
                card = maps.get('auth_to_card', {}).get(auth)
                if card is not None:
                    acct = maps.get('card_to_acct', {}).get(card)
                    holder = maps.get('acct_to_holder', {}).get(acct) if acct is not None else None
                    if holder is not None:
                        return f"ds3:holder:{holder}"
                return f"ds3:auth:{auth}"
        # merchant rows have no direct holder linkage in-row; skip to avoid creating separate entities
        if tl == 'merchant' or _get('merchant_id') is not None:
            return None
        return None

    def _fetch_all_rows(self, conn: sqlite3.Connection, table: str, columns: List[str]) -> List[Dict[str, Any]]:
        if not table:
            return []
        # Only select distinct needed columns
        cols = ', '.join([f'"{c}"' for c in set(columns) if c]) or '*'
        try:
            cur = conn.execute(f'SELECT {cols} FROM "{table}"')
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    def _ensure_table_columns(self, conn: sqlite3.Connection, table: str) -> List[str]:
        try:
            cur = conn.execute(f'PRAGMA table_info("{table}")')
            cols = [r[1] for r in cur.fetchall()]
            return cols
        except Exception:
            return []

    def _insert_or_update(self, tgt: sqlite3.Connection, table: str, row: Dict[str, Any], key_cols: List[str]) -> int:
        """Upsert-like behavior based on key cols. Returns the rowid/primary key if available."""
        # Try to find existing row by keys
        where = ' AND '.join([f'"{k}" = ?' for k in key_cols])
        params = [row.get(k) for k in key_cols]
        try:
            cur = tgt.execute(f'SELECT rowid FROM "{table}" WHERE {where} LIMIT 1', params)
            hit = cur.fetchone()
        except Exception:
            hit = None
        if hit:
            # Update changed columns (best-effort)
            set_cols = [k for k in row.keys() if k not in key_cols]
            if set_cols:
                set_sql = ', '.join([f'"{c}"=?' for c in set_cols])
                args = [row.get(c) for c in set_cols] + params
                try:
                    tgt.execute(f'UPDATE "{table}" SET {set_sql} WHERE {where}', args)
                except Exception:
                    pass
            return int(hit[0])
        # Insert
        cols = list(row.keys())
        placeholders = ', '.join(['?']*len(cols))
        col_sql = ', '.join([f'"{c}"' for c in cols])
        try:
            tgt.execute(f'INSERT INTO "{table}" ({col_sql}) VALUES ({placeholders})', [row.get(c) for c in cols])
            return int(tgt.execute('SELECT last_insert_rowid()').fetchone()[0])
        except Exception:
            return -1

    def _build_flat_entities(self, src: sqlite3.Connection, leg: sqlite3.Connection, target_cols: List[str]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
        """Build entity rows for 'banking_olap_flat_exact' by joining tables with proper keys.
        Produces at most one row per customer across both source and legacy.
        Returns a tuple (entities_map, origin_map) where origin_map[entity_key] is 'source' or 'legacy'.
        """
        def only_cols(row: Dict[str, Any]) -> Dict[str, Any]:
            return {k: row.get(k) for k in target_cols}

        entities: Dict[str, Dict[str, Any]] = {}
        ek_origin: Dict[str, str] = {}

        # Helper: fill product from account_type, using source product table if available
        def enrich_with_product(conn: sqlite3.Connection, t_row: Dict[str, Any]):
            acc_type = (t_row.get('account_type') or '').strip()
            if not acc_type:
                return
            try:
                cur = conn.execute('SELECT * FROM "product" WHERE LOWER(product_type)=LOWER(?) LIMIT 1', (acc_type,))
                pr = cur.fetchone()
                if pr:
                    prd = dict(pr)
                    for fld in ('product_id','product_name','product_type','interest_rate'):
                        if fld in target_cols and prd.get(fld) is not None and t_row.get(fld) in (None, ''):
                            t_row[fld] = prd.get(fld)
            except Exception:
                pass

        # 1) SOURCE customers
        try:
            for c in self._fetch_all_rows(src, 'customer', ['customer_id','first_name','last_name','email','phone','address','created_at']):
                t_row = {k: None for k in target_cols}
                # Customer fields
                for m_k, s_k in (
                    ('customer_id','customer_id'),('first_name','first_name'),('last_name','last_name'),
                    ('email','email'),('phone','phone'),('address','address'),('created_at','created_at')
                ):
                    if m_k in target_cols:
                        t_row[m_k] = c.get(s_k)
                # One account for this customer
                acc = None
                try:
                    cur = src.execute('SELECT * FROM "account" WHERE customer_id=? ORDER BY opened_at LIMIT 1', (c.get('customer_id'),))
                    acc = cur.fetchone()
                except Exception:
                    acc = None
                if acc:
                    accd = dict(acc)
                    for m_k, s_k in (
                        ('account_id','account_id'),('account_type','account_type'),('balance','balance'),('opened_at','opened_at')
                    ):
                        if m_k in target_cols and accd.get(s_k) is not None:
                            t_row[m_k] = accd.get(s_k)
                    # One transaction for this account
                    try:
                        cur = src.execute('SELECT * FROM "transaction" WHERE account_id=? ORDER BY transaction_date LIMIT 1', (accd.get('account_id'),))
                        txn = cur.fetchone()
                    except Exception:
                        txn = None
                    if txn:
                        txd = dict(txn)
                        for m_k, s_k in (
                            ('transaction_id','transaction_id'),('amount','amount'),('transaction_type','transaction_type'),('transaction_date','transaction_date')
                        ):
                            if m_k in target_cols and txd.get(s_k) is not None:
                                t_row[m_k] = txd.get(s_k)
                # Product enrichment from product_type
                enrich_with_product(src, t_row)
                # Mark origin
                if 'data_origin' in target_cols:
                    t_row['data_origin'] = self._origin_label('source')
                ek = None
                if c.get('email'):
                    ek = f"email:{str(c.get('email')).strip().lower()}"
                elif c.get('customer_id') is not None:
                    ek = f"cust:{str(c.get('customer_id')).strip().lower()}"
                if ek:
                    entities[ek] = t_row
                    ek_origin[ek] = 'source'
        except Exception:
            pass

        # 2) LEGACY customers
        try:
            for c in self._fetch_all_rows(leg, 'legacy_customer', ['cust_id','cust_first_nm','cust_last_nm','cust_email_addr','cust_phone_num','cust_postal_addr','cust_created_ts']):
                t_row = {k: None for k in target_cols}
                # Customer fields mapping
                mapping_pairs = (
                    ('customer_id','cust_id'),('first_name','cust_first_nm'),('last_name','cust_last_nm'),
                    ('email','cust_email_addr'),('phone','cust_phone_num'),('address','cust_postal_addr'),('created_at','cust_created_ts')
                )
                for m_k, s_k in mapping_pairs:
                    if m_k in target_cols and c.get(s_k) is not None:
                        t_row[m_k] = c.get(s_k)
                # One legacy account
                acc = None
                try:
                    cur = leg.execute('SELECT * FROM "legacy_account" WHERE cust_id=? ORDER BY acct_open_dt LIMIT 1', (c.get('cust_id'),))
                    acc = cur.fetchone()
                except Exception:
                    acc = None
                if acc:
                    accd = dict(acc)
                    # Map account_type codes
                    acct_type_cd = accd.get('acct_type_cd')
                    acct_type = None
                    if isinstance(acct_type_cd, str):
                        if acct_type_cd.upper() == 'CHK':
                            acct_type = 'Checking'
                        elif acct_type_cd.upper() == 'SAV':
                            acct_type = 'Savings'
                    for m_k, val in (
                        ('account_id', accd.get('acct_id')),
                        ('account_type', acct_type),
                        ('balance', accd.get('acct_curr_bal_amt')),
                        ('opened_at', accd.get('acct_open_dt')),
                    ):
                        if m_k in target_cols and val is not None:
                            t_row[m_k] = val
                    # One ledger entry (transaction) for this account
                    try:
                        cur = leg.execute('SELECT * FROM "legacy_ledger" WHERE acct_id=? ORDER BY entry_ts LIMIT 1', (accd.get('acct_id'),))
                        txn = cur.fetchone()
                    except Exception:
                        txn = None
                    if txn:
                        txd = dict(txn)
                        # Choose a human-readable transaction_type if present
                        txn_type = txd.get('txn_type_cd') or txd.get('dr_cr_ind')
                        for m_k, val in (
                            ('transaction_id', txd.get('ledger_entry_id')),
                            ('amount', txd.get('txn_amt')),
                            ('transaction_type', txn_type),
                            ('transaction_date', txd.get('entry_ts')),
                        ):
                            if m_k in target_cols and val is not None:
                                t_row[m_k] = val
                # Product enrichment via mapped account_type
                enrich_with_product(src, t_row)
                # Mark origin
                if 'data_origin' in target_cols:
                    t_row['data_origin'] = self._origin_label('legacy')
                ek = None
                if c.get('cust_email_addr'):
                    ek = f"email:{str(c.get('cust_email_addr')).strip().lower()}"
                elif c.get('cust_id') is not None:
                    ek = f"cust:{str(c.get('cust_id')).strip().lower()}"
                if ek:
                    # If same key exists from source, only fill missing fields
                    if ek in entities:
                        base = entities[ek]
                        for k, v in t_row.items():
                            if base.get(k) in (None, '') and v not in (None, ''):
                                base[k] = v
                    else:
                        entities[ek] = t_row
                        ek_origin[ek] = 'legacy'
        except Exception:
            pass

        # Ensure only target_cols are present
        for k in list(entities.keys()):
            entities[k] = only_cols(entities[k])
        return entities, ek_origin

    def run(self, mapping: List[Dict[str, Any]], key_config: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        Perform a best-effort migration using the provided mapping with entity-based merging.
        - Groups source/legacy rows by a stable business key per target table
        - Inserts/updates once per entity to avoid duplicates
        - Executes within a single transaction for integrity
        Returns a summary with per-table counters and basic logs.
        """
        logs: List[str] = []
        summary: Dict[str, Dict[str, int]] = {}
        src, leg, tgt = self._open_dbs()
        ds3 = self._open_ds3_db()
        try:
            tgt.execute('BEGIN')
            grouped = self._group_mapping_by_target_table(mapping)
            global_entities = set()
            for tgt_table, rows in grouped.items():
                # Determine involved columns
                target_cols = sorted({r['target'] for r in rows if r['target']})
                # For flat table, force-include customer_id so we can dedupe/populate it even if not mapped
                if tgt_table == 'banking_olap_flat_exact' and 'customer_id' not in [c.lower() for c in target_cols]:
                    target_cols = list(target_cols) + ['customer_id']
                # Entity key selection
                key_cols = (key_config or {}).get(tgt_table) or self._choose_entity_keys(tgt_table, target_cols)
                # Special-case: flat OLAP table assembled via joins
                if tgt_table == 'banking_olap_flat_exact':
                    # 1) Build source+legacy joined entities as before
                    entity_map, ek_origin_map = self._build_flat_entities(src, leg, target_cols)
                    ins = 0
                    # Track per-origin counts (approximate for src/legacy via data_origin)
                    origin_counts = {'source': 0, 'legacy': 0, 'ds3': 0}
                    processed_by_origin = {'source': 0, 'legacy': 0, 'ds3': 0}
                    # Compute processed counts directly from origins (customers and cardholders)
                    try:
                        processed_by_origin['source'] = int(src.execute('SELECT COUNT(1) FROM "customer"').fetchone()[0])
                    except Exception:
                        pass
                    try:
                        processed_by_origin['legacy'] = int(leg.execute('SELECT COUNT(1) FROM "legacy_customer"').fetchone()[0])
                    except Exception:
                        pass
                    for ek, t_row in entity_map.items():
                        rowid = self._insert_or_update(tgt, tgt_table, t_row, key_cols)
                        if rowid == -1:
                            logs.append(f"[{tgt_table}] FAILED upsert for key={ek}")
                        else:
                            ins += 1
                            # Use ek_origin_map for accurate attribution
                            src_origin = ek_origin_map.get(ek)
                            if src_origin in ('source','legacy'):
                                origin_counts[src_origin] += 1
                        if ek:
                            global_entities.add(ek)
                    # 2) Additionally, handle extra DS origins generically (e.g., ds3)
                    extra_rows = [r for r in rows if r.get('origin','').startswith('ds')]
                    if extra_rows and ds3 is not None:
                        # DS3 processed count = distinct holders
                        try:
                            processed_by_origin['ds3'] = int(ds3.execute('SELECT COUNT(DISTINCT cardholder_id) FROM "cardholder"').fetchone()[0])
                        except Exception:
                            pass
                        # Build table->columns map for ds3
                        ds3_needed: Dict[str, List[str]] = {}
                        mandatory = {
                            'cardholder': ['cardholder_id'],
                            'card_account': ['card_account_id','cardholder_id'],
                            'card': ['card_id','card_account_id'],
                            'card_txn': ['txn_id','card_id','merchant_id','auth_id'],
                            'card_auth': ['auth_id','card_id','merchant_id'],
                            'dispute_case': ['case_id','txn_id'],
                            'merchant': ['merchant_id']
                        }
                        for r in extra_rows:
                            if r['source_table'] and r['source']:
                                tbl = r['source_table']
                                ds3_needed.setdefault(tbl, []).append(r['source'])
                        # Ensure mandatory ID columns needed for anchor computation are present
                        for tbl, cols in list(ds3_needed.items()):
                            need = mandatory.get((tbl or '').lower(), [])
                            for c in need:
                                if c not in cols:
                                    cols.append(c)
                        ds3_maps = self._ds3_link_maps(ds3)
                        # Fetch and map
                        for tbl, cols in ds3_needed.items():
                            for o_row in self._fetch_all_rows(ds3, tbl, cols):
                                t_row: Dict[str, Any] = {c: None for c in target_cols}
                                for r in extra_rows:
                                    if r['source_table'] != tbl:
                                        continue
                                    src_col = r['source']; tgt_col = r['target']
                                    if not src_col or not tgt_col:
                                        continue
                                    val = o_row.get(src_col)
                                    if val is not None:
                                        t_row[tgt_col] = val
                                if 'data_origin' in target_cols:
                                    t_row['data_origin'] = self._origin_label('ds3')
                                # Prefer DS3 holder anchor for stable entity keys; also set customer_id when present
                                ds3_anchor = self._ds3_anchor_for_row(tbl, o_row, ds3_maps)
                                if ds3_anchor and 'customer_id' in target_cols and (t_row.get('customer_id') in (None, '')):
                                    try:
                                        parts = ds3_anchor.split(':')
                                        if len(parts) == 3 and parts[1] == 'holder':
                                            t_row['customer_id'] = parts[2]
                                    except Exception:
                                        pass
                                # Compute entity key after potential customer_id fill; if anchor exists, use it as ek
                                ek = self._entity_key_value(t_row, key_cols)
                                if ds3_anchor:
                                    ek = ds3_anchor
                                # Require holder-level anchor and a non-null customer_id for flat table to avoid NULL-key duplicates
                                parts = (ds3_anchor or '').split(':') if ds3_anchor else []
                                if (not ds3_anchor) or len(parts) != 3 or parts[1] != 'holder' or not t_row.get('customer_id'):
                                    continue
                                # Skip rows when no entity key can be derived (prevents NULL-key duplicates)
                                if not ek or not ek.strip('|'):
                                    continue
                                if not ek.strip('|'):
                                    # Derive from origin row when key empty (prefer DS3 relationship anchor)
                                    obk = ds3_anchor or self._origin_business_key(o_row)
                                    ek = obk or ek
                                rowid = self._insert_or_update(tgt, tgt_table, t_row, key_cols)
                                if rowid == -1:
                                    logs.append(f"[{tgt_table}] DS3 FAILED upsert for key={ek}")
                                else:
                                    ins += 1
                                    origin_counts['ds3'] += 1
                                if ek:
                                    global_entities.add(ek)
                    summary[tgt_table] = {'entities': len(entity_map), 'inserted_or_updated': ins, 'key_cols': len(key_cols), 'origin_counts': origin_counts, 'processed_by_origin': processed_by_origin}
                    logs.append(f"[{tgt_table}] joined entities={len(entity_map)} using keys={key_cols}; origin_counts={origin_counts}; processed={processed_by_origin}")
                    continue
                # Build origin-select columns mapping
                src_needed: Dict[str, List[str]] = {}
                leg_needed: Dict[str, List[str]] = {}
                ds3_needed: Dict[str, List[str]] = {}
                for r in rows:
                    if r['origin'] == 'legacy':
                        if r['source_table'] and r['source']:
                            leg_needed.setdefault(r['source_table'], []).append(r['source'])
                    elif r['origin'] == 'source':
                        if r['source_table'] and r['source']:
                            src_needed.setdefault(r['source_table'], []).append(r['source'])
                    elif r.get('origin','').startswith('ds'):
                        if r['source_table'] and r['source']:
                            ds3_needed.setdefault(r['source_table'], []).append(r['source'])
                # Fetch rows per origin table
                origin_rows: List[Tuple[str, str, Dict[str, Any]]] = []  # (origin, table, row)
                for tbl, cols in src_needed.items():
                    for row in self._fetch_all_rows(src, tbl, cols):
                        origin_rows.append(('source', tbl, {**row}))
                for tbl, cols in leg_needed.items():
                    for row in self._fetch_all_rows(leg, tbl, cols):
                        origin_rows.append(('legacy', tbl, {**row}))
                if ds3 is not None:
                    for tbl, cols in ds3_needed.items():
                        for row in self._fetch_all_rows(ds3, tbl, cols):
                            origin_rows.append(('ds3', tbl, {**row}))
                # Preload DS3 maps to compute anchors consistently
                ds3_maps = self._ds3_link_maps(ds3)
                # Merge rows into target shape keyed by entity key
                entity_map: Dict[str, Dict[str, Any]] = {}
                for origin, src_tbl_name, o_row in origin_rows:
                    # Build a target-shaped row from mapping for this origin row
                    t_row: Dict[str, Any] = {c: None for c in target_cols}
                    for r in rows:
                        if r['origin'] != origin:
                            continue
                        r_src_tbl = r['source_table']; src_col = r['source']; tgt_col = r['target']
                        if not r_src_tbl or not src_col or not tgt_col:
                            continue
                        # Copy value if present
                        val = o_row.get(src_col)
                        if val is not None:
                            t_row[tgt_col] = val
                    # Mark origin if requested
                    if 'data_origin' in target_cols:
                        t_row['data_origin'] = self._origin_label(origin)
                    # Compute entity key: favor DS3 relationship anchor to group across its tables
                    ek = self._entity_key_value(t_row, key_cols)
                    ds3_anchor: Optional[str] = None
                    if origin == 'ds3':
                        ds3_anchor = self._ds3_anchor_for_row(src_tbl_name, o_row, ds3_maps)
                        if ds3_anchor:
                            # Force grouping by holder/account anchor regardless of existing key cols
                            ek = ds3_anchor
                            # If anchor is holder:<id> and target has customer_id, populate it to stabilize upserts
                            if 'customer_id' in target_cols and (t_row.get('customer_id') in (None, '')):
                                try:
                                    # anchor format: 'ds3:holder:<id>'
                                    parts = ds3_anchor.split(':')
                                    if len(parts) == 3 and parts[1] == 'holder':
                                        t_row['customer_id'] = parts[2]
                                except Exception:
                                    pass
                    # If target key columns unavailable or empty, derive from origin business key
                    if not any((t_row.get(k) not in (None, '')) for k in key_cols):
                        obk: Optional[str] = ds3_anchor if ds3_anchor else None
                        if not obk:
                            obk = self._origin_business_key(o_row)
                        if obk:
                            ek = obk
                    # If still no entity key, skip this row (prevents product-only rows from creating entities)
                    if not ek.strip('|'):
                        continue
                    # Merge (source takes precedence over legacy by default)
                    if ek in entity_map:
                        base = entity_map[ek]
                        # Fill missing fields only
                        for c, v in t_row.items():
                            if (base.get(c) in (None, '')) and (v not in (None, '')):
                                base[c] = v
                        # Preserve data_origin preference: existing > source > legacy
                        if 'data_origin' in base and self._origin_code_from_label(base.get('data_origin')) != 'existing':
                            # Upgrade to source if current is legacy and this row is source
                            if origin == 'source' and self._origin_code_from_label(base.get('data_origin')) == 'legacy':
                                base['data_origin'] = self._origin_label('source')
                    else:
                        entity_map[ek] = t_row
                # Upsert to target
                ins = upd = 0
                origin_counts = {'source': 0, 'legacy': 0, 'ds3': 0}
                for ek, t_row in entity_map.items():
                    rowid = self._insert_or_update(tgt, tgt_table, t_row, key_cols)
                    if rowid == -1:
                        logs.append(f"[{tgt_table}] FAILED upsert for key={ek}")
                    else:
                        # Heuristic: treat as insert if key-only lookup returned none (we can't perfectly know without extra query)
                        # For simplicity, count as insert if any key part was newly seen in this run
                        ins += 1
                        # attribute counts by origin tag if present (map label -> code)
                        _lbl = t_row.get('data_origin')
                        _code = self._origin_code_from_label(_lbl)
                        if _code in origin_counts:
                            origin_counts[_code] += 1
                    if ek:
                        global_entities.add(ek)
                summary[tgt_table] = {'entities': len(entity_map), 'inserted_or_updated': ins, 'key_cols': len(key_cols), 'origin_counts': origin_counts}
                logs.append(f"[{tgt_table}] merged entities={len(entity_map)} using keys={key_cols}; origin_counts={origin_counts}")
            tgt.commit()
            return {"status": "Migration completed", "summary": summary, "logs": logs, "global_unique_entities": len(global_entities)}
        except Exception as e:
            try:
                tgt.rollback()
            except Exception:
                pass
            return {"status": "error", "error": str(e), "logs": logs}
        finally:
            try:
                src.close()
            except Exception:
                pass
            try:
                leg.close()
            except Exception:
                pass
            try:
                tgt.close()
            except Exception:
                pass
            try:
                if ds3 is not None:
                    ds3.close()
            except Exception:
                pass

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

# --- Target Model Design Agent ---
class TargetModelDesignAgent:
    """
    Proposes a target-state data model by unifying fields across the selected source schemas.
    Uses simple canonicalization inspired by BIAN/ISO domain naming (no external calls).
    """
    CANON_TABLES = {
        # banking core
        'customer': 'customer', 'legacy_customer': 'customer', 'cardholder': 'customer',
        'account': 'account', 'legacy_account': 'account', 'card_account': 'account',
        'transaction': 'transaction', 'legacy_ledger': 'transaction', 'card_txn': 'transaction',
        'product': 'product',
        # cards specifics kept as adjacent domain tables as needed
        'card': 'card', 'card_auth': 'card_auth', 'dispute_case': 'dispute_case', 'merchant': 'merchant'
    }

    FIELD_MAP = {
        # customer
        'cust_id': 'customer_id', 'customer_id': 'customer_id',
        'cardholder_id': 'customer_id',
        'first_name': 'first_name', 'cust_first_nm': 'first_name',
        'last_name': 'last_name', 'cust_last_nm': 'last_name',
        'email': 'email', 'cust_email_addr': 'email',
        'phone': 'phone', 'cust_phone_num': 'phone',
        'address': 'address', 'cust_postal_addr': 'address',
        'created_at': 'created_at', 'cust_created_ts': 'created_at',
        # account
        'account_id': 'account_id', 'acct_id': 'account_id', 'card_account_id': 'account_id',
        'customer_id_ref': 'customer_id',
        'customer_ref': 'customer_id',
        'customer_id': 'customer_id',
        'account_type': 'account_type', 'acct_type_cd': 'account_type',
        'balance': 'balance', 'acct_curr_bal_amt': 'balance',
        'opened_at': 'opened_at', 'acct_open_dt': 'opened_at',
        # transaction
        'transaction_id': 'transaction_id', 'ledger_entry_id': 'transaction_id', 'txn_id': 'transaction_id',
        'amount': 'amount', 'txn_amt': 'amount',
        'transaction_type': 'transaction_type', 'txn_type_cd': 'transaction_type', 'dr_cr_ind': 'transaction_type',
        'transaction_date': 'transaction_date', 'entry_ts': 'transaction_date',
        # product
        'product_id': 'product_id', 'product_name': 'product_name', 'product_type': 'product_type', 'interest_rate': 'interest_rate',
        # cards
        'card_id': 'card_id', 'card_status_cd': 'card_status',
        'auth_id': 'auth_id',
        'merchant_id': 'merchant_id', 'merchant_name': 'merchant_name',
        'case_id': 'case_id'
    }

    DEFAULT_TYPE = 'string'

    def _is_flat_table(self, name: str) -> bool:
        n = (name or '').strip().lower()
        return n in ('banking_olap_flat', 'banking_olap_flat_exact')

    def _classify_flat_field(self, field_name: str) -> str:
        """Classify a flat field into a canonical table using simple BIAN/ISO-aligned hints."""
        n = (field_name or '').strip().lower()
        # direct id hints first
        if n in ('customer_id','cust_id','cardholder_id','date_of_birth','dob','first_name','last_name','email','phone','address','created_at'):
            return 'customer'
        if n in ('account_id','acct_id','account_type','acct_type_cd','balance','acct_curr_bal_amt','opened_at','acct_open_dt','iban','sort_code'):
            return 'account'
        if n in ('transaction_id','txn_id','ledger_entry_id','amount','txn_amt','transaction_type','txn_type_cd','dr_cr_ind','transaction_date','entry_ts','currency'):
            return 'transaction'
        if n in ('product_id','product_name','product_type','interest_rate'):
            return 'product'
        if n in ('card_id','card_status','card_status_cd','pan','expiry','cvv'):
            return 'card'
        if n in ('auth_id','auth_ts','auth_amount'):
            return 'card_auth'
        if n in ('case_id','dispute_reason','case_status'):
            return 'dispute_case'
        if n in ('merchant_id','merchant_name','mcc','merchant_city','merchant_country'):
            return 'merchant'
        # token-based fallbacks
        tokens = [t for t in n.replace('-', '_').split('_') if t]
        ts = set(tokens)
        if {'customer','cust','cardholder','party'} & ts:
            return 'customer'
        if {'account','acct'} & ts:
            return 'account'
        if {'transaction','txn','ledger','entry'} & ts:
            return 'transaction'
        if {'product'} & ts:
            return 'product'
        if 'merchant' in ts:
            return 'merchant'
        if 'card' in ts and 'auth' not in ts:
            return 'card'
        if 'auth' in ts:
            return 'card_auth'
        if 'case' in ts or 'dispute' in ts:
            return 'dispute_case'
        # default unknowns to customer if identity-like, else transaction as generic catch-all
        if any(k in n for k in ['name','email','phone','address']):
            return 'customer'
        if any(k in n for k in ['amount','date','time']):
            return 'transaction'
        return 'customer'

    def _canon_table(self, name: str) -> str:
        key = (name or '').strip().lower()
        return self.CANON_TABLES.get(key, key)

    def _canon_field(self, table: str, field_name: str) -> str:
        key = (field_name or '').strip().lower()
        return self.FIELD_MAP.get(key, key)

    def design(self, source_schema_docs: list[dict]) -> dict:
        from collections import defaultdict
        # Collect fields per canonical table
        fields_by_table: dict[str, dict[str, str]] = defaultdict(dict)  # table -> field -> type
        for doc in source_schema_docs:
            try:
                for t in (doc or {}).get('tables', []):
                    tname = t.get('name')
                    # If a flat table is present, classify each field into canonical tables
                    if self._is_flat_table(tname):
                        for f in t.get('fields', []) or []:
                            raw_name = f.get('name')
                            if not raw_name:
                                continue
                            ctbl = self._classify_flat_field(raw_name)
                            cf = self._canon_field(ctbl, raw_name)
                            ftype = (f.get('type') or self.DEFAULT_TYPE)
                            prev = fields_by_table[ctbl].get(cf)
                            if not prev or (prev == self.DEFAULT_TYPE and ftype != self.DEFAULT_TYPE):
                                fields_by_table[ctbl][cf] = ftype
                    else:
                        ctbl = self._canon_table(tname)
                        for f in t.get('fields', []) or []:
                            raw_name = f.get('name')
                            if not raw_name:
                                continue
                            cf = self._canon_field(ctbl, raw_name)
                            ftype = (f.get('type') or self.DEFAULT_TYPE)
                            # Prefer a more specific type if encountered later
                            prev = fields_by_table[ctbl].get(cf)
                            if not prev or (prev == self.DEFAULT_TYPE and ftype != self.DEFAULT_TYPE):
                                fields_by_table[ctbl][cf] = ftype
            except Exception:
                continue

        # Ensure keys for core domains
        core_order = [
            'customer', 'account', 'transaction', 'product', 'card', 'card_auth', 'dispute_case', 'merchant'
        ]
        tables = []
        table_fields_cache: dict[str, list[dict]] = {}
        for tname in core_order + sorted([k for k in fields_by_table.keys() if k not in core_order]):
            if tname not in fields_by_table:
                continue
            flds = fields_by_table[tname]
            # Ensure primary keys for obvious tables
            pk_names = {
                'customer': 'customer_id', 'account': 'account_id', 'transaction': 'transaction_id', 'product': 'product_id',
                'card': 'card_id', 'card_auth': 'auth_id', 'dispute_case': 'case_id', 'merchant': 'merchant_id'
            }
            out_fields = []
            for fname, ftype in sorted(flds.items()):
                field_obj = {"name": fname, "type": ftype}
                if pk_names.get(tname) == fname:
                    field_obj["primary_key"] = True
                out_fields.append(field_obj)
            table_fields_cache[tname] = out_fields
            tables.append({"name": tname, "fields": out_fields})

        # Infer relationships (FKs) by matching *_id fields to primary keys of other tables
        pk_by_table = {}
        for t in tables:
            tname = t["name"]
            for f in t["fields"]:
                if f.get("primary_key"):
                    pk_by_table[tname] = f["name"]
                    break
        relationships = []
        # Helper: reverse map of pk name -> candidate tables
        pk_name_to_tables = {}
        for tb, pk in pk_by_table.items():
            pk_name_to_tables.setdefault(pk, []).append(tb)
        for t in tables:
            tname = t["name"]
            own_pk = pk_by_table.get(tname)
            for f in t["fields"]:
                fname = f.get("name")
                if not fname or fname == own_pk:
                    continue
                if fname.endswith("_id") or fname in pk_name_to_tables:
                    candidates = pk_name_to_tables.get(fname, [])
                    # Avoid self-reference unless intentional and primary key differs
                    candidates = [c for c in candidates if c != tname]
                    if candidates:
                        to_table = candidates[0]
                        relationships.append({
                            "from_table": tname,
                            "from_field": fname,
                            "to_table": to_table,
                            "to_field": pk_by_table.get(to_table),
                            "kind": "many_to_one"
                        })

        # Mermaid ER diagram (one-to-many where parent ||--o{ child)
        def _mm(name: str) -> str:
            return (name or '').upper().replace(' ', '_')
        mer_lines = ["erDiagram"]
        for rel in relationships:
            parent = _mm(rel["to_table"])  # one side
            child = _mm(rel["from_table"])  # many side
            label = rel.get("from_field") or "rel"
            mer_lines.append(f"    {parent} ||--o{{ {child} : {label}")
        er_mermaid = "\n".join(mer_lines)

        # Standards alignment hints
        alignment = {
            "customer": {"BIAN": "Party", "ISO20022": "Party"},
            "account": {"BIAN": "CurrentAccount/ProductArrangement", "ISO20022": "CashAccount"},
            "transaction": {"BIAN": "FinancialTransaction", "ISO20022": "Entry/Tx"},
            "product": {"BIAN": "Product", "ISO20022": "Product"},
            "card": {"BIAN": "Card", "ISO20022": "Card"},
            "card_auth": {"BIAN": "Authorization", "ISO20022": "Authorization"},
            "dispute_case": {"BIAN": "Case", "ISO20022": "InvestigationCase"},
            "merchant": {"BIAN": "Merchant", "ISO20022": "Merchant"}
        }

        return {
            "metadata": {
                "designed_by": "TargetModelDesignAgent",
                "standards_hint": ["BIAN", "ISO20022"],
                "version": 1
            },
            "tables": tables,
            "relationships": relationships,
            "er_diagram_mermaid": er_mermaid,
            "alignment": alignment
        }

# Instantiate agents for import in routes.py
schema_mapping_agent = SchemaMappingAgent()
transformation_rule_agent = TransformationRuleAgent()
validation_agent = ValidationAgent()
migration_execution_agent = MigrationExecutionAgent()
reconciliation_agent = ReconciliationAgent()
chatbot_agent = ChatbotAgent()
