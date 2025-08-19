

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
        # Prefer customer-level identity over account/transaction IDs for flat target tables
        cand = [
            'email', 'customer_id', f'{target_table}_id', 'reference', 'external_id', 'uuid',
            'account_id', 'transaction_id', 'id'
        ]
        tl = [c.lower() for c in target_columns]
        for k in cand:
            if k.lower() in tl:
                return [k]
        # Composite: prefer name+date style if present
        composites = [
            ['first_name', 'last_name', 'date_of_birth'],
            ['name', 'created_at'],
            ['email'],
        ]
        for comp in composites:
            if all(c in tl for c in comp):
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
        id_val = first_present(['customer_id', 'cust_id', 'customerid', 'custid'])
        if id_val:
            return f"cust:{id_val}"
        # 2) Email like
        email_val = first_present(['email', 'cust_email_addr', 'email_address'])
        if email_val:
            return f"email:{email_val}"
        # 3) Composite name
        first = first_present(['first_name', 'cust_first_nm'])
        last = first_present(['last_name', 'cust_last_nm'])
        if first or last:
            return f"name:{(first or '').strip()}_{(last or '').strip()}"
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

    def _build_flat_entities(self, src: sqlite3.Connection, leg: sqlite3.Connection, target_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Build entity rows for 'banking_olap_flat_exact' by joining tables with proper keys.
        Produces at most one row per customer across both source and legacy.
        """
        def only_cols(row: Dict[str, Any]) -> Dict[str, Any]:
            return {k: row.get(k) for k in target_cols}

        entities: Dict[str, Dict[str, Any]] = {}

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
                    t_row['data_origin'] = 'source'
                ek = None
                if c.get('email'):
                    ek = f"email:{str(c.get('email')).strip().lower()}"
                elif c.get('customer_id') is not None:
                    ek = f"cust:{str(c.get('customer_id')).strip().lower()}"
                if ek:
                    entities[ek] = t_row
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
                    t_row['data_origin'] = 'legacy'
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
        except Exception:
            pass

        # Ensure only target_cols are present
        for k in list(entities.keys()):
            entities[k] = only_cols(entities[k])
        return entities

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
        try:
            tgt.execute('BEGIN')
            grouped = self._group_mapping_by_target_table(mapping)
            global_entities = set()
            for tgt_table, rows in grouped.items():
                # Determine involved columns
                target_cols = sorted({r['target'] for r in rows if r['target']})
                # Entity key selection
                key_cols = (key_config or {}).get(tgt_table) or self._choose_entity_keys(tgt_table, target_cols)
                # Special-case: flat OLAP table assembled via joins
                if tgt_table == 'banking_olap_flat_exact':
                    entity_map = self._build_flat_entities(src, leg, target_cols)
                    ins = 0
                    for ek, t_row in entity_map.items():
                        rowid = self._insert_or_update(tgt, tgt_table, t_row, key_cols)
                        if rowid == -1:
                            logs.append(f"[{tgt_table}] FAILED upsert for key={ek}")
                        else:
                            ins += 1
                        if ek:
                            global_entities.add(ek)
                    summary[tgt_table] = {'entities': len(entity_map), 'inserted_or_updated': ins, 'key_cols': len(key_cols)}
                    logs.append(f"[{tgt_table}] joined entities={len(entity_map)} using keys={key_cols}")
                    continue
                # Build origin-select columns mapping
                src_needed: Dict[str, List[str]] = {}
                leg_needed: Dict[str, List[str]] = {}
                for r in rows:
                    if r['origin'] == 'legacy':
                        if r['source_table'] and r['source']:
                            leg_needed.setdefault(r['source_table'], []).append(r['source'])
                    else:
                        if r['source_table'] and r['source']:
                            src_needed.setdefault(r['source_table'], []).append(r['source'])
                # Fetch rows per origin table
                origin_rows: List[Tuple[str, Dict[str, Any]]] = []  # (origin, row)
                for tbl, cols in src_needed.items():
                    for row in self._fetch_all_rows(src, tbl, cols):
                        origin_rows.append(('source', {**row}))
                for tbl, cols in leg_needed.items():
                    for row in self._fetch_all_rows(leg, tbl, cols):
                        origin_rows.append(('legacy', {**row}))
                # Merge rows into target shape keyed by entity key
                entity_map: Dict[str, Dict[str, Any]] = {}
                for origin, o_row in origin_rows:
                    # Build a target-shaped row from mapping for this origin row
                    t_row: Dict[str, Any] = {c: None for c in target_cols}
                    for r in rows:
                        if r['origin'] != origin:
                            continue
                        src_tbl = r['source_table']; src_col = r['source']; tgt_col = r['target']
                        if not src_tbl or not src_col or not tgt_col:
                            continue
                        # Copy value if present
                        val = o_row.get(src_col)
                        if val is not None:
                            t_row[tgt_col] = val
                    # Mark origin if requested
                    if 'data_origin' in target_cols:
                        t_row['data_origin'] = 'source' if origin == 'source' else 'legacy'
                    # Compute entity key
                    ek = self._entity_key_value(t_row, key_cols)
                    # If target key columns unavailable or empty, derive from origin business key
                    if not any((t_row.get(k) not in (None, '')) for k in key_cols):
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
                        if 'data_origin' in base and base.get('data_origin') != 'existing':
                            # Upgrade to source if current is legacy and this row is source
                            if origin == 'source' and base.get('data_origin') == 'legacy':
                                base['data_origin'] = 'source'
                    else:
                        entity_map[ek] = t_row
                # Upsert to target
                ins = upd = 0
                for ek, t_row in entity_map.items():
                    rowid = self._insert_or_update(tgt, tgt_table, t_row, key_cols)
                    if rowid == -1:
                        logs.append(f"[{tgt_table}] FAILED upsert for key={ek}")
                    else:
                        # Heuristic: treat as insert if key-only lookup returned none (we can't perfectly know without extra query)
                        # For simplicity, count as insert if any key part was newly seen in this run
                        ins += 1
                    if ek:
                        global_entities.add(ek)
                summary[tgt_table] = {'entities': len(entity_map), 'inserted_or_updated': ins, 'key_cols': len(key_cols)}
                logs.append(f"[{tgt_table}] merged entities={len(entity_map)} using keys={key_cols}")
            tgt.commit()
            return {"status": "Migration completed", "summary": summary, "logs": logs, "global_unique_entities": len(global_entities)}
        except Exception as e:
            try:
                tgt.rollback()
            except Exception:
                pass
            return {"status": "error", "error": str(e), "logs": logs}
        finally:
            try: src.close()
            except Exception: pass
            try: leg.close()
            except Exception: pass
            try: tgt.close()
            except Exception: pass

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
