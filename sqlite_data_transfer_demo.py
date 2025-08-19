import argparse
import sqlite3
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template_string, request

app = Flask(__name__)

# --- Load schemas ---
SRC_SCHEMA_PATH = 'poc4/frontend/static/schemas/source_schema.json'
LEG_SCHEMA_PATH = 'poc4/frontend/static/schemas/legacy_schema.json'
TGT_SCHEMA_PATH = 'poc4/frontend/static/schemas/target_schema.json'

# --- Sample data for each schema ---
SAMPLE_DATA = {
    'customer': [
        {'customer_id': 1, 'first_name': 'Alice', 'last_name': 'Smith', 'email': 'alice.smith@example.com', 'phone': '555-1234', 'address': '123 Main St', 'created_at': '2022-01-10 09:15:00'},
        {'customer_id': 2, 'first_name': 'Bob', 'last_name': 'Jones', 'email': 'bob.jones@example.com', 'phone': '555-5678', 'address': '456 Oak Ave', 'created_at': '2022-02-20 14:30:00'}
    ],
    'account': [
        {'account_id': 101, 'customer_id': 1, 'account_type': 'Checking', 'balance': 1500.75, 'opened_at': '2022-01-12 10:00:00'},
        {'account_id': 102, 'customer_id': 2, 'account_type': 'Savings', 'balance': 3200.00, 'opened_at': '2022-02-22 11:30:00'}
    ],
    'transaction': [
        {'transaction_id': 1001, 'account_id': 101, 'amount': 200.00, 'transaction_type': 'Deposit', 'transaction_date': '2022-03-01 08:00:00'},
        {'transaction_id': 1002, 'account_id': 102, 'amount': 50.00, 'transaction_type': 'Withdrawal', 'transaction_date': '2022-03-02 09:00:00'}
    ],
    'product': [
        {'product_id': 501, 'product_name': 'Premium Checking', 'product_type': 'Checking', 'interest_rate': 0.5},
        {'product_id': 502, 'product_name': 'High Yield Savings', 'product_type': 'Savings', 'interest_rate': 2.0}
    ],
    # Legacy DB uses distinct customers and values
    'legacy_customer': [
        {'cust_id': 11, 'cust_first_nm': 'Charlotte', 'cust_last_nm': 'Lee', 'cust_email_addr': 'charlotte.lee@legacy.com', 'cust_phone_num': '555-7777', 'cust_postal_addr': '789 Pine Rd', 'cust_created_ts': '2020-05-10'},
        {'cust_id': 12, 'cust_first_nm': 'Daniel', 'cust_last_nm': 'Ng', 'cust_email_addr': 'daniel.ng@legacy.com', 'cust_phone_num': '555-8888', 'cust_postal_addr': '101 Maple Ln', 'cust_created_ts': '2020-06-20'}
    ],
    'legacy_account': [
        {'acct_id': 401, 'cust_id': 11, 'acct_type_cd': 'CHK', 'acct_status_cd': 'A', 'acct_open_dt': '2020-05-12', 'acct_close_dt': None, 'acct_curr_bal_amt': 980.25, 'acct_avail_bal_amt': 975.00},
        {'acct_id': 402, 'cust_id': 12, 'acct_type_cd': 'SAV', 'acct_status_cd': 'A', 'acct_open_dt': '2020-06-22', 'acct_close_dt': None, 'acct_curr_bal_amt': 4321.55, 'acct_avail_bal_amt': 4321.55}
    ],
    'legacy_ledger': [
        {'ledger_entry_id': 701, 'acct_id': 401, 'entry_ts': '2020-07-01', 'dr_cr_ind': 'C', 'txn_amt': 125.00, 'currency_cd': 'USD', 'txn_type_cd': 'Deposit', 'ref_txn_id': None, 'batch_id': 7, 'created_by': 'legacy_user', 'created_ts': '2020-07-01'},
        {'ledger_entry_id': 702, 'acct_id': 402, 'entry_ts': '2020-07-03', 'dr_cr_ind': 'D', 'txn_amt': 75.50, 'currency_cd': 'USD', 'txn_type_cd': 'Withdrawal', 'ref_txn_id': None, 'batch_id': 7, 'created_by': 'legacy_user', 'created_ts': '2020-07-03'}
    ],
    # Target DB pre-seeded with a third, distinct dataset
    'banking_olap_flat_exact': [
        {
            'customer_id': 21, 'first_name': 'Eve', 'last_name': 'Adams', 'email': 'eve.adams@targetbank.com', 'phone': '555-2222', 'address': '222 Cedar St', 'created_at': '2023-04-15 10:05:00',
            'account_id': 901, 'account_type': 'Checking', 'balance': 2450.10, 'opened_at': '2023-04-16 09:00:00',
            'transaction_id': 9001, 'amount': 500.00, 'transaction_type': 'Deposit', 'transaction_date': '2023-04-20 12:00:00',
            'product_id': 801, 'product_name': 'Everyday Checking', 'product_type': 'Checking', 'interest_rate': 0.2,
            'data_origin': 'existing'
        },
        {
            'customer_id': 22, 'first_name': 'Frank', 'last_name': 'Brown', 'email': 'frank.brown@targetbank.com', 'phone': '555-3333', 'address': '333 Birch Blvd', 'created_at': '2023-05-10 15:45:00',
            'account_id': 902, 'account_type': 'Savings', 'balance': 7800.75, 'opened_at': '2023-05-11 10:30:00',
            'transaction_id': 9002, 'amount': 125.25, 'transaction_type': 'Withdrawal', 'transaction_date': '2023-05-18 08:20:00',
            'product_id': 802, 'product_name': 'Future Saver', 'product_type': 'Savings', 'interest_rate': 1.8,
            'data_origin': 'existing'
        }
    ]
}

def load_mapping_rules(mapfile):
    if not mapfile:
        return {}, []
    with open(mapfile, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('mapping', {}), data.get('rules', [])

# Utility: Map JSON types to SQLite types
def map_type(t):
    t = t.lower()
    if t.startswith('int'): return 'INTEGER'
    if t.startswith('float') or t.startswith('decimal'): return 'REAL'
    if t.startswith('varchar') or t == 'string': return 'TEXT'
    if t == 'datetime' or t == 'timestamp': return 'TEXT'
    if t == 'uuid': return 'TEXT'
    return 'TEXT'

def load_schema(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_tables(conn, schema, prefix='src_'):
    for table in schema['tables']:
        fields = []
        pk = None
        seen = set()
        for field in table['fields']:
            fname = field['name']
            if fname in seen:
                continue
            seen.add(fname)
            col = f"{fname} {map_type(field['type'])}"
            if field.get('primary_key'): pk = fname
            fields.append(col)
        if pk:
            fields.append(f"PRIMARY KEY({pk})")
        conn.execute(f'DROP TABLE IF EXISTS "{table["name"]}"')
        sql = f"CREATE TABLE \"{table['name']}\" ({', '.join(fields)})"
        conn.execute(sql)

def insert_sample(conn, schema, prefix='src_'):
    for table in schema['tables']:
        # Deduplicate column names to avoid duplicates in flat schemas
        seen_cols = set()
        cols_unique = []
        for f in table['fields']:
            name = f.get('name')
            if not name or name in seen_cols:
                continue
            seen_cols.add(name)
            cols_unique.append(name)
        if not cols_unique:
            continue
        # Build two synthetic sample rows
        for i in range(2):
            vals = []
            for f in table['fields']:
                name = f.get('name')
                if name not in seen_cols:
                    continue
                t = map_type(f['type'])
                if t == 'INTEGER':
                    vals.append(i + 1)
                elif t == 'REAL':
                    vals.append(100.0 + i)
                elif t == 'TEXT':
                    vals.append(f"sample_{name}_{i+1}")
                else:
                    vals.append(None)
            qcols = ', '.join([f'"{c}"' for c in cols_unique])
            placeholders = ', '.join(['?' for _ in cols_unique])
            conn.execute(f"INSERT INTO \"{table['name']}\" ({qcols}) VALUES ({placeholders})", vals)

def get_conn(src_path, tgt_path):
    conn = sqlite3.connect(':memory:')
    src_schema = load_schema(src_path)
    tgt_schema = load_schema(tgt_path)
    create_tables(conn, src_schema, prefix='src_')
    create_tables(conn, tgt_schema, prefix='tgt_')
    insert_sample(conn, src_schema, prefix='src_')
    insert_sample(conn, tgt_schema, prefix='tgt_')
    return conn, src_schema, tgt_schema

# Simple mapping: match fields by normalized name
from collections import defaultdict
def get_mapping(src_schema, tgt_schema):
    def norm(n): return n.replace('_','').replace('-','').replace(' ','').lower()
    src_fields = {}
    for table in src_schema['tables']:
        for f in table['fields']:
            src_fields[norm(f['name'])] = f['name']
    mapping = defaultdict(list)
    for table in tgt_schema['tables']:
        for f in table['fields']:
            nf = norm(f['name'])
            if nf in src_fields:
                mapping[f['name']].append(src_fields[nf])
    return mapping

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SQLite Data Transfer Demo</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
  <h2 class="mb-4">SQLite In-Memory Data Transfer Demo</h2>
  <form method="post" class="mb-3">
    <div class="row">
      <div class="col-md-6">
        <label class="form-label">Select Source Table</label>
        <select name="source_table" class="form-select mb-2">
          {% for t in src_tables %}
            <option value="{{ t }}" {% if t==source_table %}selected{% endif %}>{{ t }}</option>
          {% endfor %}
        </select>
      </div>
      <div class="col-md-6">
        <label class="form-label">Select Target Table</label>
        <select name="target_table" class="form-select mb-2">
          {% for t in tgt_tables %}
            <option value="{{ t }}" {% if t==target_table %}selected{% endif %}>{{ t }}</option>
          {% endfor %}
        </select>
      </div>
    </div>
    <button class="btn btn-danger mt-3" type="submit" name="action" value="transfer">Transfer Data</button>
  </form>
  <div class="row">
    <div class="col-md-6">
      <h5>Source Table Preview</h5>
      {% if src_rows %}
      <table class="table table-bordered table-sm">
        <thead><tr>{% for col in src_cols %}<th>{{ col }}</th>{% endfor %}</tr></thead>
        <tbody>
        {% for row in src_rows %}<tr>{% for col in src_cols %}<td>{{ row[col] }}</td>{% endfor %}</tr>{% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>
    <div class="col-md-6">
      <h5>Target Table Preview</h5>
      {% if tgt_rows %}
      <table class="table table-bordered table-sm">
        <thead><tr>{% for col in tgt_cols %}<th>{{ col }}</th>{% endfor %}</tr></thead>
        <tbody>
        {% for row in tgt_rows %}<tr>{% for col in tgt_cols %}<td>{{ row[col] }}</td>{% endfor %}</tr>{% endfor %}
        </tbody>
      </table>
      {% endif %}
    </div>
  </div>
  {% if transfer_msg %}
    <div class="alert alert-success mt-3">{{ transfer_msg }}</div>
  {% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    conn, src_schema, tgt_schema = get_conn()
    src_tables = [f"src_{t['name']}" for t in src_schema['tables']]
    tgt_tables = [f"tgt_{t['name']}" for t in tgt_schema['tables']]
    source_table = request.form.get('source_table', src_tables[0])
    target_table = request.form.get('target_table', tgt_tables[0])
    transfer_msg = None
    # Preview source
    src_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({source_table})")]
    src_rows = [dict(zip(src_cols, row)) for row in conn.execute(f"SELECT * FROM {source_table}")]
    # Preview target
    tgt_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({target_table})")]
    tgt_rows = [dict(zip(tgt_cols, row)) for row in conn.execute(f"SELECT * FROM {target_table}")]
    # Transfer data
    if request.method == 'POST' and request.form.get('action') == 'transfer':
        mapping = get_mapping(src_schema, tgt_schema)
        # Only transfer mapped columns
        tgt_fields = [c for c in tgt_cols if c in mapping]
        src_fields = [mapping[c][0] for c in tgt_fields]
        if not tgt_fields or not src_fields:
            transfer_msg = "No mapped fields between selected tables. Transfer skipped."
        else:
            quoted_src_fields = [f'"{f}"' for f in src_fields]
            quoted_tgt_fields = [f'"{f}"' for f in tgt_fields]
            try:
                rows = [row for row in conn.execute(f"SELECT {', '.join(quoted_src_fields)} FROM {source_table}")]
                for row in rows:
                    conn.execute(f"INSERT INTO {target_table} ({', '.join(quoted_tgt_fields)}) VALUES ({', '.join(['?' for _ in tgt_fields])})", row)
                conn.commit()
                tgt_rows = [dict(zip(tgt_cols, row)) for row in conn.execute(f"SELECT * FROM {target_table}")]
                transfer_msg = f"Transferred {len(rows)} rows from {source_table} to {target_table}."
            except sqlite3.OperationalError as e:
                transfer_msg = f"SQL error during transfer: {e}"
    return render_template_string(TEMPLATE, src_tables=src_tables, tgt_tables=tgt_tables, source_table=source_table, target_table=target_table, src_cols=src_cols, src_rows=src_rows, tgt_cols=tgt_cols, tgt_rows=tgt_rows, transfer_msg=transfer_msg)

# --- Create and populate DBs ---
def create_and_populate_db(schema_path, db_path, sample_data):
    # Ensure target directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    except Exception:
        pass
    schema = load_schema(schema_path)
    conn = sqlite3.connect(db_path)
    # Determine desired row multiplier based on schema type
    schema_lower = os.path.basename(schema_path).lower()
    if 'source_schema' in schema_lower:
        desired_count = 10
    elif 'legacy_schema' in schema_lower:
        desired_count = 15
    else:  # target
        desired_count = 2

    FIRST_NAMES = ["Alice","Bob","Carol","David","Eve","Frank","Grace","Heidi","Ivan","Judy","Mallory","Niaj","Olivia","Peggy","Rupert","Sybil","Trent","Victor","Wendy","Yvonne"]
    LAST_NAMES = ["Smith","Jones","Brown","Taylor","Wilson","Clark","Hall","Young","Allen","King","Scott","Green","Adams","Baker","Carter","Diaz","Evans","Foster","Gomez","Hayes"]
    STREETS = ["Main St","Oak Ave","Pine Rd","Maple Ln","Cedar St","Birch Blvd","Sunset Dr","Hillcrest Ave","Riverview Rd","Lakeview Dr"]
    PRODUCT_NAMES = ["Premium Checking","High Yield Savings","Everyday Checking","Future Saver","Platinum Money Market","Student Checking","Retirement Saver"]
    PRODUCT_TYPES = ["Checking","Savings","MoneyMarket"]
    TX_TYPES = ["Deposit","Withdrawal","Transfer","Fee","Interest"]

    def rand_phone():
        return f"555-{random.randint(2000,9999)}"

    def rand_timestamp(start_year=2021, end_year=2024):
        start = datetime(start_year,1,1)
        end = datetime(end_year,12,31,23,59,59)
        delta = end - start
        rand_sec = random.randint(0, int(delta.total_seconds()))
        return (start + timedelta(seconds=rand_sec)).strftime('%Y-%m-%d %H:%M:%S')

    def rand_date(start_year=2020, end_year=2024):
        start = datetime(start_year,1,1)
        end = datetime(end_year,12,31)
        delta = (end - start).days
        d = start + timedelta(days=random.randint(0,delta))
        return d.strftime('%Y-%m-%d')

    def expand_rows(base_rows, needed, table_def):
        if not base_rows or needed <= len(base_rows):
            return base_rows[:needed]
        # Identify PK
        pk_field = None
        for f in table_def.get('fields', []):
            if f.get('primary_key'):
                pk_field = f.get('name'); break
        if not pk_field:
            for f in table_def.get('fields', []):
                nm = f.get('name','')
                if nm.endswith('_id') or nm.endswith('id'):
                    pk_field = nm; break
        if not pk_field:
            pk_field = list(base_rows[0].keys())[0]
        # Determine current max
        max_existing = 0
        for r in base_rows:
            try:
                max_existing = max(max_existing, int(r.get(pk_field, 0)))
            except Exception:
                pass
        out = list(base_rows)
        next_id = max_existing + 1
        while len(out) < needed:
            # Choose a template row for structural reference only
            template = random.choice(base_rows)
            new_row = template.copy()
            new_row[pk_field] = next_id
            # Field-wise synthesis
            for k, v in list(new_row.items()):
                kl = k.lower()
                if k == pk_field:
                    continue
                if 'first_name' in kl or kl.endswith('first_nm'):
                    new_row[k] = random.choice(FIRST_NAMES)
                elif 'last_name' in kl or kl.endswith('last_nm'):
                    new_row[k] = random.choice(LAST_NAMES)
                elif 'product_name' in kl:
                    new_row[k] = random.choice(PRODUCT_NAMES)
                elif 'product_type' in kl:
                    new_row[k] = random.choice(PRODUCT_TYPES)
                elif 'account_type' in kl or kl.endswith('type_cd'):
                    new_row[k] = random.choice(["Checking","Savings","CHK","SAV"])
                elif 'email' in kl:
                    fn = new_row.get('first_name') or new_row.get('cust_first_nm') or random.choice(FIRST_NAMES)
                    ln = new_row.get('last_name') or new_row.get('cust_last_nm') or random.choice(LAST_NAMES)
                    domain = 'legacy.com' if 'legacy' in table_def.get('name','') or 'cust_' in kl else 'example.com'
                    new_row[k] = f"{fn.lower()}.{ln.lower()}{next_id}@{domain}"
                elif 'phone' in kl:
                    new_row[k] = rand_phone()
                elif 'address' in kl or 'postal_addr' in kl:
                    new_row[k] = f"{random.randint(100,999)} {random.choice(STREETS)}"
                elif kl.endswith('created_at') or kl.endswith('opened_at') or kl.endswith('transaction_date') or kl.endswith('created_ts'):
                    new_row[k] = rand_timestamp()
                elif 'date' in kl or kl.endswith('_dt'):
                    new_row[k] = rand_date()
                elif 'balance' in kl or 'amount' in kl or 'amt' in kl:
                    # Keep numeric style
                    try:
                        new_row[k] = round(random.uniform(50, 10000), 2)
                    except Exception:
                        pass
                elif 'interest_rate' in kl:
                    new_row[k] = round(random.uniform(0.1, 3.5), 2)
                elif isinstance(v, str) and (v.startswith('sample_') or v.endswith('_type')):
                    new_row[k] = v.replace('sample_', '')
            out.append(new_row)
            next_id += 1
        return out

    # Build relational sample sets when possible
    rows_by_table = {}
    if 'source_schema' in schema_lower:
        # Expand customers to desired_count
        cust_table = next((t for t in schema['tables'] if t['name'] == 'customer'), None)
        base_customers = sample_data.get('customer', [])
        customers = expand_rows(base_customers, desired_count, cust_table) if cust_table else []
        rows_by_table['customer'] = customers
        # Create 1 account per customer
        acct_table = next((t for t in schema['tables'] if t['name'] == 'account'), None)
        accounts = []
        if acct_table:
            next_acct_id = 1000
            for c in customers:
                accounts.append({
                    'account_id': next_acct_id,
                    'customer_id': c.get('customer_id'),
                    'account_type': random.choice(['Checking','Savings']),
                    'balance': round(random.uniform(100, 10000), 2),
                    'opened_at': rand_timestamp()
                })
                next_acct_id += 1
        rows_by_table['account'] = accounts
        # Create 1 transaction per account
        txn_table = next((t for t in schema['tables'] if t['name'] == 'transaction'), None)
        txns = []
        if txn_table:
            next_txn_id = 5000
            for a in accounts:
                txns.append({
                    'transaction_id': next_txn_id,
                    'account_id': a.get('account_id'),
                    'amount': round(random.uniform(10, 1000), 2),
                    'transaction_type': random.choice(TX_TYPES),
                    'transaction_date': rand_timestamp()
                })
                next_txn_id += 1
        rows_by_table['transaction'] = txns
        # Products from base (unchanged)
        rows_by_table['product'] = sample_data.get('product', [])
    elif 'legacy_schema' in schema_lower:
        # Expand legacy customers to desired_count
        lc_table = next((t for t in schema['tables'] if t['name'] == 'legacy_customer'), None)
        base_legacy_customers = sample_data.get('legacy_customer', [])
        legacy_customers = expand_rows(base_legacy_customers, desired_count, lc_table) if lc_table else []
        rows_by_table['legacy_customer'] = legacy_customers
        # Create 1 legacy account per legacy customer
        la_table = next((t for t in schema['tables'] if t['name'] == 'legacy_account'), None)
        legacy_accounts = []
        if la_table:
            next_lacct_id = 4000
            for c in legacy_customers:
                legacy_accounts.append({
                    'acct_id': next_lacct_id,
                    'cust_id': c.get('cust_id'),
                    'acct_type_cd': random.choice(['CHK','SAV']),
                    'acct_status_cd': 'A',
                    'acct_open_dt': rand_date(),
                    'acct_close_dt': None,
                    'acct_curr_bal_amt': round(random.uniform(100, 10000), 2),
                    'acct_avail_bal_amt': round(random.uniform(100, 10000), 2)
                })
                next_lacct_id += 1
        rows_by_table['legacy_account'] = legacy_accounts
        # Create 1 legacy ledger entry per legacy account
        ll_table = next((t for t in schema['tables'] if t['name'] == 'legacy_ledger'), None)
        legacy_ledgers = []
        if ll_table:
            next_led_id = 7000
            for a in legacy_accounts:
                legacy_ledgers.append({
                    'ledger_entry_id': next_led_id,
                    'acct_id': a.get('acct_id'),
                    'entry_ts': rand_date(),
                    'dr_cr_ind': random.choice(['D','C']),
                    'txn_amt': round(random.uniform(10, 1000), 2),
                    'currency_cd': 'USD',
                    'txn_type_cd': random.choice(['Deposit','Withdrawal','Transfer']),
                    'ref_txn_id': None,
                    'batch_id': random.randint(1, 10),
                    'created_by': 'legacy_user',
                    'created_ts': rand_date()
                })
                next_led_id += 1
        rows_by_table['legacy_ledger'] = legacy_ledgers

    for table in schema['tables']:
        fields = []
        pk = None
        seen = set()
        for field in table['fields']:
            fname = field['name']
            if fname in seen:
                continue
            seen.add(fname)
            col = f"{fname} {map_type(field['type'])}"
            if field.get('primary_key'): pk = fname
            fields.append(col)
        if pk:
            fields.append(f"PRIMARY KEY({pk})")
        conn.execute(f'DROP TABLE IF EXISTS "{table["name"]}"')
        sql = f"CREATE TABLE \"{table['name']}\" ({', '.join(fields)})"
        conn.execute(sql)
        # Insert sample data (use relational sets when present)
        rows = rows_by_table.get(table['name'])
        if rows is None:
            rows = sample_data.get(table['name'], [])
            # Expand or trim rows to desired_count
            if desired_count and rows:
                rows = expand_rows(rows, desired_count, table)
        if rows:
            # Deduplicate column list to match CREATE TABLE
            seen_cols = set()
            cols_unique = []
            for f in table['fields']:
                name = f.get('name')
                if not name or name in seen_cols:
                    continue
                seen_cols.add(name)
                cols_unique.append(name)
            qcols = ', '.join([f'"{c}"' for c in cols_unique])
            placeholders = ', '.join(['?' for _ in cols_unique])
            for row in rows:
                vals = [row.get(c) for c in cols_unique]
                conn.execute(f"INSERT INTO \"{table['name']}\" ({qcols}) VALUES ({placeholders})", vals)
    conn.commit()
    return conn

# --- Reset utility: delete DB files and recreate from SAMPLE_DATA ---
def reset_all_dbs(src_schema_path: str = SRC_SCHEMA_PATH, leg_schema_path: str = LEG_SCHEMA_PATH, tgt_schema_path: str = TGT_SCHEMA_PATH):
    # Use the directory of the provided schema paths to determine the DB base dir
    # This avoids issues when the working directory is different.
    base_dir = os.path.dirname(os.path.abspath(src_schema_path))
    try:
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass
    src_db = os.path.join(base_dir, 'source.db')
    leg_db = os.path.join(base_dir, 'legacy.db')
    tgt_db = os.path.join(base_dir, 'target.db')
    for p in (src_db, leg_db, tgt_db):
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    # Create and explicitly close connections to avoid file locks on Windows
    try:
        conn1 = create_and_populate_db(src_schema_path, src_db, SAMPLE_DATA)
        conn2 = create_and_populate_db(leg_schema_path, leg_db, SAMPLE_DATA)
        conn3 = create_and_populate_db(tgt_schema_path, tgt_db, SAMPLE_DATA)
    finally:
        try:
            conn1.close()
        except Exception:
            pass
        try:
            conn2.close()
        except Exception:
            pass
        try:
            conn3.close()
        except Exception:
            pass
    return {'source': src_db, 'legacy': leg_db, 'target': tgt_db}

@app.route('/reset', methods=['POST'])
def http_reset():
    try:
        reset_all_dbs()
        return 'Databases reset and repopulated with sample data.', 200
    except Exception as e:
        return f'Failed to reset: {e}', 500

# --- Preview route ---
@app.route('/preview/<schema_type>')
def preview(schema_type):
    db_map = {
        'source': ('source_schema.json', 'source.db'),
        'legacy': ('legacy_schema.json', 'legacy.db'),
        'target': ('target_schema.json', 'target.db')
    }
    if schema_type not in db_map:
        return 'Invalid schema type', 400
    schema_file, db_file = db_map[schema_type]
    db_path = f"poc4/frontend/static/schemas/{db_file}"
    # Create/populate if not exists
    import os
    if not os.path.exists(db_path):
        create_and_populate_db(f"poc4/frontend/static/schemas/{schema_file}", db_path, SAMPLE_DATA)
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    table_data = {}
    for t in tables:
        quoted_t = f'"{t}"'
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({quoted_t})")]
        rows = [dict(zip(cols, row)) for row in conn.execute(f"SELECT * FROM {quoted_t}")]
        table_data[t] = {'columns': cols, 'rows': rows}
    html = '<h3>Preview: {} schema</h3>'.format(schema_type.capitalize())
    for t, data in table_data.items():
        html += f'<h5>Table: {t}</h5>'
        html += '<table border=1 cellpadding=4><tr>' + ''.join(f'<th>{c}</th>' for c in data['columns']) + '</tr>'
        for row in data['rows']:
            html += '<tr>' + ''.join(f'<td>{row[c]}</td>' for c in data['columns']) + '</tr>'
        html += '</table>'
    return html

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=SRC_SCHEMA_PATH)
    parser.add_argument('--legacy', type=str, default=LEG_SCHEMA_PATH)
    parser.add_argument('--target', type=str, default=TGT_SCHEMA_PATH)
    parser.add_argument('--mapfile', type=str, default=None)
    parser.add_argument('--reset', action='store_true', help='Delete existing DB files and recreate with sample data')
    args = parser.parse_args()

    if args.reset:
        reset_all_dbs(args.source, args.legacy, args.target)
        print('Databases reset and repopulated with sample data.')
    else:
        # Create/populate DBs (idempotent table recreation)
        create_and_populate_db(args.source, 'poc4/frontend/static/schemas/source.db', SAMPLE_DATA)
        create_and_populate_db(args.legacy, 'poc4/frontend/static/schemas/legacy.db', SAMPLE_DATA)
        create_and_populate_db(args.target, 'poc4/frontend/static/schemas/target.db', SAMPLE_DATA)
        print('Databases created and populated with sample data.')

        # For demo: transfer data using mapping/rules
        conn, src_schema, tgt_schema = get_conn(args.source, args.target)
        mapping, rules = load_mapping_rules(args.mapfile)

        src_tables = [f"src_{t['name']}" for t in src_schema['tables']]
        tgt_tables = [f"tgt_{t['name']}" for t in tgt_schema['tables']]
        source_table = src_tables[0]
        target_table = tgt_tables[0]
        src_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({source_table})")]
        tgt_cols = [r[1] for r in conn.execute(f"PRAGMA table_info({target_table})")]
        # Use mapping from file if present, else fallback to default
        if mapping and isinstance(mapping, list):
            # mapping is a list of dicts from session
            tgt_fields = [m['target'] for m in mapping if m.get('source') and m.get('target')]
            src_fields = [m['source'] for m in mapping if m.get('source') and m.get('target')]
        else:
            # fallback to default mapping
            tgt_fields = [c for c in tgt_cols if c in mapping]
            src_fields = [mapping[c][0] for c in tgt_fields]
        if not tgt_fields or not src_fields:
            print("No mapped fields between selected tables. Transfer skipped.")
        else:
            quoted_src_fields = [f'"{f}"' for f in src_fields]
            quoted_tgt_fields = [f'"{f}"' for f in tgt_fields]
            try:
                rows = [row for row in conn.execute(f"SELECT {', '.join(quoted_src_fields)} FROM {source_table}")]
                for row in rows:
                    conn.execute(f"INSERT INTO {target_table} ({', '.join(quoted_tgt_fields)}) VALUES ({', '.join(['?' for _ in tgt_fields])})", row)
                conn.commit()
                print(f"Transferred {len(rows)} rows from {source_table} to {target_table} using mapping and rules.")
            except sqlite3.OperationalError as e:
                print(f"SQL error during transfer: {e}")
