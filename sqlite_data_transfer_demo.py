import argparse
import sqlite3
import json
from pathlib import Path

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
        sql = f"CREATE TABLE {prefix}{table['name']} ({', '.join(fields)})"
        conn.execute(sql)

def insert_sample(conn, schema, prefix='src_'):
    for table in schema['tables']:
        cols = [f['name'] for f in table['fields'] if f['name']]
        for i in range(2):
            vals = []
            for f in table['fields']:
                t = map_type(f['type'])
                if t == 'INTEGER': vals.append(i+1)
                elif t == 'REAL': vals.append(100.0 + i)
                elif t == 'TEXT': vals.append(f"sample_{f['name']}_{i+1}")
                else: vals.append(None)
            conn.execute(f"INSERT INTO {prefix}{table['name']} ({', '.join(cols)}) VALUES ({', '.join(['?' for _ in cols])})", vals)

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
        # Fetch from source
        rows = [row for row in conn.execute(f"SELECT {', '.join(src_fields)} FROM {source_table}")]
        # Insert into target
        for row in rows:
            conn.execute(f"INSERT INTO {target_table} ({', '.join(tgt_fields)}) VALUES ({', '.join(['?' for _ in tgt_fields])})", row)
        conn.commit()
        tgt_rows = [dict(zip(tgt_cols, row)) for row in conn.execute(f"SELECT * FROM {target_table}")]
        transfer_msg = f"Transferred {len(rows)} rows from {source_table} to {target_table}."
    return render_template_string(TEMPLATE, src_tables=src_tables, tgt_tables=tgt_tables, source_table=source_table, target_table=target_table, src_cols=src_cols, src_rows=src_rows, tgt_cols=tgt_cols, tgt_rows=tgt_rows, transfer_msg=transfer_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='poc4/frontend/static/schemas/source_schema.json')
    parser.add_argument('--target', type=str, default='poc4/frontend/static/schemas/target_schema.json')
    parser.add_argument('--mapfile', type=str, default=None)
    args = parser.parse_args()

    conn, src_schema, tgt_schema = get_conn(args.source, args.target)
    mapping, rules = load_mapping_rules(args.mapfile)

    # For demo: transfer data using mapping/rules
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
    # Transfer data
    rows = [row for row in conn.execute(f"SELECT {', '.join(src_fields)} FROM {source_table}")]
    for row in rows:
        conn.execute(f"INSERT INTO {target_table} ({', '.join(tgt_fields)}) VALUES ({', '.join(['?' for _ in tgt_fields])})", row)
    conn.commit()
    print(f"Transferred {len(rows)} rows from {source_table} to {target_table} using mapping and rules.")
