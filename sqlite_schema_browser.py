from flask import Flask, render_template_string, request
import sqlite3
import json
from pathlib import Path

app = Flask(__name__)

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

def get_conn():
    conn = sqlite3.connect(':memory:')
    src_path = Path('poc4/frontend/static/schemas/source_schema.json')
    tgt_path = Path('poc4/frontend/static/schemas/target_schema.json')
    src_schema = load_schema(src_path)
    tgt_schema = load_schema(tgt_path)
    create_tables(conn, src_schema, prefix='src_')
    create_tables(conn, tgt_schema, prefix='tgt_')
    insert_sample(conn, src_schema, prefix='src_')
    insert_sample(conn, tgt_schema, prefix='tgt_')
    return conn

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SQLite Schema Browser</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-4">
  <h2 class="mb-4">SQLite In-Memory Schema Browser</h2>
  <form method="post" class="mb-3">
    <div class="input-group">
      <input type="text" name="query" class="form-control" placeholder="Enter SQL query (e.g. SELECT * FROM src_customer)" value="{{ query }}">
      <button class="btn btn-danger" type="submit">Run</button>
    </div>
  </form>
  <div class="mb-3">
    <strong>Tables:</strong>
    {% for t in tables %}
      <span class="badge bg-danger">{{ t }}</span>
    {% endfor %}
  </div>
  {% if results %}
    <table class="table table-bordered table-sm">
      <thead>
        <tr>
        {% for col in columns %}<th>{{ col }}</th>{% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in results %}
        <tr>
          {% for col in columns %}<td>{{ row[col] }}</td>{% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% elif error %}
    <div class="alert alert-danger">{{ error }}</div>
  {% endif %}
</div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    conn = get_conn()
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    query = request.form.get('query', '')
    results = []
    columns = []
    error = None
    if query:
        try:
            cursor = conn.execute(query)
            columns = [d[0] for d in cursor.description] if cursor.description else []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
        except Exception as e:
            error = str(e)
    return render_template_string(TEMPLATE, tables=tables, query=query, results=results, columns=columns, error=error)

if __name__ == '__main__':
    app.run(debug=True, port=7000)
