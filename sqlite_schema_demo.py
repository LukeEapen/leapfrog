import sqlite3
import json
from pathlib import Path

# Utility: Map JSON types to SQLite types
def map_type(t):
    t = t.lower()
    if t.startswith('int'): return 'INTEGER'
    if t.startswith('float') or t.startswith('decimal'): return 'REAL'
    if t.startswith('varchar') or t == 'string': return 'TEXT'
    if t == 'datetime' or t == 'timestamp': return 'TEXT'
    if t == 'uuid': return 'TEXT'
    return 'TEXT'

# Load schemas
def load_schema(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# Create tables from schema
def create_tables(conn, schema, prefix='src_'):
    for table in schema['tables']:
        fields = []
        pk = None
        seen = set()
        for field in table['fields']:
            fname = field['name']
            if fname in seen:
                continue  # skip duplicate
            seen.add(fname)
            col = f"{fname} {map_type(field['type'])}"
            if field.get('primary_key'): pk = fname
            fields.append(col)
        if pk:
            fields.append(f"PRIMARY KEY({pk})")
        sql = f"CREATE TABLE {prefix}{table['name']} ({', '.join(fields)})"
        conn.execute(sql)

# Insert sample data
def insert_sample(conn, schema, prefix='src_'):
    for table in schema['tables']:
        cols = [f['name'] for f in table['fields']]
        # Insert 2 sample rows
        for i in range(2):
            vals = []
            for f in table['fields']:
                t = map_type(f['type'])
                if t == 'INTEGER': vals.append(i+1)
                elif t == 'REAL': vals.append(100.0 + i)
                elif t == 'TEXT': vals.append(f"sample_{f['name']}_{i+1}")
                else: vals.append(None)
            conn.execute(f"INSERT INTO {prefix}{table['name']} ({', '.join(cols)}) VALUES ({', '.join(['?' for _ in cols])})", vals)

# Interactive query shell
def query_shell(conn):
    print("\nEnter SQL queries (type 'exit' to quit):")
    while True:
        q = input('sqlite> ')
        if q.strip().lower() == 'exit': break
        try:
            for row in conn.execute(q):
                print(row)
        except Exception as e:
            print('Error:', e)

if __name__ == '__main__':
    # Paths to schemas
    src_path = Path('poc4/frontend/static/schemas/source_schema.json')
    tgt_path = Path('poc4/frontend/static/schemas/target_schema.json')
    src_schema = load_schema(src_path)
    tgt_schema = load_schema(tgt_path)
    # Create in-memory DB
    conn = sqlite3.connect(':memory:')
    create_tables(conn, src_schema, prefix='src_')
    create_tables(conn, tgt_schema, prefix='tgt_')
    insert_sample(conn, src_schema, prefix='src_')
    insert_sample(conn, tgt_schema, prefix='tgt_')
    print('Source and target tables created with sample data.')
    print('Tables:', [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")])
    query_shell(conn)
