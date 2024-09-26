import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('C:/Users/DELL/PycharmProjects/project3/instance/employees.db')
cursor = conn.cursor()

# Query existing columns
cursor.execute('PRAGMA table_info(attendance)')
existing_columns = [column[1] for column in cursor.fetchall()]

# Add missing columns
columns_to_add = {
    'year': 'INTEGER',
    'month': 'INTEGER',
    'day': 'INTEGER',
    'checkin_time': 'TEXT',
    'checkout_time': 'TEXT'
}

for column, column_type in columns_to_add.items():
    if column not in existing_columns:
        cursor.execute(f'ALTER TABLE attendance ADD COLUMN {column} {column_type}')
        print(f'Added column: {column}')

# Commit changes and close the connection
conn.commit()
conn.close()
