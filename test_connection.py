"""
Simple test script to check SQL Server connectivity
"""

import pyodbc
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get connection information
server = os.getenv('DB_SERVER')
database = os.getenv('DB_NAME')
username = os.getenv('DB_USER')
password = os.getenv('DB_PASSWORD')

print(f"Attempting to connect to {server} database {database} with user {username}")

# Try direct ODBC connection
try:
    # First with standard settings
    conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    print("Trying connection with:", conn_str)
    
    conn = pyodbc.connect(conn_str, timeout=10)
    print("Connection successful!")
    
    # Test query
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    row = cursor.fetchone()
    print("SQL Server version:", row[0])
    
    # Close connection
    conn.close()
    
except Exception as e:
    print("Connection failed:", str(e))
    
    # Try with TCP explicitly
    try:
        conn_str = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=tcp:{server},1433;DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes;Encrypt=no"
        print("\nTrying TCP connection with:", conn_str)
        
        conn = pyodbc.connect(conn_str, timeout=10)
        print("TCP Connection successful!")
        
        # Test query
        cursor = conn.cursor()
        cursor.execute("SELECT @@VERSION")
        row = cursor.fetchone()
        print("SQL Server version:", row[0])
        
        # Close connection
        conn.close()
        
    except Exception as e2:
        print("TCP Connection failed:", str(e2))
