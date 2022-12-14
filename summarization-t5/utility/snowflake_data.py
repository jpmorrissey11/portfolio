import numpy as np
import sys
import snowflake.snowpark.functions as F
from snowflake.snowpark import Session

# Create a session

DEFAULT_GET_SCHEMA_QUERY = """
    SELECT 'CUPOLA_PROD."' || "schema" || '_unfiltered"' FROM CUPOLEA_PROD.PUBLIC.imports i ORDER BY i."schema" desc LIMIT 1
"""

user = "XXX"
account = "XXX"
warehouse = "XXX"
database = "XXX"
schema = "XXX"


connection_parameters = {
    "user": user,
    "account": account,
    "warehouse": warehouse,
    "database": database,
    "schema": schema,
    "authenticator": "externalbrowser",
}

session = Session.builder.configs(connection_parameters).create()

print(
    session.sql(
        "select current_warehouse(), current_database(), current_schema()"
    ).collect()
)

first_year_customer = """
select
    *
from table
"""


s = session.sql(first_year_customer)
df = s.select("*").to_pandas()

df.columns = [c.lower() for c in df.columns]


df.to_csv("../data/snowflake_data.csv", index=False)
