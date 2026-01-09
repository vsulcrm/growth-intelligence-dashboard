import streamlit as st

st.title("Debug App")
st.write("Streamlit is working!")

try:
    import pandas as pd
    st.write(f"Pandas version: {pd.__version__}")
except ImportError as e:
    st.error(f"Pandas Error: {e}")

try:
    import duckdb
    st.write(f"DuckDB version: {duckdb.__version__}")
    # Test simple query
    df = duckdb.query("SELECT 1 as a").df()
    st.write("DuckDB Query successful:", df)
except Exception as e:
    st.error(f"DuckDB Error: {e}")
