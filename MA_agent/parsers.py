

from langchain_core.output_parsers import BaseOutputParser

import re

def extract_sql_code(text: str) -> str | None:
    """
    Extracts the SQL query from a block of text. Handles:
      1) SQLQuery: ```sql ...``` fences
      2) ```sql ...``` fences
      3) ``` … ``` fences containing a SELECT
      4) SQLQuery: … (no fences)
      5) Bare SELECT …; up to semicolon
    Returns the SQL (trimmed), or None if no query found.
    """
    patterns = [
        # 1) SQLQuery: ```sql ...```
        r"SQLQuery:\s*```sql\s*(?P<sql>[\s\S]+?)```",
        # 2) ```sql ...```
        r"```sql\s*(?P<sql>[\s\S]+?)```",
        # 3) ``` … ``` containing SELECT
        r"```(?:[\s\S]*?)\s*(?P<sql>SELECT[\s\S]+?)```",
        # 4) SQLQuery: … but stop before SQLResult/Answer blocks
        r"SQLQuery:\s*(?P<sql>[\s\S]+?)(?=\n\s*(SQLResult:|Answer:)|\n\s*\n|$)",
        # 5) Bare SELECT …; up to semicolon
        r"(?P<sql>SELECT[\s\S]+?;)(?=\s|$)",
    ]

    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            sql = m.group("sql").strip()
            # Drop any trailing analysis blocks that sneaked into the capture.
            for stop_token in ("SQLResult:", "Answer:"):
                if stop_token in sql:
                    sql = sql.split(stop_token, 1)[0].strip()
            # strip any wrapping quotes
            if (sql.startswith(("'", '"')) and sql.endswith(("'", '"'))):
                sql = sql[1:-1].strip()
            return sql

    return None
        
def extract_python_code(text: str):
    """
    Extracts Python code from a block of text. Handles:
      1) ```python ... ``` fences
      2) ``` ... ``` fences containing Python constructs
      3) Bare 'def' or 'import' blocks up to a blank line
    Returns the code (trimmed), or None if no code found.
    """
    patterns = [
        # 1) ```python ... ```
        r"```python\s*(?P<code>[\s\S]+?)```",
        # 2) ``` ... ``` containing a Python keyword
        r"```(?:[\s\S]*?)\s*(?P<code>(?:def |import |class )[\s\S]+?)```",
        # 3) Bare def/import/class up to the next blank line or end
        r"(?P<code>(?:def |import |class )[\s\S]+?)(?=\n\s*\n|$)",
    ]
    
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            code = m.group("code").strip()
            # strip any wrapping quotes
            if (code.startswith(("'", '"')) and code.endswith(("'", '"'))):
                code = code[1:-1].strip()
            return code

    return None

def extract_html_code(text: str):
    """
    Extracts HTML code from a block of text. Handles:
      1) ```html ... ``` fences
      2) ``` ... ``` fences containing HTML constructs
      3) Bare HTML documents or fragments
    Returns the HTML (trimmed), or None if no HTML found.
    """
    html_start = (
        r"(?:<!DOCTYPE html>|<html\b|<head\b|<body\b|<main\b|"
        r"<section\b|<article\b|<div\b|<table\b|<svg\b|"
        r"<canvas\b|<style\b|<script\b)"
    )
    patterns = [
        # 1) ```html ... ```
        r"```html\s*(?P<code>[\s\S]+?)```",
        # 2) ``` ... ``` containing HTML constructs
        rf"```(?:[\w+-]*)?\s*(?P<code>{html_start}[\s\S]+?)```",
        # 3) Complete bare HTML document
        r"(?P<code><!DOCTYPE html>[\s\S]*?</html>)",
        r"(?P<code><html\b[\s\S]*?</html>)",
        # 4) Bare HTML fragment
        rf"(?P<code>{html_start}[\s\S]+)",
    ]
    
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            code = m.group("code").strip()
            # strip any wrapping quotes
            if (code.startswith(("'", '"')) and code.endswith(("'", '"'))):
                code = code[1:-1].strip()
            return code

    return None

class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):
        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        else:
            # Assume ```sql wasn't used
            return text

class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```python wasn't used
            return text

class HTMLOutputParser(BaseOutputParser):
    def parse(self, text: str):        
        html_code = extract_html_code(text)
        if html_code is not None:
            return html_code
        else:
            # Assume ```html wasn't used
            return text
