#!/usr/bin/env python3
"""dictionary.md 원문을 dictionary.html 의 mdsrc 블록에 주입한다.

dictionary.md 가 바뀌면 이 스크립트를 다시 실행해 플래시카드 내용을 최신화한다.
멱등: 같은 입력으로 몇 번을 실행해도 결과가 같다.

    python3 build_dictionary_html.py
"""
import re
from pathlib import Path

base = Path(__file__).parent
md = (base / "dictionary.md").read_text(encoding="utf-8")
assert "</script" not in md.lower(), "md 에 </script 가 있으면 script 블록이 깨진다"

html_path = base / "dictionary.html"
html = html_path.read_text(encoding="utf-8")
pat = re.compile(r'(<script type="text/markdown" id="mdsrc">\n).*?(\n</script>)', re.S)
assert pat.search(html), "mdsrc 블록을 찾지 못했다"

html_path.write_text(pat.sub(lambda m: m.group(1) + md + m.group(2), html, count=1), encoding="utf-8")
entries = md.count("\n### ") + md.startswith("### ")
print(f"주입 완료: md {len(md.splitlines())}줄, ### 항목 {entries}개")
