from typing import List
import re

def extract_json_only(text: str) -> List[str]:
    rs = r'(\[(\s*\{\s*"[^"]+"\s*\:\s*("[^"]*"|\d+|null)(\s*,\s*"[^"]+"\s*\:\s*("[^"]*"|\d+|null))*\s*\})(\s*,(\s*\{\s*"[^"]+"\s*\:\s*("[^"]*"|\d+|null)(\s*,\s*"[^"]*"\s*\:\s*("[^"]*"|\d+|null))*\s*\}))*\s*\])'

    matches = re.findall(rs, text)
    matches = [match[0] for match in matches]

    return matches