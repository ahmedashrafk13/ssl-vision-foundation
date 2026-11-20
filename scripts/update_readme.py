"""Replace the README's '## Results' section with results/summary.md (image links re-pointed into results/)."""
import re
from pathlib import Path

readme, summary = Path("README.md"), Path("results/summary.md")
body = summary.read_text(encoding="utf-8").split("\n", 2)[2]  # drop the '# Results' title line
body = re.sub(r"^## ", "### ", body, flags=re.M).replace("](", "](results/")
text = readme.read_text(encoding="utf-8")
head = text.split("\n## Results", 1)[0].rstrip()
head = re.sub(r"> Status:.*?\n(?:>.*\n)*", "> Status: trained and evaluated; numbers below are from `results/*.jsonl`.\n", head)
readme.write_text(head + "\n\n## Results\n" + body, encoding="utf-8")
print("README updated")
