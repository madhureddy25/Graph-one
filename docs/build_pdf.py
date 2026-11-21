"""Render docs/architecture.md -> architecture.pdf (Playwright Chromium)."""
import asyncio
import pathlib
import sys

import markdown
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSS = """
@page { size: A4; margin: 13mm 13mm 12mm 13mm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 9.4pt; line-height: 1.42; color: #1d2433; }
h1 { font-size: 15pt; margin: 0 0 4px; color: #0f172a; }
h2 { font-size: 10.6pt; margin: 9px 0 3px; color: #0f3b78; border-bottom: 1px solid #d6dde8; padding-bottom: 2px; }
p { margin: 3px 0; } ul, ol { margin: 2px 0 2px 16px; padding: 0; } li { margin: 1px 0; }
pre { background: #f4f6fa; border: 1px solid #dde3ec; padding: 6px 8px; font-size: 6.9pt; line-height: 1.28; white-space: pre; overflow: hidden; }
code { font-family: Menlo, Consolas, monospace; font-size: 7.6pt; background: #f1f4f8; padding: 0 2px; border-radius: 2px; }
pre code { background: none; padding: 0; font-size: 6.6pt; }
table { border-collapse: collapse; width: 100%; margin: 4px 0; font-size: 8.6pt; }
th, td { border: 1px solid #d6dde8; padding: 3px 5px; vertical-align: top; text-align: left; } th { background: #eef2f8; }
"""


async def main(out: pathlib.Path) -> None:
    md = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    html = markdown.markdown(md, extensions=["tables", "fenced_code"])
    doc = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{html}</body></html>"
    async with async_playwright() as p:
        b = await p.chromium.launch()
        page = await b.new_page()
        await page.set_content(doc)
        await page.pdf(path=str(out), format="A4", print_background=True)
        await b.close()
    print("wrote", out)


if __name__ == "__main__":
    asyncio.run(main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "architecture.pdf"))
