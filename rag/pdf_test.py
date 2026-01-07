from pypdf import PdfReader

pdf_path = r"data\docs\Bharath Reddy - Data Engineer.pdf"

reader = PdfReader(pdf_path)
print("Pages:", len(reader.pages))

text = reader.pages[0].extract_text()
print("First page text length:", 0 if text is None else len(text))

print("\n--- First 500 chars ---\n")
print((text or "")[:500])
