# utils/text_processing.py

import io
import PyPDF2
import docx

def validate_uploaded_file(uploaded_file) -> (bool, str):
    filename = uploaded_file.name.lower()
    if not (filename.endswith(".txt") or filename.endswith(".pdf") or filename.endswith(".docx")):
        return False, "Unsupported file type. Please upload a .txt, .pdf, or .docx file."
    return True, ""

def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    
    if filename.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    
    elif filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    
    else:
        raise ValueError("Unsupported file type during extraction")
