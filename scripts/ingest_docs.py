#!/usr/bin/env python3
import os
import re
import json
import docx
from typing import List, Dict, Any, Optional

# File mapping configurations
DOCUMENTS_CONFIG = {
    "road_traffic_law.docx": {
        "document_id": "road_traffic_law",
        "document_title": "Закон Республики Казахстан \"О дорожном движении\"",
        "source_url": "https://adilet.zan.kz/rus/docs/Z1400000194",
        "type": "law"
    },
    "road_traffic_rules.docx": {
        "document_id": "road_traffic_rules",
        "document_title": "Правила дорожного движения Республики Казахстан",
        "source_url": "https://adilet.zan.kz/rus/docs/P2300000534",
        "type": "rules"
    },
    "code_of_administrative_offences.docx": {
        "document_id": "code_of_administrative_offences",
        "document_title": "Кодекс Республики Казахстан об административных правонарушениях",
        "source_url": "https://adilet.zan.kz/rus/docs/K1400000235",
        "type": "law"
    },
    "insurance_law.docx": {
        "document_id": "insurance_law",
        "document_title": "Закон Республики Казахстан \"Об обязательном страховании гражданско-правовой ответственности владельцев транспортных средств\"",
        "source_url": "https://adilet.zan.kz/rus/docs/Z030000446_",
        "type": "law"
    }
}

JURISDICTION = "Kazakhstan"
LANGUAGE = "ru"
DOCUMENT_VERSION = "2026-07"

# Directory paths
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def clean_line(text: str) -> str:
    """
    Cleans spacing, collapses tabs and non-breaking spaces, and removes repeated service headings.
    """
    # Replace non-breaking spaces, tab characters, and zero-width spaces with simple spaces
    text = text.replace('\xa0', ' ').replace('\t', ' ').replace('\u200b', ' ')
    
    # Strip editor service placeholders
    text = re.sub(r'Примечание ИЗПИ!', '', text)
    
    # Collapse double spaces
    text = re.sub(r' +', ' ', text)
    
    return text.strip()

def parse_docx_to_lines(path: str) -> List[str]:
    """
    Reads a docx file paragraph by paragraph, splitting internally by \n to preserve structure,
    and returns a cleaned list of non-empty text lines.
    """
    doc = docx.Document(path)
    lines = []
    for p in doc.paragraphs:
        # Split by manual newline to handle combined paragraph copy-pastes
        for line in p.text.split("\n"):
            cleaned = clean_line(line)
            if cleaned:
                lines.append(cleaned)
    return lines

def parse_law(lines: List[str], config: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
    """
    Parses standard laws containing Глава/Раздел and Статья.
    """
    records = []
    
    # Pattern to match: Глава 1. Общие положения или Раздел 1. Общие положения
    chapter_regex = re.compile(r'^\s*(Глава|Раздел)\s+(\d+(?:-\d+)*)\.?\s*(.*)', re.IGNORECASE)
    # Pattern to match: Статья 1. Основные понятия
    article_regex = re.compile(r'^\s*Статья\s+(\d+(?:-\d+)*)\.?\s*(.*)', re.IGNORECASE)
    
    current_chapter = "Преамбула"
    current_article_num = "Преамбула"
    current_article_title = ""
    current_text_blocks = []
    
    def save_current_record():
        if current_text_blocks:
            joined_text = "\n".join(current_text_blocks).strip()
            if joined_text:
                art_id = f"{config['document_id']}_{current_article_num.replace('Статья ', '').strip()}"
                records.append({
                    "document_id": config["document_id"],
                    "document_title": config["document_title"],
                    "source_file": filename,
                    "chapter": current_chapter,
                    "article": current_article_num,
                    "title": current_article_title,
                    "text": joined_text,
                    "language": LANGUAGE,
                    "jurisdiction": JURISDICTION,
                    "document_version": DOCUMENT_VERSION,
                    "source_url": config["source_url"],
                    "article_id": art_id
                })

    for line in lines:
        # Check for chapter
        chap_match = chapter_regex.match(line)
        if chap_match:
            # First, save any pending article text
            save_current_record()
            current_text_blocks = []
            
            chap_type = chap_match.group(1).strip().capitalize()
            chap_num = chap_match.group(2).strip()
            chap_title = chap_match.group(3).strip()
            current_chapter = f"{chap_type} {chap_num}"
            if chap_title:
                current_chapter += f". {chap_title}"
            continue
            
        # Check for article
        art_match = article_regex.match(line)
        if art_match:
            save_current_record()
            current_text_blocks = []
            
            art_num = art_match.group(1).strip()
            art_title = art_match.group(2).strip()
            current_article_num = f"Статья {art_num}"
            current_article_title = art_title
            
            # Start the text blocks with the line itself (the article header)
            current_text_blocks.append(line)
            continue
            
        # Otherwise, accumulate text block under active article
        current_text_blocks.append(line)
        
    # Save the final record
    save_current_record()
    return records

def parse_rules(lines: List[str], config: Dict[str, Any], filename: str) -> List[Dict[str, Any]]:
    """
    Parses road traffic rules containing Глава and Пункт.
    """
    records = []
    
    # Pattern to match: Глава 1. Общие положения
    chapter_regex = re.compile(r'^\s*Глава\s+(\d+(?:-\d+)*)\.?\s*(.*)', re.IGNORECASE)
    # Pattern to match: 1. Настоящие Правила...
    point_regex = re.compile(r'^\s*(\d+)\.\s+(.*)')
    
    current_chapter = "Преамбула"
    current_point_num = "Преамбула"
    current_text_blocks = []
    
    def save_current_record():
        if current_text_blocks:
            joined_text = "\n".join(current_text_blocks).strip()
            if joined_text:
                art_id = f"{config['document_id']}_{current_point_num.replace('Пункт ', '').strip()}"
                records.append({
                    "document_id": config["document_id"],
                    "document_title": config["document_title"],
                    "source_file": filename,
                    "chapter": current_chapter,
                    "article": current_point_num,
                    "title": current_chapter,  # Rules don't have separate article titles, use chapter context
                    "text": joined_text,
                    "language": LANGUAGE,
                    "jurisdiction": JURISDICTION,
                    "document_version": DOCUMENT_VERSION,
                    "source_url": config["source_url"],
                    "article_id": art_id
                })

    for line in lines:
        # Check for chapter
        chap_match = chapter_regex.match(line)
        if chap_match:
            save_current_record()
            current_text_blocks = []
            
            chap_num = chap_match.group(1).strip()
            chap_title = chap_match.group(2).strip()
            current_chapter = f"Глава {chap_num}"
            if chap_title:
                current_chapter += f". {chap_title}"
            continue
            
        # Check for point (e.g. "84. Остановка и стоянка...")
        point_match = point_regex.match(line)
        if point_match:
            save_current_record()
            current_text_blocks = []
            
            p_num = point_match.group(1).strip()
            current_point_num = f"Пункт {p_num}"
            
            # Start accumulating
            current_text_blocks.append(line)
            continue
            
        # Otherwise, accumulate
        current_text_blocks.append(line)
        
    save_current_record()
    return records

def process_file(filename: str, config: Dict[str, Any]):
    raw_path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(raw_path):
        print(f"Error: Raw file {raw_path} not found. Skipping.")
        return

    print(f"Parsing document: {filename}...")
    lines = parse_docx_to_lines(raw_path)
    print(f"Extracted {len(lines)} structured lines.")

    if config["type"] == "law":
        records = parse_law(lines, config, filename)
    else:
        records = parse_rules(lines, config, filename)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_filename = config["document_id"] + ".jsonl"
    out_path = os.path.join(PROCESSED_DIR, out_filename)
    
    print(f"Writing {len(records)} records to {out_path}...")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Finished processing {filename}.\n")

def main():
    print("Starting document ingestion pipeline...\n")
    for filename, config in DOCUMENTS_CONFIG.items():
        process_file(filename, config)
    print("All documents processed successfully!")

if __name__ == "__main__":
    main()
