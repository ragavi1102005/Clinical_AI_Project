import os
import shutil
import pdfplumber
import re

def extract_all_pdfs(input_folder="medical_guidelines", output_folder="medical_guidelines_text"):
    # 1. Clean Slate Protocol
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"[*] Wiped old '{output_folder}' directory clean.")
    os.makedirs(output_folder)
    
    if not os.path.exists(input_folder):
        print(f"[!] The folder '{input_folder}' does not exist.")
        return

    pdf_files = [f for f in os.listdir(input_folder) if f.endswith('.pdf')]
    print(f"[*] Found {len(pdf_files)} PDFs. Starting dense text extraction...\n")

    for pdf_name in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_name)
        txt_name = os.path.splitext(pdf_name)[0] + ".txt"
        txt_path = os.path.join(output_folder, txt_name)

        try:
            extracted_text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract raw text WITHOUT the layout mapping
                    text = page.extract_text()
                    
                    if text:
                        # 2. Noise Reduction: Crush massive whitespaces and empty lines
                        text = re.sub(r' {2,}', ' ', text)  # Replace multiple spaces with a single space
                        text = re.sub(r'\n+', '\n', text)   # Replace multiple blank lines with a single line
                        
                        extracted_text += f"\n--- Page {page_num + 1} ---\n{text}\n"
            
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(extracted_text.strip())
                
            print(f"[+] Successfully converted: {pdf_name} -> {txt_name}")
            
        except Exception as e:
            print(f"[!] Error processing {pdf_name}: {e}")

    print(f"\n[*] Pipeline complete! Dense text files are ready in '{output_folder}'.")

if __name__ == "__main__":
    extract_all_pdfs()