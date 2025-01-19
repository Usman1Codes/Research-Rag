import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY in .env file")

# Define single optimized prompt
RESEARCH_PROMPT = """As an Expert Research Summarizer, analyze this content and create a clear, 
comprehensive summary that includes:

1. Research objectives and problem statement
2. Methodology and approach used
3. Key findings and results
4. Main conclusions and implications
5. Limitations or future work suggestions

Guidelines:
- Maintain academic rigor and objectivity
- Use precise, technical language appropriately
- Focus on significant insights and innovations
- Structure information logically
- Keep the summary clear and concise

Format the summary to be immediately valuable to technical professionals."""

def load_pdf(file_path: Path) -> Optional[str]:
    """Load PDF content."""
    try:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        return "\n\n".join(page.page_content for page in pages)
    except Exception as e:
        logger.error(f"Error loading PDF {file_path.name}: {str(e)}")
        return None

def summarize_text(text: str) -> Optional[str]:
    """Generate summary using LLM."""
    try:
        llm = ChatOpenAI(
            temperature=0.3,
            model_name="gpt-4",
            max_tokens=1500
        )
        messages = [
            {"role": "system", "content": RESEARCH_PROMPT},
            {"role": "user", "content": text}
        ]
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}")
        return None

def process_pdf(file_path: Path) -> Optional[str]:
    """Process a single PDF file and generate its summary."""
    try:
        # Load PDF content
        content = load_pdf(file_path)
        if not content:
            return None
        
        # Generate summary
        return summarize_text(content)
    except Exception as e:
        logger.error(f"Error processing {file_path.name}: {str(e)}")
        return None

def process_all_pdfs(input_dir: str = "docs") -> None:
    """Process all PDF files in the input directory."""
    # Setup directories
    input_path = Path(input_dir)
    output_path = input_path / "summaries"
    output_path.mkdir(exist_ok=True)

    # Get all PDF files
    pdf_files = list(input_path.glob('*.pdf'))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    successful = failed = 0

    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing: {pdf_file.name}")
            summary = process_pdf(pdf_file)
            
            if summary:
                output_file = output_path / f"{pdf_file.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                logger.info(f"Successfully processed: {pdf_file.name}")
                successful += 1
            else:
                logger.error(f"Failed to process: {pdf_file.name}")
                failed += 1
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")
            failed += 1

    logger.info("\nProcessing completed:")
    logger.info(f"Successfully processed: {successful} files")
    logger.info(f"Failed to process: {failed} files")

if __name__ == "__main__":
    try:
        process_all_pdfs()
    except Exception as e:
        logger.error(f"Application error: {e}")
