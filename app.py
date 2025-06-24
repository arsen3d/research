import gradio as gr
import os
import PyPDF2
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from pathlib import Path

def greet(name):
    """Simple greeting function"""
    if name:
        return f"Hello, {name}! Welcome to Gradio! 🎉"
    else:
        return "Hello! Please enter your name to get a personalized greeting. 👋"

def search_documents(query, num_results=5):
    """Search the ChromaDB collection for relevant paragraphs"""
    if not query.strip():
        return "Please enter a search query."
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            collection = client.get_collection("pdf_documents")
        except:
            return "No documents found. Please upload and process some PDF files first."
        
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=min(num_results, 10)  # Limit to max 10 results
        )
        
        if not results['documents'] or not results['documents'][0]:
            return f"No relevant documents found for query: '{query}'"
        
        # Format results
        search_results = []
        search_results.append(f"🔍 Search Results for: '{query}'\n")
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances), 1):
            source_file = metadata.get('source_file', 'Unknown') if metadata else 'Unknown'
            para_index = metadata.get('paragraph_index', 'N/A') if metadata else 'N/A'
            
            # Calculate similarity score (ChromaDB uses cosine distance, so smaller is better)
            # Convert distance to similarity percentage (distance 0 = 100%, distance 2 = 0%)
            similarity_score = max(0, (2 - distance) / 2 * 100)
            
            # Clean up the document text by removing excessive whitespace and newlines
            cleaned_doc = ' '.join(doc.split())
            
            search_results.append(f"📄 Result {i} (Similarity: {similarity_score:.1f}%):")
            search_results.append(f"   Source: {source_file} (Paragraph {para_index})")
            search_results.append(f"   Content: {cleaned_doc[:500]}{'...' if len(cleaned_doc) > 500 else ''}")
            search_results.append("")
        
        return "\n".join(search_results)
    
    except Exception as e:
        return f"Error searching documents: {str(e)}"

def process_pdf(pdf_files):
    """Process uploaded PDF files, extract text, split into paragraphs, and store in ChromaDB"""
    if pdf_files is None or len(pdf_files) == 0:
        return "No files uploaded. Please select PDF files."
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Get or create a collection for PDF documents
        try:
            collection = client.get_collection("pdf_documents")
        except:
            collection = client.create_collection("pdf_documents")
        
        # Initialize text splitter for paragraphs
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Adjust based on your needs
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        
        results = []
        total_size = 0
        valid_files = 0
        total_paragraphs = 0
        
        for i, pdf_file_path in enumerate(pdf_files, 1):
            # Get file info
            file_name = os.path.basename(pdf_file_path)
            file_size = os.path.getsize(pdf_file_path)
            
            # Basic validation
            if not file_name.lower().endswith('.pdf'):
                results.append(f"❌ File {i}: {file_name} - Error: Not a PDF file")
                continue
            
            try:
                # Extract text from PDF
                with open(pdf_file_path, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    text_content = ""
                    
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n"
                
                if not text_content.strip():
                    results.append(f"❌ File {i}: {file_name} - Error: No extractable text found")
                    continue
                
                # Split text into paragraphs/chunks
                paragraphs = text_splitter.split_text(text_content)
                
                if not paragraphs:
                    results.append(f"❌ File {i}: {file_name} - Error: No paragraphs extracted")
                    continue
                
                # Generate unique IDs for each paragraph
                file_hash = hashlib.md5(file_name.encode()).hexdigest()[:8]
                paragraph_ids = [f"{file_hash}_para_{j}" for j in range(len(paragraphs))]
                
                # Prepare metadata for each paragraph
                metadatas = [
                    {
                        "source_file": file_name,
                        "file_size": file_size,
                        "paragraph_index": j,
                        "total_paragraphs": len(paragraphs),
                        "file_path": pdf_file_path
                    }
                    for j in range(len(paragraphs))
                ]
                
                # Add paragraphs to ChromaDB collection
                collection.add(
                    documents=paragraphs,
                    ids=paragraph_ids,
                    metadatas=metadatas
                )
                
                # Add to results
                results.append(f"✅ File {i}: {file_name} ({file_size:,} bytes) - {len(paragraphs)} paragraphs stored in ChromaDB")
                total_size += file_size
                valid_files += 1
                total_paragraphs += len(paragraphs)
                
            except Exception as e:
                results.append(f"❌ File {i}: {file_name} - Error: {str(e)}")
                continue
        
        # Summary
        summary = f"\n📊 Summary:\n"
        summary += f"• Total files uploaded: {len(pdf_files)}\n"
        summary += f"• Valid PDF files processed: {valid_files}\n"
        summary += f"• Total size: {total_size:,} bytes\n"
        summary += f"• Total paragraphs stored in ChromaDB: {total_paragraphs}\n"
        summary += f"• ChromaDB collection: pdf_documents\n"
        summary += f"• Status: Ready for research!\n"
        
        return "\n".join(results) + summary
    
    except Exception as e:
        return f"Error processing files: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Researcher") as demo:
    gr.Markdown("# 📚 Research Assistant")
    gr.Markdown("Upload multiple PDF files for analysis and search through the stored content")
    
    with gr.Tabs():
        with gr.TabItem("PDF Upload"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF Files",
                        file_types=[".pdf"],
                        file_count="multiple",
                        type="filepath"
                    )
                    upload_btn = gr.Button("Process PDFs", variant="primary")
                
                with gr.Column():
                    pdf_output = gr.Textbox(
                        label="Upload Status",
                        lines=10,
                        interactive=False
                    )
            
            # Connect the button to the PDF processing function
            upload_btn.click(fn=process_pdf, inputs=pdf_input, outputs=pdf_output)
            
            # Also trigger on file upload
            pdf_input.upload(fn=process_pdf, inputs=pdf_input, outputs=pdf_output)
        
        with gr.TabItem("Document Search"):
            with gr.Row():
                with gr.Column():
                    search_input = gr.Textbox(
                        label="Search Query",
                        placeholder="Enter your search query here...",
                        lines=2
                    )
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Number of Results"
                    )
                    search_btn = gr.Button("Search Documents", variant="primary")
                
                with gr.Column():
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=15,
                        interactive=False
                    )
            
            # Connect the search function
            search_btn.click(fn=search_documents, inputs=[search_input, num_results], outputs=search_output)
            
            # Also trigger on Enter key press
            search_input.submit(fn=search_documents, inputs=[search_input, num_results], outputs=search_output)
        
        # with gr.TabItem("Greeting"):
        #     with gr.Row():
        #         with gr.Column():
        #             name_input = gr.Textbox(
        #                 label="Your Name", 
        #                 placeholder="Enter your name here...",
        #                 lines=1
        #             )
        #             greet_btn = gr.Button("Greet Me!", variant="primary")
                
        #         with gr.Column():
        #             output = gr.Textbox(
        #                 label="Greeting",
        #                 lines=2,
        #                 interactive=False
        #             )
            
        #     # Connect the button to the function
        #     greet_btn.click(fn=greet, inputs=name_input, outputs=output)
            
            # Also trigger on Enter key press
       #     name_input.submit(fn=greet, inputs=name_input, outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)