import gradio as gr
import os
import PyPDF2
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import hashlib
from pathlib import Path
from openai import OpenAI
import json

def greet(name):
    """Simple greeting function"""
    if name:
        return f"Hello, {name}! Welcome to Gradio! 🎉"
    else:
        return "Hello! Please enter your name to get a personalized greeting. 👋"

def search_documents(query, num_results=5, anura_api_key=""):
    """Search the ChromaDB collection for relevant paragraphs and enhance with LLM analysis"""
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
        
        search_results_text = "\n".join(search_results)
        
        # Enhance with LLM analysis if API key is provided
        enhanced_results = enhance_search_with_llm(query, search_results_text, anura_api_key)
        return enhanced_results
    
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

def call_anura_api(prompt, anura_api_key):
    """Call Anura API to enhance search results with LLM analysis"""
    if not anura_api_key or not anura_api_key.strip():
        return None
    
    try:
        # Initialize OpenAI client with Anura base URL
        client = OpenAI(
            base_url="https://anura-testnet.lilypad.tech/api/v1",
            api_key=anura_api_key
        )

        # Call the LLM using OpenAI client
        completion = client.chat.completions.create(
            model="llama3.1:8b",
            messages=[
                {"role": "system", "content": "You are a helpful AI research assistant. Analyze the search results and provide insights, summaries, or answer questions based on the context provided."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Error calling Anura API: {str(e)}"

def enhance_search_with_llm(query, search_results_text, anura_api_key):
    """Enhance search results using Anura API LLM"""
    if not anura_api_key or not anura_api_key.strip():
        return search_results_text
    
    enhancement_prompt = f"""
Based on the user's query: "{query}"

And the following search results from documents:
{search_results_text}

Please provide:
1. A concise summary of the key findings
2. Direct answers to the user's question if possible
3. Any insights or connections between the different search results
4. Suggestions for further research if needed

Format your response clearly with headers and bullet points where appropriate.
"""
    
    llm_enhancement = call_anura_api(enhancement_prompt, anura_api_key)
    
    if llm_enhancement and not llm_enhancement.startswith("Error"):
        return f"{search_results_text}\n\n🤖 **AI Analysis & Summary:**\n{llm_enhancement}"
    else:
        return f"{search_results_text}\n\n🤖 **AI Analysis:** {llm_enhancement or 'Unable to generate analysis'}"

def advanced_rag_search_chat(message, history, num_results=5, anura_api_key=""):
    """Advanced RAG search with chat interface and history tracking"""
    if not message.strip():
        return history, history
    
    if not anura_api_key or not anura_api_key.strip():
        error_response = "Advanced RAG search requires an Anura API key. Please provide your API key."
        history.append([message, error_response])
        return history, history
    
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        
        try:
            collection = client.get_collection("pdf_documents")
        except:
            error_response = "No documents found. Please upload and process some PDF files first."
            history.append([message, error_response])
            return history, history
        
        # Query the collection for more results for better context
        results = collection.query(
            query_texts=[message],
            n_results=min(num_results * 2, 20)  # Get more results for analysis
        )
        
        if not results['documents'] or not results['documents'][0]:
            error_response = f"No relevant documents found for query: '{message}'"
            history.append([message, error_response])
            return history, history
        
        # Combine all relevant documents for comprehensive analysis
        documents = results['documents'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Create comprehensive context for LLM
        context_chunks = []
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            source_file = metadata.get('source_file', 'Unknown') if metadata else 'Unknown'
            para_index = metadata.get('paragraph_index', 'N/A') if metadata else 'N/A'
            similarity_score = max(0, (2 - distance) / 2 * 100)
            
            context_chunks.append(f"[Source: {source_file}, Paragraph {para_index}, Similarity: {similarity_score:.1f}%]\n{doc}")
        
        comprehensive_context = "\n\n---\n\n".join(context_chunks)
        
        # Build conversation context from history
        conversation_context = ""
        if history:
            conversation_context = "\n\nPrevious conversation:\n"
            for user_msg, assistant_msg in history[-3:]:  # Include last 3 exchanges for context
                conversation_context += f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
        
        # Enhanced RAG prompt for comprehensive analysis with conversation context
        rag_prompt = f"""
You are an expert research assistant analyzing documents to answer user queries in a conversational manner.
{conversation_context}
Current User Query: "{message}"

Document Context:
{comprehensive_context}

Please provide a comprehensive analysis that includes:

1. **Direct Answer**: If the documents contain a direct answer to the user's question, provide it clearly.

2. **Key Findings**: Summarize the most relevant information from the documents related to the query.

3. **Supporting Evidence**: Quote specific passages from the documents that support your findings (include source references).

4. **Analysis & Insights**: Provide deeper analysis, connections between different sources, and implications.

5. **Context Awareness**: If this relates to previous questions in our conversation, acknowledge the connection and build upon earlier discussion.

6. **Follow-up Questions**: Suggest relevant follow-up questions the user might want to explore.

Format your response conversationally but with clear structure. Always cite your sources when making claims.
"""
        
        llm_analysis = call_anura_api(rag_prompt, anura_api_key)
        
        if llm_analysis and not llm_analysis.startswith("Error"):
            response = f"{llm_analysis}\n\n---\n\n📊 **Search Statistics:**\n• Documents analyzed: {len(documents)}\n• Sources: {len(set(m.get('source_file', 'Unknown') for m in metadatas))}\n• Analysis powered by Anura API"
        else:
            response = f"Error in advanced analysis: {llm_analysis or 'Unable to generate comprehensive analysis'}"
        
        # Add to history
        history.append([message, response])
        return history, history
    
    except Exception as e:
        error_response = f"Error in advanced RAG search: {str(e)}"
        history.append([message, error_response])
        return history, history

def clear_chat_history():
    """Clear the chat history"""
    return [], []

# Create the Gradio interface
with gr.Blocks(title="AI-Enhanced Research Assistant") as demo:
    gr.Markdown("# 📚 AI-Enhanced Research Assistant")
    gr.Markdown("Upload PDF files and search through them with AI-powered analysis using Anura API integration")
    
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
                    anura_api_key = gr.Textbox(
                        label="Anura API Key (Optional)",
                        placeholder="Enter your Anura API key for AI-enhanced analysis...",
                        lines=1,
                        type="password"
                    )
                    search_btn = gr.Button("Search Documents", variant="primary")
                
                with gr.Column():
                    search_output = gr.Textbox(
                        label="Search Results",
                        lines=20,
                        interactive=False
                    )
            
            # Connect the search function
            search_btn.click(fn=search_documents, inputs=[search_input, num_results, anura_api_key], outputs=search_output)
            
            # Also trigger on Enter key press
            search_input.submit(fn=search_documents, inputs=[search_input, num_results, anura_api_key], outputs=search_output)
        
        with gr.TabItem("RAG Chat"):
            gr.Markdown("### 🤖 AI Research Chat with Document Analysis")
            gr.Markdown("Have a conversation with AI about your documents. Chat history is maintained for context.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    rag_num_results = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Documents per Query"
                    )
                    rag_anura_api_key = gr.Textbox(
                        label="Anura API Key",
                        placeholder="Enter your Anura API key for chat functionality...",
                        lines=1,
                        type="password"
                    )
                    clear_btn = gr.Button("Clear Chat History", variant="secondary")
                
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="AI Research Assistant Chat",
                        height=400,
                        show_label=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="",
                            placeholder="Ask questions about your documents...",
                            lines=1,
                            scale=4,
                            container=False
                        )
                        submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            # Chat state to maintain history
            chat_history = gr.State([])
            
            # Function to handle user input and update chat
            def user_input(message, history):
                if message.strip():
                    return "", history + [[message, ""]]
                return message, history
            
            def bot_response(history, num_results, api_key):
                if history and history[-1][1] == "":  # If there's a pending user message
                    user_message = history[-1][0]
                    # Remove the pending message and get response
                    history_without_pending = history[:-1]
                    updated_history, _ = advanced_rag_search_chat(user_message, history_without_pending, num_results, api_key)
                    return updated_history
                return history
            
            # Connect the chat functionality
            msg.submit(user_input, [msg, chatbot], [msg, chatbot]).then(
                bot_response, [chatbot, rag_num_results, rag_anura_api_key], [chatbot]
            )
            
            submit_btn.click(user_input, [msg, chatbot], [msg, chatbot]).then(
                bot_response, [chatbot, rag_num_results, rag_anura_api_key], [chatbot]
            )
            
            clear_btn.click(clear_chat_history, outputs=[chatbot, chat_history])
        
        with gr.TabItem("About & Usage"):
            gr.Markdown("""
            ## 📋 How to Use This Research Assistant
            
            ### 🔧 Features:
            1. **PDF Upload**: Upload multiple PDF documents for analysis
            2. **Document Search**: Basic search through your documents with optional AI enhancement
            3. **Advanced RAG Chat**: Interactive AI-powered research chat with conversation history
            
            ### 🤖 Anura API Integration:
            This application integrates with the **Anura API** to provide enhanced search results using Large Language Models (LLMs).
            
            **What you get with Anura API:**
            - AI-powered document analysis and summarization
            - Interactive chat interface with conversation memory
            - Intelligent answers to your research questions
            - Connections and insights across multiple documents
            - Comprehensive research recommendations
            - Context-aware responses that build on previous questions
            
            **To use AI features:**
            1. Get your Anura API key from: https://anura-testnet.lilypad.tech/
            2. Enter your API key in the search forms or chat interface
            3. Enjoy enhanced, intelligent search results and conversations!
            
            ### 🔍 Search Types:
            - **Document Search**: Shows relevant document excerpts with optional AI summary
            - **Advanced RAG Chat**: Interactive conversational AI that remembers your questions and builds context over time
            
            ### 💬 Chat Features:
            - **Conversation Memory**: The AI remembers previous questions and answers in your session
            - **Context Building**: Each question builds on previous ones for deeper analysis
            - **Follow-up Questions**: The AI suggests relevant follow-up questions
            - **Clear History**: Reset the conversation anytime with the "Clear Chat History" button
            
            ### 💡 Tips:
            - Upload multiple related PDFs for better cross-document analysis
            - Use specific, clear questions for better AI responses
            - Build on previous questions - the AI remembers your conversation
            - Ask follow-up questions to dive deeper into topics
            - The AI can identify gaps in information and suggest further research
            - All your documents are stored locally and securely
            """)
        
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