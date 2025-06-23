import gradio as gr
import os

def greet(name):
    """Simple greeting function"""
    if name:
        return f"Hello, {name}! Welcome to the Research Assistant! 🎉"
    else:
        return "Hello! Please enter your name to get a personalized greeting. 👋"

def process_single_pdf(pdf_file):
    """Process a single uploaded PDF file"""
    if pdf_file is None:
        return "No file uploaded. Please select a PDF file."
    
    try:
        # Get file info
        file_name = os.path.basename(pdf_file.name)
        file_size = os.path.getsize(pdf_file.name)
        
        # Basic validation
        if not file_name.lower().endswith('.pdf'):
            return f"❌ Error: {file_name} is not a PDF file"
        
        # Process info
        result = f"✅ Successfully uploaded: {file_name}\n"
        result += f"📊 File size: {file_size:,} bytes\n"
        result += f"📂 Temporary path: {pdf_file.name}\n"
        result += f"🚀 Status: Ready for processing!\n\n"
        result += f"📝 Next steps:\n"
        result += f"• File validation: ✅ Complete\n"
        result += f"• Text extraction: 🔄 Ready\n"
        result += f"• Analysis: 🔄 Ready\n"
        
        return result
    
    except Exception as e:
        return f"Error processing file: {str(e)}"

# Create the Gradio interface using Blocks
with gr.Blocks(title="Research Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📚 Research Assistant")
    gr.Markdown("Upload PDF files for analysis or get a personalized greeting.")
    
    with gr.Tabs():
        with gr.TabItem("📄 PDF Upload"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("📤 Process PDF", variant="primary")
                
                with gr.Column():
                    pdf_output = gr.Textbox(
                        label="Upload Status",
                        lines=12,
                        interactive=False,
                        placeholder="Upload a PDF file to see processing status here..."
                    )
            
            # Connect the button to the PDF processing function
            upload_btn.click(fn=process_single_pdf, inputs=pdf_input, outputs=pdf_output)
            
            # Also trigger on file upload
            pdf_input.upload(fn=process_single_pdf, inputs=pdf_input, outputs=pdf_output)
        
        with gr.TabItem("👋 Greeting"):
            with gr.Row():
                with gr.Column():
                    name_input = gr.Textbox(
                        label="Your Name", 
                        placeholder="Enter your name here...",
                        lines=1
                    )
                    greet_btn = gr.Button("👋 Greet Me!", variant="primary")
                
                with gr.Column():
                    greet_output = gr.Textbox(
                        label="Greeting",
                        lines=3,
                        interactive=False,
                        placeholder="Your personalized greeting will appear here..."
                    )
            
            # Connect the button to the function
            greet_btn.click(fn=greet, inputs=name_input, outputs=greet_output)
            
            # Also trigger on Enter key press
            name_input.submit(fn=greet, inputs=name_input, outputs=greet_output)

if __name__ == "__main__":
    demo.launch(share=True)
