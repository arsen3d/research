import gradio as gr
import os

def greet(name):
    """Simple greeting function"""
    if name:
        return f"Hello, {name}! Welcome to Gradio! 🎉"
    else:
        return "Hello! Please enter your name to get a personalized greeting. 👋"

def process_pdf(pdf_files):
    """Process uploaded PDF files"""
    if pdf_files is None or len(pdf_files) == 0:
        return "No files uploaded. Please select PDF files."
    
    try:
        results = []
        total_size = 0
        valid_files = 0
        
        for i, pdf_file_path in enumerate(pdf_files, 1):
            # Get file info
            file_name = os.path.basename(pdf_file_path)
            file_size = os.path.getsize(pdf_file_path)
            
            # Basic validation
            if not file_name.lower().endswith('.pdf'):
                results.append(f"❌ File {i}: {file_name} - Error: Not a PDF file")
                continue
            
            # Add to results
            results.append(f"✅ File {i}: {file_name} ({file_size:,} bytes)")
            total_size += file_size
            valid_files += 1
        
        # Summary
        summary = f"\n📊 Summary:\n"
        summary += f"• Total files uploaded: {len(pdf_files)}\n"
        summary += f"• Valid PDF files: {valid_files}\n"
        summary += f"• Total size: {total_size:,} bytes\n"
        summary += f"• Status: Ready for processing!\n"
        
        return "\n".join(results) + summary
    
    except Exception as e:
        return f"Error processing files: {str(e)}"

# Create the Gradio interface
with gr.Blocks(title="Researcher") as demo:
    gr.Markdown("# 📚 Research Assistant")
    gr.Markdown("Upload multiple PDF files for analysis")
    
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