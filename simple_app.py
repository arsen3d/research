import gradio as gr

def greet(name):
    """Simple greeting function"""
    if name:
        return f"Hello, {name}! Welcome to Gradio! 🎉"
    else:
        return "Hello! Please enter your name to get a personalized greeting. 👋"

# Create a simple interface
iface = gr.Interface(
    fn=greet,
    inputs=gr.Textbox(placeholder="Enter your name"),
    outputs="text",
    title="Simple Research Assistant",
    description="A simple test interface"
)

if __name__ == "__main__":
    iface.launch(share=True)
