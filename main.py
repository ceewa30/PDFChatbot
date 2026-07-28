import os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from src.app import create_pdf_chatbot


def main():
    print("Hello from pdfchatbot!")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_pdf_path = os.path.normpath(os.path.join(script_dir, "data", "Agentic_AI.pdf"))

    # Loading the default PDF immediately on startup
    current_chain = [create_pdf_chatbot(pdf_path=default_pdf_path)]

    # Define frontend wrapper functions
    def process_new_pdf(file_obj):
        """Processes a new user-uploaded PDF file and replaces the current chain."""
        if file_obj is None:
            return "No file uploaded. Using existing database."

        try:
            new_chain = create_pdf_chatbot(pdf_path=file_obj.name)
            current_chain[0] = new_chain
            return "New PDF processed successfully! Memory refreshed."
        except Exception as e:
            return f"Error processing PDF: {str(e)}"

    def predict(message, history):
        """Interacts with the currently loaded LangChain RAG pipeline."""
        if not current_chain[0]:
            return "Error: No active PDF database loaded."

        formatted_history = []
        for turn in history:
            if turn['role'] == 'user':
                formatted_history.append(HumanMessage(content=turn['content']))
            elif turn['role'] == 'assistant':
                formatted_history.append(AIMessage(content=turn['content']))

        response = current_chain[0].invoke({
            "input": message,
            "chat_history": history
        })
        return response["answer"]



    # Build the Gradio Layout
    with gr.Blocks(title="PDF Chatbot") as demo:
        gr.Markdown("# 📄 AI PDF Chatbot with Memory")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Process PDF", variant="primary")
                status_output = gr.Textbox(
                    label="Status", 
                    interactive=False, 
                    value=f"Ready. Loaded baseline: {os.path.basename(default_pdf_path)}"
                    )
                
            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=predict
                )

        # Wire up the initialization logic
        upload_btn.click(
            fn=process_new_pdf, 
            inputs=[pdf_input], 
            outputs=[status_output]
        )

    print("Lunching Gradio Interface...")
    demo.launch()



if __name__ == "__main__":
    main()
