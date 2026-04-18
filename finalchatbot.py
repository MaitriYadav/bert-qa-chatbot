import tkinter as tk
from tkinter import scrolledtext, messagebox, StringVar, ttk
import requests
import threading
import json

class BertQABotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("BERT QA Chatbot")
        self.root.geometry("800x700")
        self.root.configure(bg="#f5f5f5")
        
        # API Configuration
        self.API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
        
        # Create main frames
        self.create_widgets()
        
        # Initialize conversation
        self.waiting_for_context = True
        self.current_context = ""
        
    def create_widgets(self):
        # Token frame
        self.token_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.token_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(self.token_frame, text="Hugging Face API Token:", bg="#f5f5f5").pack(side=tk.LEFT)
        self.token_entry = tk.Entry(self.token_frame, width=40, show="*")
        self.token_entry.pack(side=tk.LEFT, padx=10)
        
        # Mode selection
        self.mode_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.mode_frame.pack(fill=tk.X, padx=20, pady=5)
        
        self.status_label = tk.Label(self.mode_frame, 
                                    text="Waiting for context... Please provide a text passage first.",
                                    bg="#f5f5f5", fg="#FF6347")
        self.status_label.pack(side=tk.TOP, pady=5)
        
        # Context frame
        self.context_frame = tk.LabelFrame(self.root, text="Context (Paste text for BERT to analyze)", bg="#f5f5f5")
        self.context_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.context_text = scrolledtext.ScrolledText(self.context_frame, height=10, wrap=tk.WORD)
        self.context_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.set_context_button = tk.Button(self.context_frame, text="Set as Context", 
                                           command=self.set_context, bg="#4CAF50", fg="white")
        self.set_context_button.pack(pady=5)
        
        # Chat display frame
        self.chat_frame = tk.LabelFrame(self.root, text="Q&A Session", bg="#f5f5f5")
        self.chat_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, height=10, wrap=tk.WORD)
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.chat_display.config(state=tk.DISABLED)
        
        # User input frame
        self.input_frame = tk.Frame(self.root, bg="#f5f5f5")
        self.input_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.user_input = tk.Entry(self.input_frame, width=70)
        self.user_input.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", lambda event: self.send_question())
        
        self.send_button = tk.Button(self.input_frame, text="Ask", command=self.send_question,
                                    bg="#2196F3", fg="white")
        self.send_button.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Focus on context input initially
        self.context_text.focus_set()
    
    def set_context(self):
        context = self.context_text.get(1.0, tk.END).strip()
        if not context:
            messagebox.showerror("Error", "Please provide a context text.")
            return
            
        self.current_context = context
        self.waiting_for_context = False
        self.status_label.config(text="Context set! You can now ask questions about this text.", fg="#4CAF50")
        
        # Add to chat display
        self.update_chat_display("System", "Context has been set. You can now ask questions about this text.")
        
        # Enable and focus on question input
        self.user_input.config(state=tk.NORMAL)
        self.user_input.focus_set()
    
    def update_chat_display(self, sender, message):
        self.chat_display.config(state=tk.NORMAL)
        
        # Format based on sender
        if sender == "You":
            self.chat_display.insert(tk.END, f"\n{sender}: ", "user")
            self.chat_display.tag_config("user", foreground="blue", font=("Arial", 10, "bold"))
        elif sender == "BERT":
            self.chat_display.insert(tk.END, f"\n{sender}: ", "bert")
            self.chat_display.tag_config("bert", foreground="green", font=("Arial", 10, "bold"))
        else:
            self.chat_display.insert(tk.END, f"\n{sender}: ", "system")
            self.chat_display.tag_config("system", foreground="gray", font=("Arial", 10, "italic"))
        
        # Add the message
        self.chat_display.insert(tk.END, message)
        
        # Scroll to the bottom
        self.chat_display.see(tk.END)
        self.chat_display.config(state=tk.DISABLED)
    
    def send_question(self):
        # Check if context is set
        if self.waiting_for_context:
            messagebox.showinfo("Info", "Please set a context first by clicking 'Set as Context'.")
            return
            
        # Get user question
        question = self.user_input.get().strip()
        if not question:
            return
            
        # Get API token
        api_token = self.token_entry.get().strip()
        if not api_token:
            messagebox.showerror("Error", "Please enter your Hugging Face API token")
            return
            
        # Add question to chat
        self.update_chat_display("You", question)
        
        # Clear input
        self.user_input.delete(0, tk.END)
        
        # Disable send button
        self.send_button.config(state=tk.DISABLED)
        self.status_var.set("Thinking...")
        
        # Process in background to keep UI responsive
        threading.Thread(target=self.get_answer, args=(question, api_token), daemon=True).start()
    
    def get_answer(self, question, api_token):
        try:
            # Prepare headers
            headers = {"Authorization": f"Bearer {api_token}"}
            
            # Prepare payload for BERT QA
            payload = {
                "inputs": {
                    "question": question,
                    "context": self.current_context
                }
            }
            
            # Make API call
            response = requests.post(self.API_URL, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract answer
                if isinstance(result, dict):
                    answer = result.get("answer", "I couldn't find an answer in the provided context.")
                    score = result.get("score", 0)
                    
                    # Format response with confidence
                    formatted_answer = f"{answer} (Confidence: {score:.2%})"
                else:
                    formatted_answer = "Unexpected response format from API."
            else:
                # Handle errors
                if response.status_code == 503:
                    formatted_answer = "Model is loading. Please try again in a few moments."
                else:
                    formatted_answer = f"Error: {response.status_code} - {response.text}"
                    
        except Exception as e:
            formatted_answer = f"Error: {str(e)}"
        
        # Update UI (must be done in main thread)
        self.root.after(0, lambda: self.update_ui_with_response(formatted_answer))
    
    def update_ui_with_response(self, response):
        # Add to chat
        self.update_chat_display("BERT", response)
        
        # Update status
        self.status_var.set("Ready")
        
        # Re-enable send button
        self.send_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    root = tk.Tk()
    app = BertQABotApp(root)
    root.mainloop()