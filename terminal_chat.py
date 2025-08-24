#!/usr/bin/env python3
"""
Terminal-based chat interface for Convex Optimization Curves
Fixes output formatting issues
"""

import sys
from convex_optimization_curves import OptimizationChatbot

class TerminalChat:
    def __init__(self):
        self.chatbot = OptimizationChatbot()
        self.history = []
        
    def format_response(self, response: str) -> str:
        """Format response for terminal display"""
        # Replace markdown bold with terminal formatting
        formatted = response.replace("**", "")
        formatted = formatted.replace("✓", "[✓]")
        formatted = formatted.replace("✗", "[✗]")
        
        # Add proper line breaks
        lines = formatted.split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip().startswith("•"):
                formatted_lines.append("  " + line)
            elif line.strip().startswith("-"):
                formatted_lines.append("    " + line)
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    def print_welcome(self):
        """Print welcome message"""
        print("\n" + "="*60)
        print("CONVEX OPTIMIZATION CURVES - INTERACTIVE CHAT")
        print("Based on 'Are Convex Optimization Curves Convex?'")
        print("by Guy Barzilai and Ohad Shamir")
        print("="*60)
        print("\nType your questions about convex optimization curves.")
        print("Commands: 'help', 'examples', 'clear', 'history', 'quit'\n")
    
    def print_help(self):
        """Print help information"""
        print("\n" + "-"*40)
        print("HELP - Available Topics:")
        print("-"*40)
        print("You can ask about:")
        print("  • Convex optimization curves")
        print("  • Step size effects")
        print("  • Main theorems")
        print("  • Gradient norm behavior")
        print("  • Examples and counter-examples")
        print("\nExample questions:")
        print("  - What are convex optimization curves?")
        print("  - How does step size affect the curve?")
        print("  - What does theorem 1 say?")
        print("  - Show me an example")
        print("-"*40 + "\n")
    
    def print_examples(self):
        """Print example queries"""
        print("\n" + "-"*40)
        print("EXAMPLE QUESTIONS:")
        print("-"*40)
        examples = [
            "What makes an optimization curve convex?",
            "Why does step size matter?",
            "What happens when eta is between 1/L and 2/L?",
            "How do gradient norms behave?",
            "What's the difference between theorems 1 and 2?",
            "Give me a practical example"
        ]
        for i, ex in enumerate(examples, 1):
            print(f"{i}. {ex}")
        print("-"*40 + "\n")
    
    def run(self):
        """Main chat loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit' or user_input.lower() == 'exit':
                    print("\nGoodbye! Thanks for learning about convex optimization curves.")
                    break
                
                elif user_input.lower() == 'help':
                    self.print_help()
                    continue
                
                elif user_input.lower() == 'examples':
                    self.print_examples()
                    continue
                
                elif user_input.lower() == 'clear':
                    print("\033[H\033[J", end="")  # Clear screen
                    self.print_welcome()
                    continue
                
                elif user_input.lower() == 'history':
                    if not self.history:
                        print("\nNo chat history yet.")
                    else:
                        print("\n" + "-"*40)
                        print("CHAT HISTORY:")
                        print("-"*40)
                        for i, (q, a) in enumerate(self.history, 1):
                            print(f"\n{i}. Q: {q}")
                            print(f"   A: {a[:100]}...")
                        print("-"*40)
                    continue
                
                # Get response from chatbot
                response = self.chatbot.answer_question(user_input)
                formatted_response = self.format_response(response)
                
                # Print response
                print("\nAssistant:", formatted_response)
                
                # Save to history
                self.history.append((user_input, response))
                
            except KeyboardInterrupt:
                print("\n\nUse 'quit' to exit properly.")
                continue
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.")

def main():
    """Main entry point"""
    chat = TerminalChat()
    chat.run()

if __name__ == "__main__":
    main()