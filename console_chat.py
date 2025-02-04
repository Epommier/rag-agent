from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from agent import generate_answer, RagState, retrieve_references, generate_question
from tools import write_to_file

class ConsoleChat:
    def __init__(self):
        self.console = Console()

    def display_message(self, message: str, sender: str = "Agent"):
        """
        Display a message in the console with markdown support.

        Args:
            message (str): The message to display.
            sender (str): The sender of the message (default: "Agent").
        """
        if not message:
            return

        self.console.print(f"[bold blue]{sender}:[/bold blue]")
        markdown_message = Markdown(message)
        self.console.print(markdown_message)

    def get_user_input(self, prompt: str = "You: ") -> str:
        """
        Get input from the user.

        Args:
            prompt (str): The prompt to display to the user (default: "You: ").

        Returns:
            str: The user's input.
        """
        return Prompt.ask(prompt)

def chat_with_agent():
    chat = ConsoleChat()
    chat.display_message("# Welcome to the Chat!", "System")

    # Create a RagState object with the user's question
    state: RagState = {
        "references": [],
        "messages": [],
        "question": "",
        "queries": [],
        "response": ""
    }

    while True:
        user_input = chat.get_user_input()
        if user_input.lower() in ["exit", "quit"]:
            chat.display_message("Goodbye!", "System")
            break

        if not state["question"]:
            chat.display_message("Generating answer...", "System")
            state["question"] = user_input

            # Generate queries
            state |= generate_question(state)

            # Retrieve references
            state |= retrieve_references(state)

        else:
            state["messages"] = {"content": user_input, "role": "user"}

        # Generate an answer using the agent
        state |= generate_answer(state)

        # Display the agent's response
        chat.display_message(state["messages"][-1]["content"], "Agent")
        write_to_file([state["messages"][-1]["content"]], "data\\output\\response.md")

if __name__ == "__main__":
    chat_with_agent() 