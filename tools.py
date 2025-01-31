from bs4 import BeautifulSoup

def write_to_file(messages: list, filename: str):
    try:
        with open(filename, 'w') as file:
            file.write('\n'.join(messages))
            return
    except Exception as e:
        print(f"Error in writting file '{filename}': {e}")
        return
    
def extract_message_content(message: str) -> str:
    soup = BeautifulSoup(message, 'html.parser')

    contemplator = ' '.join([p.get_text() for p in soup.find_all('contemplator')])
    final_answer = ' '.join([p.get_text() for p in soup.find_all('final_answer')])

    if not final_answer:
        return f"REFLECTION\n{contemplator}\n"
    else:
        return f"REFLECTION\n{contemplator}\nFINAL ANSWER\n{final_answer}\n"

