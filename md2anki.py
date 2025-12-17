import re
import sys
import genanki
import random

def fix_latex_delimiters(text):
    """Convert standard LaTeX delimiters to Anki format."""
    # Replace display math: $$ ... $$ -> [$] ... [/$]
    text = re.sub(r'\$\$(.+?)\$\$', r'[$]\1[/$]', text, flags=re.DOTALL)
    
    # Replace inline math: $ ... $ -> [$] ... [/$]
    # Be careful not to match $$ which was already handled
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', r'[$]\1[/$]', text)
    
    return text

def parse_markdown_to_anki(markdown_text):
    """Parse markdown Q&A format into Anki cards."""
    lines = markdown_text.split('\n')
    cards = []
    current_question = None
    current_answer_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a top-level bullet (question or answer)
        if re.match(r'^[-*]\s+', line):
            # If we have a current question and answer, save the card
            if current_question and current_answer_lines:
                answer = '\n'.join(current_answer_lines).strip()
                cards.append((current_question, answer))
                current_answer_lines = []
                current_question = None
            
            # If we don't have a question yet, this is the question
            if current_question is None:
                current_question = re.sub(r'^[-*]\s+', '', line).strip()
            else:
                # This is an answer to the previous question
                current_answer_lines.append(re.sub(r'^[-*]\s+', '', line))
        
        # Check if this is a sub-bullet (part of the answer)
        elif re.match(r'^\s+[-*]\s+', line) and current_question:
            current_answer_lines.append(line)
        
        # Regular text that's part of the answer
        elif line.strip() and current_question and current_answer_lines:
            current_answer_lines.append(line)
        
        i += 1
    
    # Don't forget the last card
    if current_question and current_answer_lines:
        answer = '\n'.join(current_answer_lines).strip()
        cards.append((current_question, answer))
    
    return cards

def create_anki_deck(cards, deck_name='My Deck', output_file='output.apkg'):
    """Create an Anki deck using genanki."""
    # Create a model (card template)
    my_model = genanki.Model(
        random.randrange(1 << 30, 1 << 31),
        'Simple Model',
        fields=[
            {'name': 'Question'},
            {'name': 'Answer'},
        ],
        templates=[
            {
                'name': 'Card 1',
                'qfmt': '{{Question}}',
                'afmt': '{{FrontSide}}<hr id="answer">{{Answer}}',
            },
        ])
    
    # Create a deck
    my_deck = genanki.Deck(
        random.randrange(1 << 30, 1 << 31),
        deck_name)
    
    # Add cards to the deck
    for question, answer in cards:
        # Fix LaTeX delimiters
        question = fix_latex_delimiters(question)
        answer = fix_latex_delimiters(answer)
        
        # Convert markdown formatting to HTML
        answer = answer.replace('\n', '<br>')
        
        # Create note
        note = genanki.Note(
            model=my_model,
            fields=[question, answer])
        
        my_deck.add_note(note)
    
    # Write deck to file
    genanki.Package(my_deck).write_to_file(output_file)
    
    print(f"Created {len(cards)} Anki cards in '{output_file}'")
    print(f"\nTo use:")
    print(f"1. Open Anki")
    print(f"2. File > Import")
    print(f"3. Select '{output_file}'")

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py input.md [deck_name] [output.apkg]")
        print("\nExample: python script.py notes.md 'Physics Ch. 1' physics.apkg")
        sys.exit(1)
    
    input_file = sys.argv[1]
    deck_name = sys.argv[2] if len(sys.argv) > 2 else 'Econometrics'
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'econometrics.apkg'
    
    # Read markdown file
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    
    # Parse and create cards
    cards = parse_markdown_to_anki(markdown_text)
    
    if not cards:
        print("Warning: No Q&A pairs found in the markdown file")
        print("\nExpected format:")
        print("- First bullet is the question")
        print("- Second bullet at same level is the answer")
        print("  - Optional sub-points in the answer")
        sys.exit(1)
    
    create_anki_deck(cards, deck_name, output_file)

if __name__ == '__main__':
    main()
