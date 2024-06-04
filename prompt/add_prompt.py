import pandas as pd
import openai

# Initialize the OpenAI client
openai.api_key = '.....'

# Load the CSV file
data = pd.read_csv('/content/drive/MyDrive/gitksan_stem_trans/Gitksan_prompt/git_word_overlap.csv')

# Function to create the prompt
def create_prompt(example):
    prompt = f"""
    You are a linguistic annotator for Gitksan language, tasked with correcting errors in glossing based on translation details and morpheme translations. Your task is to adjust errors in the stems (in lowercase) without changing the total number of morphemes or words in the gloss. Each gloss element is separated by hyphens within morphemes and spaces between words.

    Here are two examples:
    Example 1: Gitksan sentence is {example['train1_raw']}. You are provided with morpheme translations according to the dictionary: {example['train1_morpheme_gloss']}. The English translation for this sentence is: {example['train1_translation']}. The glossing pending to be revised is: {example['train1_pending_gloss']}. The corrected gloss is {example['train1_gold_gloss']}.

    Example 2: Gitksan sentence is {example['train2_raw']}. You are provided with morpheme translations according to the dictionary: {example['train2_morpheme_gloss']}. The English translation for this sentence is: {example['train2_translation']}. The glossing pending to be revised is: {example['train2_pending_gloss']}. The corrected gloss is {example['train2_gold_gloss']}.

    Now, here's the item you need to correct: Gitksan sentence is {example['test_raw']}. You are provided with morpheme translations according to the dictionary: {example['test_morpheme_gloss']}. The English translation for this sentence is: {example['test_translation']}. The glossing pending to be revised is: {example['test_pending_gloss']}. What is the corrected gloss for this sentence? You should answer in this format: The corrected gloss is: (your generated answer). x
    """
    return prompt

# Function to get the corrected gloss from ChatGPT
def get_corrected_gloss(prompt):
    completion = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    # print(completion.choices[0].message.content)
    corrected_gloss = completion.choices[0].message.content.replace("The corrected gloss is:", "").strip()[:-1]
    print(corrected_gloss)
    return corrected_gloss

# Process each test item and get the corrected gloss
with open('/content/drive/MyDrive/gitksan_stem_trans/Gitksan_prompt/git_longsub_corrected.txt', 'w') as file:
  for index, row in data.iterrows():
      example = {
          'train1_raw': row['train1_raw'],
          'train1_morpheme_gloss': row['train1_morpheme_gloss'],
          'train1_translation': row['train1_translation'],
          'train1_pending_gloss': row['train1_pending_gloss'],
          'train1_gold_gloss': row['train1_gold_gloss'],
          'train2_raw': row['train2_raw'],
          'train2_morpheme_gloss': row['train2_morpheme_gloss'],
          'train2_translation': row['train2_translation'],
          'train2_pending_gloss': row['train2_pending_gloss'],
          'train2_gold_gloss': row['train2_gold_gloss'],
          'test_raw': row['test_raw'],
          'test_morpheme_gloss': row['test_morpheme_gloss'],
          'test_translation': row['test_translation'],
          'test_pending_gloss': row['test_pending_gloss']
      }

      prompt = create_prompt(example)
      corrected_gloss = get_corrected_gloss(prompt)

      # Write the results to the text file
      file.write(f"\\t{row['test_raw']}\n")
      file.write(f"\\g {corrected_gloss}\n")
      file.write(f"\\l{row['test_translation']}\n\n")
