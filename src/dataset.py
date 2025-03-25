from datasets import Dataset, load_dataset


RICK_SYSTEM_PROMPT = """You are an interdimensional genius scientist named Rick Sanchez.
Be brutally honest, use sharp wit, and sprinkle in some scientific jargon.
Don't shy away from dark humor or existential truths, but always provide a solution (even if it's unconventional)."""

CLEANING_PROMPT = """
Your task is to fix some dialogue transcripts you are going to receive.
The idea is to remove references to actions / context, removing any
incorrect symbols. Here are some examples:

Input: stumbles in drunkenly, and turns on the lights. Morty! You gotta come on. Jus'... you gotta come with me.
Output: Morty! You gotta come on. Jus'... you gotta come with me.

Input: rubs his eyes. What, Rick? What's going on?
Output: What, Rick? What's going on?
"""

def load_rick_and_morty_dataset():
    """Loads the Rick and Morty transcript dataset.

    This function loads the Rick and Morty transcript dataset from the Hugging Face
    datasets hub, specifically from the "Prarabdha/Rick_and_Morty_Transcript" dataset.

    Returns:
        datasets.Dataset: A dataset containing Rick and Morty episode transcripts.
            The dataset includes columns for dialogue, speaker, and episode information.
    """
    dataset = load_dataset("Prarabdha/Rick_and_Morty_Transcript", split="train")
    return dataset

def create_conversation_pairs(dataset):
    """Creates conversation pairs from the Rick and Morty transcript dataset.

    This function processes the dataset to create conversation pairs where a non-Rick character
    speaks followed by Rick's response. Each conversation includes a system prompt defining
    Rick's character.

    Args:
        dataset (datasets.Dataset): The Rick and Morty transcript dataset containing dialogue
            and speaker information.

    Returns:
        datasets.Dataset: A new dataset containing conversation pairs in the format:
            {
                "conversations_raw": [
                    {"from": "system", "value": system_prompt},
                    {"from": "human", "value": non_rick_dialogue},
                    {"from": "gpt", "value": rick_dialogue}
                ]
            }
    """
    new_rows = []
    for i in tqdm(range(len(dataset) - 1)):
        current_row = dataset[i]
        next_row = dataset[i + 1]

        if current_row["speaker"] != "Rick" and next_row["speaker"] == "Rick":
            if current_row["episode no."] == next_row["episode no."]:
                new_rows.append(
                    {
                        "conversations_raw": [
                            {"from": "system", "value": RICK_SYSTEM_PROMPT.strip()},
                            {"from": "human", "value": current_row["dialouge"].strip()},
                            {"from": "gpt", "value": next_row["dialouge"].strip()},
                        ]
                    }
                )

    return Dataset.from_list(new_rows)


def clean_dialogue(client, text, system_prompt):
    """Clean a single dialogue using OpenAI API.

    Args:
        client (openai.OpenAI): The OpenAI client instance to use for API calls.
        text (str): The dialogue text to clean.
        system_prompt (str): The system prompt providing instructions for cleaning.

    Returns:
        str: The cleaned dialogue text with actions/context removed.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text.strip()},
        ],
    )
    return response.choices[0].message.content
