import re
import json
import logging
import os
from langchain.chat_models import AzureChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


os.environ["OPENAI_API_KEY"] = "f5c173e7b5a8445894254e9e703ebc30"
os.environ["OPENAI_API_VERSION"] = "2024-02-15-preview"


def initialize_chat_model(temperature=0.0):
    return AzureChatOpenAI(
        deployment_name="tcoegpt4o",
        azure_endpoint="https://llmexplorationgpt4o.openai.azure.com/",
        temperature=temperature
    )


def llm_completeness_analysis(chat_model, paragraph, summary):
    """
    Analyze the completeness of a summary compared to the paragraph and return structured JSON output.
    """
    template = (
        "You are an intuitive agent tasked with comparing a call conversation and its summary. Your goal is to assess whether the summary fully captures all the information in the call conversation. "
        "Identify the primary goal of the document and the essential points or arguments it makes."
        "Read the provided summary and compare it with the call conversation."
        "Check if all the information of the call conversation are present in the summary."
        "If any information is missing from the summary, list all the missing details in bullet points."
        "For each missing point, provide a detailed explanation of what is missing."
        "Display the exact lines from the call conversation that convey the missing information."
        "Based on your analysis, assign a score between 0 and 1 to the summary."
        "If more than two missing information is found then significantly reduce the score."
        "0 means the summary misses most or all of the crucial points."
        "1 means the summary captures all important details accurately."
        "list down the highlighted lines in separate lines instead of gpt generated way."
        "call conversation: '''{}'''\n\n"
        "Summary: '''{}'''\n\n"
        "Provide output in a JSON format with the following keys:\n"
        " - score: The completeness score between 0 and 1.\n"
        " - missing_information_details: A list of objects, each containing:\n"
        "    - missing_point: The missing point in the summary\n"
        "    - explanation: A detailed explanation of the missing point\n"
        "    - exact_lines_missing: A list of exact lines from the conversation."
    ).format(paragraph, summary)

    response = chat_model.call_as_llm(template)

    # Clean up the response in case it includes code block markers (```)
    cleaned_response = re.sub(r'```json\n|\n```', '', response).strip()

    try:
        # Attempt to parse response into JSON
        result = json.loads(cleaned_response)
    except json.JSONDecodeError:
        # Handle cases where the response isn't valid JSON
        logging.error("Response is not in JSON format. Returning raw text.")
        result = {
            "error": "Response is not in valid JSON format",
            "raw_response": response
        }

    return result

def analyze_completeness(paragraph,summary, temperature=0.0):
    """ Analyze the text for misogyny using the LLM. """
    chat_model = initialize_chat_model(temperature)
    # logging.info(f"Analyzing text: {text}")

    llm_judgment = llm_completeness_analysis(chat_model, paragraph,summary)
    return llm_judgment




# Example usage
if __name__ == "__main__":

    file_path = 'interrupt.txt'
    with open(file_path, 'r') as file:
        summary = file.read()

    file_path = 'main.txt'
    with open(file_path, 'r') as file:
        paragraph = file.read()

    completeness_analysis = analyze_completeness(paragraph, summary)
    print(json.dumps(completeness_analysis, indent=4))  # Pretty-print the JSON response

