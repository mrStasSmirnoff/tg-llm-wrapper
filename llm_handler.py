import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def query_llm(llm_client, messages):
    """
    Send the user's message (plus a system prompt, if any) to DeepSeek and return the response.
    """
    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
        )
        result_text = response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        result_text = "I'm sorry, but I couldn't process your request at this time."
    logging.info(f"LLM response: {result_text}")

    return result_text

