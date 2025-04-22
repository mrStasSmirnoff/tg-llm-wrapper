import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def query_llm(llm_client, user_message, system_prompt=None):
    """
    Send the user's message (plus a system prompt, if any) to DeepSeek and return the response.
    """
    if not system_prompt:
        system_prompt = "You are a helpful assistant. Answer the user's questions as accurately as possible."

    try:
        response = llm_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            stream=False,
        )
        result_text = response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error querying LLM: {e}")
        result_text = "I'm sorry, but I couldn't process your request at this time."
    logging.info(f"LLM response: {result_text}")

    return result_text

