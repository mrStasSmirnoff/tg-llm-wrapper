"""

"""
import os
import logging
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters
)
from functools import partial
from openai import OpenAI
from llm_handler import query_llm
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"

LOCAL_LANG = json.loads(Path("locale.json").read_text(encoding="utf-8"))

## TODO: move to helpers
def t(key: str, user_lang: str = "en") -> str:
    """
    Translate (t) a key to the specified language
    """
    lang = user_lang if user_lang in ("ru", "en") else "en"
    return LOCAL_LANG.get(key, {}).get(lang, LOCAL_LANG.get(key, {}).get("en", ""))


def get_lang(update: Update) -> str:
    """Detect user language, default to 'en'."""
    code = update.effective_user.language_code or ""
    return "ru" if code.startswith("ru") else "en"


## /start COMMAND
## inline button example: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/inlinekeyboard.py
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Sends a welcome message when the command /start is issued
    """
    lang = get_lang(update) # TODO: write a wrapper for lang
    logger.info(f"Received /start command from {update.effective_user.first_name}")

    keyboard = [
        [InlineKeyboardButton(t("button_edit", lang), callback_data='edit_system_prompt')],
        [InlineKeyboardButton(t("button_reset", lang), callback_data='reset_ctx')],
        [InlineKeyboardButton(t("button_show", lang), callback_data='show_prompt')]
    ]
    # a lightweight container that tells Telegram how to arrange one or more buttons into rows/columns
    reply_markup = InlineKeyboardMarkup(keyboard)
    welcome_text = (
        "Hello! I'm your LLM bot powered by DeepSeek.\n\n"
        "Send me any text, and I'll forward it to the LLM.\n\n"
        "To set a custom system prompt, click the button below or use:\n"
        "/systemprompt <Your prompt here>"
    )

    await update.message.reply_text(
        t("welcome", lang),
        reply_markup=reply_markup,
        parse_mode="Markdown")

## Inline Button Callback
## relevant docs: https://core.telegram.org/bots/2-0-intro
## https://core.telegram.org/bots/api/

async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE)-> None:
    """
    This is called whenever a user clicks an inline button.
    We look at 'callback_data' to see what the user clicked.
    """
    lang = get_lang(update)
    query = update.callback_query
    await query.answer()

    if query.data == "edit_system_prompt":
        # If the user clicks "Edit System Prompt," instruct them to use /systemprompt
        await query.message.reply_text(
            t("ask_prompt", lang),
            parse_mode="Markdown"
        )
    elif query.data == "reset_ctx":
        context.user_data['history'] = []
        await query.message.reply_text(
            t("ctx_reset", lang),
            parse_mode="Markdown"
        )
    elif query.data == "show_prompt":
        prompt = context.user_data.get("system_prompt")
        if prompt:
            text = t("show_prompt", lang).format(prompt=prompt)
        else:
            text = t("no_prompt", lang)
        await query.message.reply_text(text, parse_mode="Markdown")


## System Prompt Command
async def set_system_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE)-> None:
    """
    /systemprompt <prompt>:
    Set a custom system prompt for the LLM.
    its stored in the context.user_data so each user has their own system prompt
    """
    lang = get_lang(update)
    prompt_text = ' '.join(context.args).strip()
    if not prompt_text:
        await update.message.reply_text(
            t("ask_prompt", lang),
            parse_mode="Markdown"
        )
        return

    context.user_data["system_prompt"] = prompt_text
    await update.message.reply_text(
        f"{t('prompt_saved', lang)}\nYour current system prompt(Системный промпт):\n{prompt_text}",
        parse_mode="Markdown"
    )

# Current system prompt
async def show_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lang = get_lang(update)
    prompt = context.user_data.get("system_prompt", None)
    if prompt:
        text = t("show_prompt", lang).format(prompt=prompt)
    else:
        text = t("no_prompt", lang)
    await update.message.reply_text(text, parse_mode="Markdown")


async def reset_context_command(update: Update, context: ContextTypes.DEFAULT_TYPE)-> None:
    """
    /resetcontext:
    Reset the context for the current user.
    """
    lang = get_lang(update)
    context.user_data['history'] = []
    await update.message.reply_text(
        t("ctx_reset", lang),
        parse_mode="Markdown"
    )
    logger.info(f"Context reset for user {update.effective_user.first_name}")


## Help Command (/help)
async def help_command(update, context):
    """
    Help command to show usage info
    """
    lang = get_lang(update)
    await update.message.reply_text(t("help", lang), parse_mode="Markdown")

## User Message Handler (history & LLM call)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE, llm_client):
    """
    Handle incoming messages and forward them to the LLM.
    """
    lang = get_lang(update)
    user_message = update.message.text
    user_id = update.effective_user.id
    logger.info(f"Message from user {user_id}: {user_message}")

    history: list[dict] = context.user_data.get("history", [])
    history.append({"role": "user", "content": user_message})


    user_system_prompt = context.user_data.get("system_prompt", "You are a helpful assistant")
    if user_system_prompt:
        system_msg = {"role": "system", "content": user_system_prompt}
        messages_to_llm = [system_msg] + history
    else:
        messages_to_llm = history

    MAX_HISTORY_LENGTH = 40
    if len(history) > MAX_HISTORY_LENGTH:
        history = history[-MAX_HISTORY_LENGTH:]
        messages_to_llm = [{"role": "system", "content": user_system_prompt}] + history
        logger.info(f"History truncated to the last {MAX_HISTORY_LENGTH} messages.")
        await update.message.reply_text(
            t("history_trim", lang),
            parse_mode="Markdown"
        )

    #llm_client = context.bot_data.get("deepseek_client")
    if not llm_client:
        await update.message.reply_text(
            "Sorry, I can't contact DeepSeek right now. Please try again later"
        )
        logger.error("DeepSeek client not found in bot data.")
        return

    response = query_llm(llm_client, messages_to_llm)

    history.append({"role": "assistant", "content": response})
    context.user_data["history"] = history

    await update.message.reply_text(response)


## Deepseek client docs: https://api-docs.deepseek.com/

def main():
    """
    This function:
      1) Loads environment variables.
      2) Creates the DeepSeek client using your credentials.
      3) Builds the Telegram bot's application object.
      4) Registers all commands/handlers.
      5) Starts polling for updates.
    """
    deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)

    if TELEGRAM_BOT_TOKEN is None:
        logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
        sys.exit(1)

    # Build TG bot application
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    # Store the DeepSeek client in the application context
    application.bot_data["deepseek_client"] = deepseek_client

    # Register handlers
    # /start command
    application.add_handler(CommandHandler('start', start_command))

    # /help
    application.add_handler(CommandHandler('help', help_command))

    # /systemprompt <text>
    application.add_handler(CommandHandler("systemprompt", set_system_prompt))

    # /reset_ctx
    application.add_handler(CommandHandler("resetcontext", reset_context_command))

    # /showprompt
    application.add_handler(CommandHandler("showprompt", show_prompt_command))

    #Inline button callback
    application.add_handler(CallbackQueryHandler(callback_query_handler))

    # Catch all other messages, this is where we handle user messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,
                                        partial(handle_message,
                                                llm_client=deepseek_client)))

    try:
        logger.info("Starting bot...")
        application.run_polling()
    except TelegramError as e:
        logger.error(f"Telegram error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main()