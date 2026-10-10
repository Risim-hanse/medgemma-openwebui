import os

import chainlit as cl
import httpx

# Change this if your FastAPI backend is running on another machine.
API_URL = os.getenv("MEDGUARD_API_URL", "http://127.0.0.1:8000/chat")


@cl.on_chat_start
async def start():
    await cl.Message(
        content=(
            "# Welcome to MedGuard 💊\n"
            "Ask a medication-safety question in simple language.\n\n"
            "**Important:** MedGuard's AI answers are informational, not a verified "
            "drug-interaction check. Do not change or stop prescribed medicines based "
            "on a chatbot response; ask a pharmacist or qualified clinician."
        )
    ).send()


@cl.on_message
async def handle_message(message: cl.Message):
    question = message.content.strip()

    if not question:
        await cl.Message(content="Please type a question first.").send()
        return

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(API_URL, json={"message": question})

        if response.is_error:
            await cl.Message(
                content=(
                    f"The MedGuard API returned an error (HTTP {response.status_code}). "
                    "Check that your FastAPI server is running and that the API URL is correct."
                )
            ).send()
            return

        data = response.json()
        reply = data.get("reply")

        if not reply:
            await cl.Message(
                content="The API responded, but no 'reply' field was found in its JSON response."
            ).send()
            return

        await cl.Message(content=reply).send()

    except httpx.ConnectError:
        await cl.Message(
            content=(
                "I couldn't connect to the MedGuard API. Make sure your FastAPI server "
                "is running at " + API_URL
            )
        ).send()
    except httpx.TimeoutException:
        await cl.Message(
            content="Gemma took too long to respond. Please try again."
        ).send()
    except Exception as exc:
        await cl.Message(
            content=f"Something went wrong: {type(exc).__name__}. Check the terminal for details."
        ).send()

