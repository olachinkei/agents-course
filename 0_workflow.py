from openai import OpenAI

client = OpenAI()


def response(instructions: str, user_input: str):

    response = client.responses.create(
        model="gpt-4.1",
        instructions=instructions,
        input=user_input,
    )
    return response.text


def process_transcript(transcript: str):
    summary = response("Summarize into 3-5 sentences", transcript)
    tone = response("Determine the tone of the transcript", transcript)
    return summary, tone
