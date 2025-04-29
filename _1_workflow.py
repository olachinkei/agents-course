# import weave
from openai import OpenAI

import config

# weave.init(project_name=config.WEAVE_PROJECT)
client = OpenAI()

# @weave.op()
def response(instructions: str, user_input: str):
    response = client.responses.create(
        model="gpt-4.1",
        instructions=instructions,
        input=user_input,
    )
    return response.output[0].content[0].text


# @weave.op()
def process_transcript(transcript: str):
    summary = response("Summarize into 3-5 sentences", transcript)
    tone = response("Determine the tone of the transcript", transcript)
    return summary, tone


if __name__ == "__main__":
    transcript = "Hello, how are you?"
    summary, tone = process_transcript(transcript)
    print(summary)
    print(tone)
