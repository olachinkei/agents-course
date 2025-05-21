import weave
from openai import OpenAI

import config

weave.init(project_name=config.WEAVE_PROJECT)
client = OpenAI()


@weave.op()
def response(instructions: str, user_input: str):
    response = client.responses.create(
        model="gpt-4.1",
        instructions=instructions,
        input=user_input,
    )
    return response.output[0].content[0].text


@weave.op()
def process_transcript(transcript: str):
    summary = response("Summarize into 3-5 sentences", transcript)
    tone = response("Determine the tone of the transcript", transcript)
    return summary, tone


@weave.op()
def chapter_1_workflow():
    transcript = (
        "Interviewer: Good morning! Thank you for joining us today. "
        "Can you tell us a little about your experience with remote work?\n"
        "Interviewee: Good morning! Absolutely. I've been working remotely for the past three years, "
        "primarily as a software engineer. I enjoy the flexibility it offers, though it does come with challenges.\n"
        "Interviewer: What are some of those challenges?\n"
        "Interviewee: Communication can be tricky at times, especially across different time zones. "
        "Staying motivated and maintaining work-life balance also requires conscious effort.\n"
        "Interviewer: How do you overcome those challenges?\n"
        "Interviewee: I set clear boundaries for my work hours and make sure to take regular breaks. "
        "For communication, I rely on tools like Slack and regular video calls to stay connected with my team."
    )
    summary, tone = process_transcript(transcript)
    print(summary)
    print(tone)


if __name__ == "__main__":
    chapter_1_workflow()
