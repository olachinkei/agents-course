import json
from openai import OpenAI
from utils import tag, fn_to_schema


class MiniAgent:
    def __init__(self, instructions: str, tools: list, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.instructions, self.model = instructions, model
        self.tools = {fn.__name__: fn for fn in tools}
        self.schema = [fn_to_schema(fn) for fn in tools]
        self._seen_ids = set()  # avoid double‑printing items

    # ---------- item handler -------------------------------------------
    def _handle_item(self, item):
        if item.id in self._seen_ids:  # already processed
            return []
        self._seen_ids.add(item.id)

        if item.type == "reasoning":
            print(tag("reasoning") + "".join(item.summary))
            return []

        if item.type == "message":
            txt = "".join(p.text for p in item.content if p.type == "output_text")
            print(tag("message") + txt + tag("endmessage"))
            return []

        if item.type == "function_call":
            args = json.loads(item.arguments or "{}")
            print(tag("function_call") + f"{item.name}({json.dumps(args)})")
            result = self.tools[item.name](**args) if args else self.tools[item.name]()
            print(tag("function_output") + json.dumps(result))
            return [
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result),
                }
            ]

        return []

    # ---------- main loop ----------------------------------------------
    def run(self, user_text: str):
        print("Input:", user_text)
        turn_input = [{"role": "user", "content": user_text}]
        prev_id, items = None, []

        while turn_input:
            stream = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=self.schema,
                input=turn_input,
                previous_response_id=prev_id,
                stream=True,
                reasoning={"effort": "low"},
            )
            turn_input = []  # collect next‑turn inputs

            for event in stream:
                if event.type == "response.output_item.done":
                    turn_input += self._handle_item(event.item)
                    items.append(event.item)

                if event.type == "response.completed":
                    prev_id = event.response.id
                    # TEMP PATCH: handle any items that never streamed
                    for it in event.response.output:
                        turn_input += self._handle_item(it)
                        items.append(it)

        return items
