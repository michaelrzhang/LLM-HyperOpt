
import json
import openai
import re

class LLMOptimizer:
    def __init__(self, is_expert="generic", model="gpt-3.5-turbo",
        temperature = 0.0,
        max_tokens = 600,
        frequency_penalty = 0,
        use_cot = False,
    ):
        self.model = model
        self.is_expert = is_expert
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.use_cot = use_cot
        self.messages = []

        # Initialize the conversation with the given system prompt
        self.initial_config()

    def initial_config(self):
        """This expert prompt does not seem to make too much of a difference (see Appendix in paper), but conditioning on good performance is generally a good idea."""
        if self.is_expert == "generic":
            message = {"role":"system", "content": "You are a machine learning expert."}
            self.messages.append(message)

    def call_llm(self, max_retries=2):
        tries = 0
        while tries < max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    frequency_penalty=self.frequency_penalty
                )
                # make sure we have the right version
                if self.model == "gpt-4":
                    assert response.model == 'gpt-4-0613'
                elif self.model == "gpt-3.5-turbo":
                    assert response.model == "gpt-3.5-turbo-0613"
                config = response["choices"][0]["message"]["content"]
                self.messages.append({"role":"assistant", "content": config})
                return config 
            except Exception as e:
                tries += 1
                print(e)
                import time; time.sleep(30)
        print(response)
        raise Exception("Failed to call LLM, max retries exceeded")

    def _parse_raw_message(self, raw_message):
        # Parse the raw message into model source code, optimizer source code, and hparams
        json_match = re.search(r'```json\n(.*)\n```', raw_message, re.DOTALL)
        if json_match is None:
            raise Exception("Failed to parse raw message")
        params = json.loads(json_match.group(1).strip())
        assert isinstance(params, dict)
        assert "x1" in params and "x2" in params
        
        return params
    
    def parse_message(self, raw_message):
        if "Output: " in raw_message:
            raw_message = raw_message.split("Output: ")[1]
        try:
            params = json.loads(raw_message)
            params = params["x"]
        except:
            print("***Raising exception...")
            print(raw_message)
            raise Exception("Failed to parse message")
        return params
        
    
    def ask(self, prompt):
        self.messages.append({"role":"user", "content": prompt})
        raw_message = self.call_llm()
        params = self.parse_message(raw_message)
        return params
    
    def get_current_messages(self):
        return self.messages