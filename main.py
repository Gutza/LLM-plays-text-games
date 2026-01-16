from jericho import FrotzEnv
from openai import OpenAI
import os

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class LLMAgent:
    def __init__(self, client):
        self.client = client
        self.model = "gpt-5-mini"
        self.game_history = [{
            "role": "system",
            "content": "You are playing the game Zork I. You are the player and you are trying to win the game. Commands should be short and concise (never more than one short sentence, typically just one or two words). You are connected directly to the game engine."
        }]
    
    def get_action(self, observation):
        self.game_history.append({"role": "user", "content": observation})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.game_history
        )
        model_response = response.choices[0].message.content
        self.game_history.append({"role": "assistant", "content": model_response})
        return model_response

llm_agent = LLMAgent(client)

env = FrotzEnv("games/zork1.z5") 
observation, info = env.reset()

print(observation)

# Your loop
while not env.game_over():
    # 1. Send observation to LLM
    # 2. Get command back (e.g., "open mailbox")
    command = llm_agent.get_action(observation)

    print("LLM command: ", command)
    print("<"*70)
    
    # 3. Step the game
    observation, reward, done, info = env.step(command)
    print(observation)
    print(f"Reward: {reward}")
    print(f"Done: {done}")
    print(f"Info: {info}")
    print(">"*70)