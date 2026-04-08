from fastapi import FastAPI
from src.env import PRReviewEnv
from src.models import PRReviewAction

app = FastAPI()
env = PRReviewEnv(task="easy")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(task: str = "easy"):
    global env
    env = PRReviewEnv(task=task)
    obs = env.reset()
    return obs.model_dump()


@app.post("/step")
def step(action: PRReviewAction):
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


@app.get("/state")
def state():
    return env.state()
