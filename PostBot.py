# ===== Importing libraries =====
from instagrapi import Client
from dotenv import load_dotenv
import os
import random as r
import time
import torch
from diffusers import StableDiffusionPipeline
import ollama
import schedule
from datetime import datetime


# ===== Client and Setup =====
client = Client()
load_dotenv()
USERNAME = os.getenv("EMAIL")
PASSWORD = os.getenv("PASSWORD")

model_id = "runwayml/stable-diffusion-v1-5"
ollama_model = "qwen3.5:2b"

if torch.cuda.is_available():
    print("Loading GPU version...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
else:
    print("Loading CPU version...")
    pipe = StableDiffusionPipeline.from_pretrained(model_id)


# ===== Login function that handles sessions =====
def login_user():
    session_file = "session.json"
    
    if os.path.exists(session_file):
        client.load_settings(session_file)

        try:
            client.login(USERNAME, PASSWORD)
            client.get_timeline_feed()
            return
            
        except Exception as e:
            old_settings = client.get_settings()
            client.set_settings({})
            client.set_uuids(old_settings["uuids"])

    client.login(USERNAME, PASSWORD)
    client.dump_settings(session_file)


# ===== Image generation function that creates post image and returns post caption =====
def generate_image() -> str:
    # Prompt for prompt generation for image
    prompt = ("""
        Write a prompt where you ask for a generated image of cool futuristic settings,
        such as buildings, landscapes, shops, etc... Start the prompt with 'Generate an image
        of ...'
        """)

    # Generate prompt
    try:
        response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": prompt}], think=False)
        text = response["message"]["content"]
    except Exception as e:
        print("Could not generate prompt...")
        print(e)
        return ""

    # Actual image generation
    try:
        print("Generating image...")
        image = pipe(text).images[0]
        image.save("Post.png")
        print("Generation done!")
    except Exception as e:
        print("Could not generate image :(")
        return ""

    # Generate caption for image
    return text[21:]


# ===== Session logic =====
def run_session():
    # Login first (obviously)
    login_user()

    # Create the actual post
    time.sleep(r.randint(45, 120))
    print("Creating post...")
    post_caption = generate_image()
    client.photo_upload("Post.png", post_caption)
    print("Post created!")


# ===== Daily scheduling: 1 or 2 sessions, at random times =====
def schedule_todays_sessions():
    schedule.clear("sessions")  # Drop yesterday's slots before creating today's
    num_sessions = r.randint(1, 2)
    chosen_times = []

    while len(chosen_times) < num_sessions:
        # Random time of day, kept within waking hours to look natural
        hour = r.randint(8, 22)
        minute = r.randint(0, 59)
        slot = f"{hour:02d}:{minute:02d}"
        # Avoid two sessions landing in the same hour
        if all(abs(hour - int(t[:2])) >= 2 for t in chosen_times):
            chosen_times.append(slot)
    for slot in chosen_times:
        schedule.every().day.at(slot).do(run_session).tag("sessions")
    print(f"Scheduled {num_sessions} session(s) today at: {', '.join(sorted(chosen_times))}")


# ===== Entry point =====
schedule.every().day.at("00:01").do(schedule_todays_sessions)
schedule_todays_sessions()

while True:
    schedule.run_pending()
    time.sleep(30) 