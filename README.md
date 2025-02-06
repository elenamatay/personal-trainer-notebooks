# AI-powered Gym Personal Trainer

## Overview
This repo showcases a project that aims to provide personal recommendations on strength exercises. Such recommendations goal is to help users improve their training through evaluating technique or form, their tempo, range of movement, and potentially could even detect the fatigue.

## Key Files

Since this project was on a public website and could be again in the future, I only published a couple of code sample files here instead of the whole repo. The main files to explore in this repo are five notebooks:
- `1. First tests - Demo.ipynb` - First tests to see to what extent Gemini's multi-modal capabilities are able to correct the form of a user performing a strength exercise, by comparing a video of their performance to a reference video which explains the right technique and common mistakes.
- `2. Intra-series detection.ipynb` - Check up to what extent Gemini is capable of identifying difference between reps in one same exercise series (e.g. changes in velocity, range of movement, etc.). This could be later used to evaluate the user's fatigue.
- `3. Scrapper enhancer.ipynb` - Create a JSON structure for the most complex pages in strengthlog -[example](https://www.strengthlog.com/deadlift/)-, for which a traditional scrapper was not enough to produce a useful, well-structured, output format. This JSONs will later be used as reference data.
- `4. Improving Exercise Classification.ipynb` - Try to improve Gymally's current main issue: exercise classification (around 40% accuracy with a total dataset of ~250 exercises) with prompt engineering.
- `5. Vector Search (multimodal embeddings) for Exercise classification.ipynb` - Try to improve Gymally's current main issue: exercise classification (around 40% accuracy with a total dataset of ~250 exercises). In this case, we want to try using a RAG-based approach with multi-modal embeddings -of strength exercise videos- for exercise classification.


## How to Use
1. Clone the repository and move to the relevant folder:
```
git clone https://github.com/elenamatay/baby-names-ai.git](https://github.com/elenamatay/ai-playground.git
cd personal/side-projects/gym-personal-trainer
```

2. Set up the environment by installing the required dependencies:
```
pip install -r requirements.txt
```

3. Run the Jupyter notebooks:
- Select the notebook you're most interested in.
- Run it end-to-end.

**⚠️⚠️Warning⚠️⚠️:** This notebook may imply high Google Cloud consumption charges, so be mindful before running the notebooks!
