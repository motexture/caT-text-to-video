---
license: apache-2.0
---
# caT text to video

Conditionally augmented text-to-video model. Uses pre-trained weights from modelscope text-to-video model, augmented with temporal conditioning transformers to create a smooth transition between clips.

## Installation

### Clone the Repository

```bash
git clone https://github.com/motexture/caT-text-to-video.git
cd caT
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
python run.py
```

Visit the provided URL in your browser to interact with the interface and start generating videos.
