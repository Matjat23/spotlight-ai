# Spotlight AI

Search and tag nonprofit photo collections with vision-language models.
Built by our team at the AI4Good Hackathon 2026 (UNF FL-DSSG, Ponte Vedra Beach, October 2026).

## The problem
The Performers Academy and Dreams Come True have thousands of event photos without names or tags.
Finding the right photo for a newsletter or a donor report means scrolling for hours.

## What it does
- Writes a caption for every photo
- Adds tags from a label list made with the partners
- Lets staff search the collection by typing what they are looking for
- Runs fully on our own laptop, so no photos leave the device

## Privacy
The photos show children, some of them seriously ill. No photos, videos, captions or embeddings are
committed to this repository. The .gitignore blocks all image, video and data files.

## Setup
See [SETUP.md](SETUP.md). Short version on Windows:

```
powershell -ExecutionPolicy Bypass -File setup.ps1
```
