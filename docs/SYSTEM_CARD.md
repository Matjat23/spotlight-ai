# spotlight.io: system card and data-governance note

AI4Good Hackathon 2026, Problem 2 (Dreams Come True), Track B: semantic search and curation.
Team: Matty Jatta, Isak Rodung, Joshua.

spotlight.io lets Dreams Come True staff find photos and video moments by describing them in plain
words, pivot to the same event, place or person, and accept or correct what the AI wrote. It runs
on one laptop. No photo, video or text is sent to a cloud service.

## 1. Models

All models are open and run locally on a laptop GPU (RTX 4070, 8 GB).

| Model | Job | Why we chose it | License |
|---|---|---|---|
| Qwen2.5-VL-3B-Instruct, 4-bit | Writes the caption and picks activity, location, clothing, objects and visible moments as JSON | Followed our rules best in our comparison with BLIP and Florence-2: no guessed relationships, knew mascots, structured output. About 5 s per photo. | Qwen Research License (non-commercial) |
| CLIP ViT-B/32 | Image vector for search, zero-shot labels for activity and location | Fast, finds photos by what is visible | MIT |
| all-MiniLM-L6-v2 | Text vector of the description, search by meaning | Small and fast, understands that "dinner" and "eating at a table" are close | Apache 2.0 |
| InsightFace buffalo_l | Groups faces into anonymous groups P01, P02 for the "same person" pivot | Accurate face grouping, approved by the organizers (see section 5) | Non-commercial research |
| Whisper small | Writes down what is said in videos, with timestamps | Runs offline, many languages | MIT |

Search combines three rankings (CLIP image, MiniLM text, BM25 keywords) with Reciprocal Rank Fusion.
Events are grouped by date, or by agglomerative clustering of image vectors when a photo has no date.
Videos are cut into frames at scene changes, and each frame is described like a photo.

We tested and replaced BLIP (invented objects, guessed family roles) and tested Florence-2 (long
captions that guess feelings). The comparison is in compare_models.py and our model sheet.

## 2. Known limitations

- Captions can still invent small details or miss small objects. Staff can edit every field.
- CLIP is weak on small details and shirt colors.
- Face grouping is less accurate for young children, some skin tones, people seen from the side,
  and children who grow over the years. Mascot heads are sometimes detected as faces.
- Video moments are described as still frames, so motion ("runs towards the mascot") is not
  captured. Whisper can mishear words and sometimes "hears" words in music; we filter repeated
  and impossibly fast lines.
- Search results: [fill in Recall@5 from evaluate.py once the test set has its correct photos].
- Tested on 127 photos and 9 videos. Speed and accuracy on the full archive are not measured yet.

## 3. Out-of-distribution handling and uncertainty

- Labels come from a fixed list we can change with DCT. If Qwen answers with a label outside the
  list, CLIP picks the closest label instead, and the app shows the confidence ("AI 62%").
- A search only returns photos that are close enough to the query. A query about something not in
  the archive returns few or no results instead of the whole archive in random order.
- Every record starts as "AI suggestion". Staff accept or edit it, and the status changes to
  accepted or edited. Questions the AI should not guess (is this a family, is this the wish moment)
  are only answered by staff.
- Photos turned sideways are detected and turned upright before they are described.

## 4. Bias and privacy risks

- The captioner may describe people by appearance in a biased way. We ask it for clothing and
  visible actions only, never age, ethnicity, health, disability or relationships, and we remove
  relationship words ("mother", "siblings") from captions.
- The AI never names anyone and never guesses emotions. "Moments" are visible actions only
  (smiling, hugging, cheering).
- Face grouping can merge two different people or split one person. The groups are anonymous
  and only used to show "more photos of this person" to staff.
- Spoken names in videos become searchable through Whisper. This is a known risk; a name filter
  before saving is the next step.

## 5. Data-governance note

**Consent.** Each photo has a photo-release status: unknown, internal only, or OK for marketing.
The app warns before a download when there is no signed release. Staff set this field; the AI
never does.

**Privacy and data minimisation.**
- Everything runs on 127.0.0.1. No photo, video, caption or transcript is sent to a cloud service.
- Original file names can contain children's names, so the app never shows them, downloads are
  named spotlight_photo_12.jpg, and the CSV export and our sample output have no file names.
- The app log stores counts and timings only, never search words, captions or file names.
- Faces: the brief asks to avoid biometric face tracking. We asked the organizers on site and they
  approved face grouping for this event, with no identity research. We keep it minimal: faces
  become anonymous groups (P01, P02), no names are attached, no face images are stored, only the
  position of each face in its photo. Face vectors are used during the run and not saved. DCT can
  switch face grouping off completely by running pipeline.py with --no-faces.

**Access control.** Staff log in with their own account. Passwords are stored as salted PBKDF2
hashes. Two roles: staff search, edit, download and add photos; admins also manage accounts,
duplicates, the bin and the System page. Every edit, download, removal and answer is in the
review log with who did it and when.

**Retention.** Removed photos go to a bin and are deleted for good after 7, 15, 30 or 60 days, as
the admin chooses. Records keep their number so the log stays readable.

**Data custody after the event.** No photos, videos, captions, transcripts or indexes are in the
GitHub repository (.gitignore blocks data/, outputs/ and all image and video formats). After the
hackathon we delete the DCT photos, videos and all generated files (outputs/, thumbnails, logs)
from our laptops, and hand any further use over to DCT.
