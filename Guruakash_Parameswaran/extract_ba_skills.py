import pandas as pd
import os

JOBS_FILE = r"C:\Users\kesav\Downloads\archive\linkedin_job_postings.csv"
SKILLS_FILE = "job_skills.csv"
OUTPUT_FILE = "business_analyst_skills.csv"
CHUNK_SIZE = 100_000

print("Reading job postings...")

jobs = pd.read_csv(
    JOBS_FILE,
    usecols=["job_link", "job_title"],
    low_memory=False
)

jobs["job_title"] = jobs["job_title"].fillna("").astype(str)

ba_jobs = jobs[
    jobs["job_title"].str.contains(
        "business analyst",
        case=False,
        na=False
    )
].copy()

ba_links = set(ba_jobs["job_link"].dropna())

print(f"Business Analyst postings found: {len(ba_links):,}")

print("Reading job_skills file in chunks...")

found_chunks = []
total_rows = 0
matched_rows = 0

for chunk in pd.read_csv(
    SKILLS_FILE,
    usecols=["job_link", "job_skills"],
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    matched = chunk[chunk["job_link"].isin(ba_links)]

    if not matched.empty:
        found_chunks.append(matched.copy())
        matched_rows += len(matched)

    print(
        f"Processed {total_rows:,} rows | "
        f"BA matches found: {matched_rows:,}"
    )

print("Creating output file...")

if found_chunks:
    result = pd.concat(found_chunks, ignore_index=True)
    result = result.drop_duplicates(subset=["job_link"])

    result.to_csv(OUTPUT_FILE, index=False)

    print("Success!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Business Analyst jobs matched: {len(result):,}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / (1024**2):.2f} MB")
else:
    print("No matching Business Analyst jobs were found.")
