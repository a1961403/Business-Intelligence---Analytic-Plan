import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

JOBS_FILE = "linkedin_job_postings.csv"
SKILLS_FILE = "job_skills.csv"

OUTPUT_DIR = Path("guruakash_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the required job-posting fields.
jobs = pd.read_csv(
    JOBS_FILE,
    usecols=["job_link", "job_title", "job_location"],
    low_memory=False
)

print("Original job postings:", len(jobs))

# Normalise job titles for consistent filtering.
jobs["job_title_clean"] = (
    jobs["job_title"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

# Identify Business Analyst title candidates.
title_candidates = jobs[
    jobs["job_title_clean"].str.contains(
        "business analyst",
        case=False,
        na=False
    )
].copy()

ba_links = set(title_candidates["job_link"].dropna())

print(
    "Business Analyst title candidates:",
    len(title_candidates)
)

# Read the large skills file in chunks.
matched_chunks = []

for chunk in pd.read_csv(
    SKILLS_FILE,
    usecols=["job_link", "job_skills"],
    chunksize=100000,
    low_memory=False
):
    matched = chunk[
        chunk["job_link"].isin(ba_links)
    ].copy()

    if matched.empty:
        continue

    matched["job_skills_clean"] = (
        matched["job_skills"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Keep postings containing the exact "business analysis" skill.
    matched = matched[
        matched["job_skills_clean"]
        .str.split(",")
        .apply(
            lambda skills:
            "business analysis" in
            [skill.strip() for skill in skills]
        )
    ]

    if not matched.empty:
        matched_chunks.append(matched)

if not matched_chunks:
    raise ValueError(
        "No records matched the confirmed BA AND rule."
    )

# Create the final Business Analyst dataset.
ba_skills = pd.concat(
    matched_chunks,
    ignore_index=True
)

ba_skills = ba_skills.drop_duplicates(
    subset=["job_link"]
)

# Add job title and location information.
ba_data = ba_skills.merge(
    jobs[["job_link", "job_title", "job_location"]],
    on="job_link",
    how="left",
    validate="one_to_one"
)

print(
    "Final Business Analyst postings:",
    len(ba_data)
)

ba_data.to_csv(
    OUTPUT_DIR / "guruakash_ba_dataset.csv",
    index=False
)

# Produce the data exploration summary.
summary = pd.DataFrame({
    "Metric": [
        "Original job postings",
        "BA title candidates",
        "Final BA postings",
        "Missing job_link",
        "Missing job_title",
        "Missing job_skills",
        "Missing job_location",
        "Unique job titles",
        "Unique job locations"
    ],
    "Result": [
        len(jobs),
        len(title_candidates),
        len(ba_data),
        ba_data["job_link"].isna().sum(),
        ba_data["job_title"].isna().sum(),
        ba_data["job_skills"].isna().sum(),
        ba_data["job_location"].isna().sum(),
        ba_data["job_title"].nunique(),
        ba_data["job_location"].nunique()
    ]
})

print("\nDATA EXPLORATION SUMMARY")
print(summary.to_string(index=False))

summary.to_csv(
    OUTPUT_DIR / "data_exploration_summary.csv",
    index=False
)

# Check completeness and uniqueness of key variables.
quality = pd.DataFrame({
    "Variable": [
        "job_link",
        "job_title",
        "job_skills",
        "job_location"
    ],
    "Missing_Count": [
        ba_data["job_link"].isna().sum(),
        ba_data["job_title"].isna().sum(),
        ba_data["job_skills"].isna().sum(),
        ba_data["job_location"].isna().sum()
    ],
    "Unique_Count": [
        ba_data["job_link"].nunique(),
        ba_data["job_title"].nunique(),
        ba_data["job_skills"].nunique(),
        ba_data["job_location"].nunique()
    ]
})

quality["Missing_Percent"] = (
    quality["Missing_Count"] / len(ba_data) * 100
)

print("\nDATA QUALITY")
print(quality.to_string(index=False))

quality.to_csv(
    OUTPUT_DIR / "guruakash_quality_checks.csv",
    index=False
)

# Split the skills field into individual skills.
skills = ba_data[["job_link", "job_skills"]].copy()

skills["individual_skill"] = (
    skills["job_skills"]
    .fillna("")
    .str.split(",")
)

skills = skills.explode("individual_skill")

skills["individual_skill"] = (
    skills["individual_skill"]
    .astype(str)
    .str.strip()
    .str.lower()
)

skills = skills[
    skills["individual_skill"] != ""
]

# Standardise selected equivalent skill names.
skill_mapping = {
    "ms excel": "excel",
    "microsoft excel": "excel",
    "problemsolving": "problem solving",
    "problem-solving": "problem solving",
    "communication skills": "communication"
}

skills["individual_skill"] = (
    skills["individual_skill"].replace(skill_mapping)
)

# Count each skill at most once per job posting.
skills = skills.drop_duplicates(
    subset=["job_link", "individual_skill"]
)

skill_counts = (
    skills
    .groupby("individual_skill")
    .size()
    .reset_index(name="job_postings")
    .sort_values("job_postings", ascending=False)
)

print("\nTOP 20 SKILLS")
print(
    skill_counts.head(20)
    .to_string(index=False)
)

skill_counts.to_csv(
    OUTPUT_DIR / "guruakash_skill_frequency.csv",
    index=False
)

# Exclude "business analysis" because it is part of the BA selection rule.
plot_data = skill_counts[
    skill_counts["individual_skill"] != "business analysis"
].head(15).copy()

plot_data["percentage"] = (
    plot_data["job_postings"] / len(ba_data) * 100
)

plot_data = plot_data.sort_values("job_postings")

plt.figure(figsize=(10, 7))

plt.barh(
    plot_data["individual_skill"],
    plot_data["job_postings"]
)

plt.xlabel("Number of Business Analyst job postings")
plt.ylabel("Associated skill")
plt.title("Top 15 Skills Associated with Business Analyst Job Postings")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_1_top_15_associated_skills.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n================================")
print("ANALYSIS COMPLETE")
print("================================")
print("Final BA postings:", len(ba_data))
