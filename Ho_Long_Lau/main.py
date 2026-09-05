import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ============================================================
# 0. CONFIGURATION
# ============================================================

DATA_DIR = Path(r"")

POSTINGS_FILE = DATA_DIR / "linkedin_job_postings.csv"
SKILLS_FILE = DATA_DIR / "job_skills.csv"
SUMMARY_FILE = DATA_DIR / "job_summary.csv"

OUTPUT_DIR = DATA_DIR / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 80)
print("1. LOADING DATA")
print("=" * 80)

postings = pd.read_csv(POSTINGS_FILE)
skills = pd.read_csv(SKILLS_FILE)
summary = pd.read_csv(SUMMARY_FILE)

print("\nPostings shape:", postings.shape)
print("Skills shape:", skills.shape)
print("Summary shape:", summary.shape)

print("\nPostings columns:")
print(postings.columns.tolist())

print("\nSkills columns:")
print(skills.columns.tolist())

print("\nSummary columns:")
print(summary.columns.tolist())

# ============================================================
# 2. BASIC DATASET PROFILING
# ============================================================

def profile_dataset(df, name):
    print("\n" + "=" * 80)
    print(f"DATASET PROFILE: {name}")
    print("=" * 80)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print("\nMissing values:")
    missing = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    missing_table = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    })

    print(missing_table[missing_table["missing_count"] > 0])

    print("\nDuplicate rows:", df.duplicated().sum())

    if "job_link" in df.columns:
        print("Duplicate job_link:", df["job_link"].duplicated().sum())
        print("Unique job_link:", df["job_link"].nunique())

    return missing_table


postings_missing = profile_dataset(postings, "linkedin_job_postings")
skills_missing = profile_dataset(skills, "job_skills")
summary_missing = profile_dataset(summary, "job_summary")

# Save profiling information
postings_missing.to_csv(
    OUTPUT_DIR / "postings_missing.csv"
)
skills_missing.to_csv(
    OUTPUT_DIR / "skills_missing.csv"
)
summary_missing.to_csv(
    OUTPUT_DIR / "summary_missing.csv"
)

# ============================================================
# 3. CHECK JOB_LINK INTEGRITY
# ============================================================

print("\n" + "=" * 80)
print("2. JOB_LINK INTEGRITY")
print("=" * 80)

post_links = set(postings["job_link"].dropna())
skill_links = set(skills["job_link"].dropna())
summary_links = set(summary["job_link"].dropna())

print("Unique posting links:", len(post_links))
print("Unique skill links:", len(skill_links))
print("Unique summary links:", len(summary_links))

print("\nPostings with skills:",
      len(post_links & skill_links))

print("Postings with summary:",
      len(post_links & summary_links))

print("Postings with BOTH:",
      len(post_links & skill_links & summary_links))

print("Postings missing skills:",
      len(post_links - skill_links))

print("Postings missing summary:",
      len(post_links - summary_links))

# ============================================================
# 4. CHECK DUPLICATES BY JOB_LINK
# ============================================================

print("\n" + "=" * 80)
print("3. JOB_LINK DUPLICATES")
print("=" * 80)

for name, df in [
    ("Postings", postings),
    ("Skills", skills),
    ("Summary", summary)
]:
    if "job_link" in df.columns:
        duplicated_links = df[
            df["job_link"].duplicated(keep=False)
        ]

        print(f"\n{name}:")
        print("Rows involved in duplicate job_links:",
              len(duplicated_links))

        if len(duplicated_links) > 0:
            print(
                duplicated_links["job_link"]
                .value_counts()
                .head(10)
            )

# ============================================================
# 5. MERGE DATASETS
# ============================================================

print("\n" + "=" * 80)
print("4. MERGING DATASETS")
print("=" * 80)

# Select useful columns only
posting_cols = [
    "job_link",
    "job_title",
    "company",
    "job_location",
    "first_seen"
]

posting_cols = [
    col for col in posting_cols
    if col in postings.columns
]

postings_base = postings[posting_cols].copy()

# Avoid accidental many-to-many merge if duplicate job_links exist
postings_base = postings_base.drop_duplicates(
    subset=["job_link"]
)

skills_base = skills[
    [col for col in ["job_link", "job_skills"]
     if col in skills.columns]
].copy()

skills_base = skills_base.drop_duplicates(
    subset=["job_link"]
)

summary_base = summary.copy()

summary_base = summary_base.drop_duplicates(
    subset=["job_link"]
)

# Merge
merged = postings_base.merge(
    skills_base,
    on="job_link",
    how="left"
)

merged = merged.merge(
    summary_base,
    on="job_link",
    how="left",
    suffixes=("", "_summary")
)

print("Merged shape:", merged.shape)

print("\nMerged missing values:")
print(
    merged.isna().mean()
    .mul(100)
    .round(2)
    .sort_values(ascending=False)
)

merged.to_csv(
    OUTPUT_DIR / "merged_dataset.csv",
    index=False
)

# ============================================================
# 6. JOB TITLE INSPECTION
# ============================================================

print("\n" + "=" * 80)
print("5. JOB TITLE EXPLORATION")
print("=" * 80)

merged["job_title_clean"] = (
    merged["job_title"]
    .fillna("")
    .astype(str)
    .str.strip()
)

merged["job_title_lower"] = (
    merged["job_title_clean"]
    .str.lower()
)

print("\nTotal unique job titles:",
      merged["job_title_clean"].nunique())

print("\nExample job titles:")
print(
    merged["job_title_clean"]
    .value_counts()
    .head(30)
)

# ============================================================
# 7. BASELINE BUSINESS ANALYST FILTER
# ============================================================

print("\n" + "=" * 80)
print("6. BUSINESS ANALYST FILTER")
print("=" * 80)

ba_mask = merged["job_title_lower"].str.contains(
    r"\bbusiness analyst\b",
    regex=True,
    na=False
)

ba_jobs = merged[ba_mask].copy()

print("Total postings:", len(merged))
print("Business Analyst postings:", len(ba_jobs))

if len(merged) > 0:
    print(
        "BA percentage:",
        round(len(ba_jobs) / len(merged) * 100, 2),
        "%"
    )

print("\nBusiness Analyst titles:")
print(
    ba_jobs["job_title_clean"]
    .value_counts()
    .head(50)
)

ba_jobs.to_csv(
    OUTPUT_DIR / "business_analyst_jobs.csv",
    index=False
)

# ============================================================
# 8. EXPLORE RELATED ANALYST TITLES
# ============================================================

print("\n" + "=" * 80)
print("7. RELATED ANALYST TITLE EXPLORATION")
print("=" * 80)

analyst_keywords = [
    "business analyst",
    "business systems analyst",
    "business intelligence analyst",
    "data analyst",
    "financial analyst",
    "systems analyst",
    "marketing analyst",
    "process analyst",
    "operations analyst",
    "technical analyst",
    "product analyst"
]

title_results = []

for keyword in analyst_keywords:

    mask = merged["job_title_lower"].str.contains(
        re.escape(keyword),
        na=False
    )

    count = mask.sum()

    title_results.append({
        "title_pattern": keyword,
        "job_count": count
    })

related_titles = pd.DataFrame(title_results)

related_titles["percentage"] = (
    related_titles["job_count"] /
    len(merged) * 100
).round(2)

print(related_titles)

related_titles.to_csv(
    OUTPUT_DIR / "related_analyst_titles.csv",
    index=False
)

# ============================================================
# 9. POTENTIAL BA SUBTYPES
# ============================================================

print("\n" + "=" * 80)
print("8. BUSINESS ANALYST SUBTYPES")
print("=" * 80)

ba_subtype_patterns = {
    "Business Analyst": r"\bbusiness analyst\b",
    "Business Systems Analyst": r"\bbusiness systems analyst\b",
    "Business Intelligence Analyst": r"\bbusiness intelligence analyst\b",
    "Senior Business Analyst": r"\bsenior business analyst\b",
    "Junior Business Analyst": r"\bjunior business analyst\b",
    "Lead Business Analyst": r"\blead business analyst\b",
    "Technical Business Analyst": r"\btechnical business analyst\b",
    "IT Business Analyst": r"\bit business analyst\b",
    "Data Business Analyst": r"\bdata business analyst\b",
    "Business Process Analyst": r"\bbusiness process analyst\b"
}

subtype_results = []

for subtype, pattern in ba_subtype_patterns.items():

    mask = merged["job_title_lower"].str.contains(
        pattern,
        regex=True,
        na=False
    )

    subtype_results.append({
        "subtype": subtype,
        "job_count": mask.sum()
    })

subtypes = pd.DataFrame(subtype_results)

subtypes["percentage_of_all_jobs"] = (
    subtypes["job_count"] /
    len(merged) * 100
).round(2)

subtypes["percentage_of_ba_jobs"] = (
    subtypes["job_count"] /
    len(ba_jobs) * 100
).round(2)

print(subtypes)

subtypes.to_csv(
    OUTPUT_DIR / "ba_subtypes.csv",
    index=False
)

# ============================================================
# 10. LOCATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("9. BA LOCATION ANALYSIS")
print("=" * 80)

if "job_location" in ba_jobs.columns:

    location_counts = (
        ba_jobs["job_location"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    print("\nTop 20 BA locations:")
    print(location_counts.head(20))

    location_df = (
        location_counts
        .reset_index()
    )

    location_df.columns = [
        "location",
        "job_count"
    ]

    location_df["percentage"] = (
        location_df["job_count"] /
        len(ba_jobs) * 100
    ).round(2)

    location_df.to_csv(
        OUTPUT_DIR / "ba_locations.csv",
        index=False
    )

# ============================================================
# 11. SKILL CLEANING
# ============================================================

print("\n" + "=" * 80)
print("10. SKILL PROCESSING")
print("=" * 80)

if "job_skills" not in ba_jobs.columns:

    print("job_skills column not found.")

else:

    ba_jobs["job_skills_clean"] = (
        ba_jobs["job_skills"]
        .fillna("")
        .astype(str)
    )

    # Split comma-separated skills
    skill_rows = []

    for _, row in ba_jobs.iterrows():

        job_link = row["job_link"]
        skill_string = row["job_skills_clean"]

        if not skill_string.strip():
            continue

        individual_skills = skill_string.split(",")

        for skill in individual_skills:

            skill = skill.strip().lower()

            if skill:
                skill_rows.append({
                    "job_link": job_link,
                    "skill": skill
                })

    skills_long = pd.DataFrame(skill_rows)

    print("Skill records:", len(skills_long))
    print("Unique raw skills:",
          skills_long["skill"].nunique())

    # ========================================================
    # 12. NORMALIZE COMMON SKILL VARIATIONS
    # ========================================================

    def normalize_skill(skill):

        skill = skill.lower().strip()

        skill = re.sub(r"\s+", " ", skill)

        replacements = {

            "microsoft excel": "excel",
            "ms excel": "excel",

            "microsoft sql server": "sql",
            "sql server": "sql",

            "power bi": "power bi",
            "microsoft power bi": "power bi",

            "tableau software": "tableau",

            "python programming": "python",
            "python (programming language)": "python",

            "business analysis": "business analysis",
            "business analyst": "business analysis",

            "requirements analysis": "requirements analysis",

            "requirements gathering": "requirements analysis",

            "stakeholder management": "stakeholder management",

            "process improvement": "process improvement"
        }

        return replacements.get(skill, skill)

    skills_long["skill_normalized"] = (
        skills_long["skill"]
        .apply(normalize_skill)
    )

    # Remove empty values
    skills_long = skills_long[
        skills_long["skill_normalized"].str.len() > 0
    ]

    print(
        "Unique normalized skills:",
        skills_long["skill_normalized"].nunique()
    )

    skills_long.to_csv(
        OUTPUT_DIR / "ba_skills_long.csv",
        index=False
    )

# ============================================================
# 13. SKILL FREQUENCY BY JOB
# ============================================================

print("\n" + "=" * 80)
print("11. TOP SKILLS")
print("=" * 80)

if len(skills_long) > 0:

    # Count jobs rather than raw occurrences
    skill_job_counts = (
        skills_long[
            ["job_link", "skill_normalized"]
        ]
        .drop_duplicates()
        .groupby("skill_normalized")
        .size()
        .sort_values(ascending=False)
    )

    skill_df = (
        skill_job_counts
        .reset_index()
    )

    skill_df.columns = [
        "skill",
        "job_count"
    ]

    skill_df["percentage_of_ba_jobs"] = (
        skill_df["job_count"] /
        len(ba_jobs) * 100
    ).round(2)

    print(
        "\nTop 50 skills by number of BA job postings:"
    )

    print(
        skill_df.head(50).to_string(index=False)
    )

    skill_df.to_csv(
        OUTPUT_DIR / "ba_skill_frequency.csv",
        index=False
    )

# ============================================================
# 14. SKILL CATEGORY ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("12. SKILL CATEGORY ANALYSIS")
print("=" * 80)

skill_categories = {

    "Technical / Data": [
        "sql",
        "python",
        "r",
        "data analysis",
        "data analytics",
        "statistics",
        "machine learning",
        "database",
        "database management",
        "data visualization"
    ],

    "Business Analysis": [
        "business analysis",
        "requirements analysis",
        "requirements gathering",
        "business process",
        "process improvement",
        "business requirements",
        "process mapping",
        "business intelligence"
    ],

    "Stakeholder / Soft Skills": [
        "stakeholder management",
        "communication",
        "leadership",
        "problem solving",
        "project management",
        "collaboration",
        "presentation"
    ],

    "Tools / Platforms": [
        "excel",
        "power bi",
        "tableau",
        "jira",
        "confluence",
        "sap",
        "salesforce",
        "microsoft office"
    ]
}

def assign_category(skill):

    for category, skill_list in skill_categories.items():

        if skill in skill_list:
            return category

    return "Other"

if len(skills_long) > 0:

    skills_long["category"] = (
        skills_long["skill_normalized"]
        .apply(assign_category)
    )

    category_job_counts = (
        skills_long[
            ["job_link", "category"]
        ]
        .drop_duplicates()
        .groupby("category")
        .size()
        .sort_values(ascending=False)
    )

    category_df = (
        category_job_counts
        .reset_index()
    )

    category_df.columns = [
        "category",
        "job_count"
    ]

    category_df["percentage_of_ba_jobs"] = (
        category_df["job_count"] /
        len(ba_jobs) * 100
    ).round(2)

    print(
        category_df.to_string(index=False)
    )

    category_df.to_csv(
        OUTPUT_DIR / "ba_skill_categories.csv",
        index=False
    )

# ============================================================
# 15. SKILLS PER JOB
# ============================================================

print("\n" + "=" * 80)
print("13. NUMBER OF SKILLS PER BA JOB")
print("=" * 80)

if len(skills_long) > 0:

    skills_per_job = (
        skills_long[
            ["job_link", "skill_normalized"]
        ]
        .drop_duplicates()
        .groupby("job_link")
        .size()
    )

    print(
        "Mean skills per job:",
        round(skills_per_job.mean(), 2)
    )

    print(
        "Median skills per job:",
        skills_per_job.median()
    )

    print(
        "Minimum skills:",
        skills_per_job.min()
    )

    print(
        "Maximum skills:",
        skills_per_job.max()
    )

    skills_per_job.to_csv(
        OUTPUT_DIR / "skills_per_job.csv"
    )

# ============================================================
# 16. VISUALISATION 1
# TOP 10 SKILLS
# ============================================================

if len(skill_df) > 0:

    top10 = skill_df.head(10).sort_values(
        "job_count"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top10["skill"],
        top10["job_count"]
    )

    plt.xlabel(
        "Number of Business Analyst Job Postings"
    )

    plt.ylabel("Skill")

    plt.title(
        "Top 10 Skills Required in Business Analyst Job Postings"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top10_ba_skills.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# 17. VISUALISATION 2
# SKILL CATEGORY
# ============================================================

if "category_df" in locals():

    category_plot = category_df.sort_values(
        "job_count"
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        category_plot["category"],
        category_plot["job_count"]
    )

    plt.xlabel(
        "Number of Business Analyst Job Postings"
    )

    plt.ylabel("Skill Category")

    plt.title(
        "Skill Categories in Business Analyst Job Postings"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "skill_categories.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# 18. VISUALISATION 3
# BA LOCATION
# ============================================================

if "location_df" in locals():

    top_locations = (
        location_df
        .head(10)
        .sort_values("job_count")
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_locations["location"],
        top_locations["job_count"]
    )

    plt.xlabel(
        "Number of Business Analyst Job Postings"
    )

    plt.ylabel("Location")

    plt.title(
        "Top 10 Locations for Business Analyst Jobs"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top10_ba_locations.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

# ============================================================
# 19. VISUALISATION 4 (ADDED FOR HD)
# BIVARIATE ANALYSIS: LOCATION VS SKILL CATEGORY
# ============================================================

print("\n" + "=" * 80)
print("14. BIVARIATE ANALYSIS: LOCATION VS CATEGORY (FIGURE 4)")
print("=" * 80)

if "location_df" in locals() and "skills_long" in locals() and "ba_jobs" in locals():

    top_5_locs = location_df.head(5)["location"].tolist()

    skills_with_loc = skills_long.merge(
        ba_jobs[["job_link", "job_location"]].dropna(subset=["job_location"]),
        on="job_link",
        how="inner"
    )

    skills_with_loc["job_location"] = skills_with_loc["job_location"].astype(str).str.strip()

    bivariate_df = skills_with_loc[skills_with_loc["job_location"].isin(top_5_locs)]

    cross_tab = pd.crosstab(
        bivariate_df["job_location"], 
        bivariate_df["category"]
    )

    cross_tab = cross_tab.reindex(top_5_locs)
    
    print("\nBivariate Cross-Tabulation (Location vs Category):")
    print(cross_tab)

    cross_tab_core = cross_tab.drop(columns=["Other"])
    cross_tab_pct = cross_tab_core.div(cross_tab_core.sum(axis=1), axis=0) * 100

    plt.figure(figsize=(10, 6))
    
    cross_tab_pct.plot(
        kind="bar", 
        stacked=True, 
        colormap="tab10", 
        figsize=(12, 7)
    )

    plt.xlabel("Top 5 Job Locations")
    plt.ylabel("Percentage of Required Skill Categories (%)")
    plt.title("Figure 4: Bivariate Analysis of Skill Categories across Top Hubs")
    
    plt.legend(title="Skill Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "figure4_bivariate_location_vs_category.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    
    cross_tab_pct.to_csv(
        OUTPUT_DIR / "bivariate_location_vs_category_pct.csv"
    )

# ============================================================
# 20. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL EDA SUMMARY")
print("=" * 80)

print("Total LinkedIn postings:", len(merged))
print("BA postings:", len(ba_jobs))

if len(merged) > 0:
    print(
        "BA percentage:",
        round(len(ba_jobs) / len(merged) * 100, 2),
        "%"
    )

if len(skills_long) > 0:

    print(
        "Unique normalized skills:",
        skills_long["skill_normalized"].nunique()
    )

    print(
        "\nTop 10 skills:"
    )

    print(
        skill_df.head(10).to_string(index=False)
    )

print("\nOutput files saved to:")
print(OUTPUT_DIR)
