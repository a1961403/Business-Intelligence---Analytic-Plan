"""
Guruakash Parameswaran
Assignment 1 - Advanced Data Exploration / Section 4 Upgrade

Purpose:
Extend the initial Business Analyst skill-frequency analysis in response to
tutor feedback. The analysis adds:
1. Monthly BA posting volume and month-on-month change.
2. Monthly prevalence of the five leading associated skills.
3. Top-10 employer comparison and their skill profiles.
4. Skill co-occurrence among the leading skills.
5. Geographic concentration as a supplementary diagnostic.

Inputs:
- linkedin_job_postings.csv
- job_skills.csv

The script preserves the group's confirmed BA operational rule:
normalised job_title contains "business analyst" AND job_skills contains
the exact skill "business analysis".

Run from the folder containing the two CSV files:
    python guruakash_advanced_exploration.py
"""

from pathlib import Path
from itertools import combinations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "guruakash_advanced_outputs"
OUT.mkdir(exist_ok=True)

POSTINGS = BASE / "linkedin_job_postings.csv"
SKILLS = BASE / "job_skills.csv"

# -----------------------------
# 1. Load and link source data
# -----------------------------
postings = pd.read_csv(POSTINGS, low_memory=False)
skills = pd.read_csv(SKILLS, low_memory=False)

required_posting = ["job_link", "job_title", "job_location", "first_seen", "company"]
missing_posting = [c for c in required_posting if c not in postings.columns]
if missing_posting:
    raise ValueError(
        f"Missing required columns in linkedin_job_postings.csv: {missing_posting}"
    )

required_skill = ["job_link", "job_skills"]
missing_skill = [c for c in required_skill if c not in skills.columns]
if missing_skill:
    raise ValueError(
        f"Missing required columns in job_skills.csv: {missing_skill}"
    )

postings["job_link"] = postings["job_link"].astype("string").str.strip()
skills["job_link"] = skills["job_link"].astype("string").str.strip()
skills["job_skills"] = skills["job_skills"].astype("string").fillna("")

# Keep one skill-list record per posting key.
skills = skills.drop_duplicates(subset=["job_link"])

merged = postings[
    ["job_link", "job_title", "job_location", "first_seen", "company"]
].merge(
    skills[["job_link", "job_skills"]],
    on="job_link",
    how="inner",
    validate="one_to_one"
)

# ----------------------------------
# 2. Confirmed Business Analyst rule
# ----------------------------------
title_mask = merged["job_title"].astype("string").str.lower().str.contains(
    r"\bbusiness\s+analyst\b", regex=True, na=False
)

skill_mask = merged["job_skills"].astype("string").str.lower().str.contains(
    r"(^|,\s*)business analysis(\s*,|$)",
    regex=True,
    na=False
)

ba = merged.loc[title_mask & skill_mask].copy()
ba["first_seen"] = pd.to_datetime(ba["first_seen"], errors="coerce")
ba["month"] = ba["first_seen"].dt.to_period("M").astype("string")

# -----------------------------
# 3. Normalise and explode skills
# -----------------------------
skill_long = (
    ba[["job_link", "job_skills"]]
    .assign(individual_skill=ba["job_skills"].str.split(","))
    .explode("individual_skill")
)

skill_long["individual_skill"] = (
    skill_long["individual_skill"]
    .astype("string")
    .str.strip()
    .str.lower()
)

skill_long = skill_long[
    skill_long["individual_skill"].notna()
    & (skill_long["individual_skill"] != "")
].copy()

skill_mapping = {
    "ms excel": "excel",
    "microsoft excel": "excel",
    "communication skills": "communication",
    "problem-solving": "problem solving",
    "problemsolving": "problem solving",
}

skill_long["individual_skill"] = skill_long["individual_skill"].replace(skill_mapping)

# One occurrence of a skill per job.
skill_long = skill_long.drop_duplicates(
    subset=["job_link", "individual_skill"]
)

N_BA = len(ba)

skill_counts = (
    skill_long.groupby("individual_skill")["job_link"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="job_postings")
)
skill_counts["percentage"] = skill_counts["job_postings"] / N_BA * 100

# Exclude the defining skill from substantive skill rankings.
associated = skill_counts[
    skill_counts["individual_skill"] != "business analysis"
].copy()

associated.to_csv(
    OUT / "advanced_all_associated_skill_counts.csv", index=False
)

# ------------------------------------
# 4. Monthly BA posting trend
# ------------------------------------
monthly = (
    ba.dropna(subset=["first_seen"])
    .groupby("month")["job_link"]
    .nunique()
    .reset_index(name="ba_postings")
)

monthly["mom_change"] = monthly["ba_postings"].pct_change() * 100
monthly["mom_change_pp"] = monthly["ba_postings"].diff()

monthly.to_csv(
    OUT / "advanced_monthly_ba_posting_trend.csv", index=False
)

# ---------------------------------------------
# 5. Monthly prevalence of top five skills
# ---------------------------------------------
top5 = associated.head(5)["individual_skill"].tolist()

monthly_skill = (
    skill_long[
        (skill_long["individual_skill"].isin(top5))
        & skill_long["job_link"].isin(ba["job_link"])
    ]
    .merge(ba[["job_link", "month"]], on="job_link", how="left")
    .groupby(["month", "individual_skill"])["job_link"]
    .nunique()
    .reset_index(name="postings_with_skill")
)

monthly_skill = monthly_skill.merge(
    monthly[["month", "ba_postings"]],
    on="month",
    how="left"
)
monthly_skill["skill_share_pct"] = (
    monthly_skill["postings_with_skill"]
    / monthly_skill["ba_postings"]
    * 100
)

monthly_skill.to_csv(
    OUT / "advanced_monthly_top5_skill_share.csv", index=False
)

# ---------------------------------------------
# 6. Top 10 companies and skill requirements
# ---------------------------------------------
ba_company = ba.copy()
ba_company["company_clean"] = (
    ba_company["company"]
    .astype("string")
    .str.strip()
)
ba_company.loc[
    ba_company["company_clean"].isin(["", "nan", "none", "<na>"]),
    "company_clean"
] = pd.NA

company_counts = (
    ba_company.dropna(subset=["company_clean"])
    .groupby("company_clean")["job_link"]
    .nunique()
    .sort_values(ascending=False)
    .reset_index(name="ba_postings")
)

top10_companies = company_counts.head(10).copy()
top10_companies["share_of_ba_pct"] = (
    top10_companies["ba_postings"] / N_BA * 100
)

top10_companies.to_csv(
    OUT / "advanced_top10_companies.csv", index=False
)

company_skill = (
    skill_long
    .merge(
        ba_company[["job_link", "company_clean"]],
        on="job_link",
        how="left"
    )
)

company_skill = company_skill[
    company_skill["company_clean"].isin(
        top10_companies["company_clean"]
    )
    & (company_skill["individual_skill"] != "business analysis")
]

top_company_skills = associated.head(15)["individual_skill"].tolist()

company_skill = company_skill[
    company_skill["individual_skill"].isin(top_company_skills)
]

company_skill_counts = (
    company_skill.groupby(
        ["company_clean", "individual_skill"]
    )["job_link"]
    .nunique()
    .reset_index(name="postings_with_skill")
)

company_skill_matrix = company_skill_counts.merge(
    top10_companies[["company_clean", "ba_postings"]],
    on="company_clean",
    how="left"
)

company_skill_matrix["skill_share_pct"] = (
    company_skill_matrix["postings_with_skill"]
    / company_skill_matrix["ba_postings"]
    * 100
)

company_skill_matrix.to_csv(
    OUT / "advanced_top10_company_skill_profile.csv", index=False
)

# ---------------------------------------------
# 7. Skill co-occurrence among top ten skills
# ---------------------------------------------
top10_skills = associated.head(10)["individual_skill"].tolist()

job_skill_sets = (
    skill_long[skill_long["individual_skill"].isin(top10_skills)]
    .groupby("job_link")["individual_skill"]
    .apply(set)
)

co_rows = []

for a, b in combinations(top10_skills, 2):
    jobs_a = {job for job, skills_set in job_skill_sets.items() if a in skills_set}
    jobs_b = {job for job, skills_set in job_skill_sets.items() if b in skills_set}
    intersection = len(jobs_a & jobs_b)
    union = len(jobs_a | jobs_b)
    jaccard = intersection / union if union else np.nan

    co_rows.append({
        "skill_a": a,
        "skill_b": b,
        "cooccurring_postings": intersection,
        "jaccard_similarity": round(jaccard, 4)
    })

cooccurrence = pd.DataFrame(co_rows).sort_values(
    ["jaccard_similarity", "cooccurring_postings"],
    ascending=False
)

cooccurrence.to_csv(
    OUT / "advanced_skill_cooccurrence.csv", index=False
)

# ---------------------------------------------
# 8. Top geographic locations
# ---------------------------------------------
location_counts = (
    ba["job_location"]
    .astype("string")
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .value_counts()
    .reset_index()
)
location_counts.columns = ["job_location", "ba_postings"]
location_counts["share_of_ba_pct"] = (
    location_counts["ba_postings"] / N_BA * 100
)

location_counts.head(20).to_csv(
    OUT / "advanced_top20_locations.csv", index=False
)

# -----------------------------
# 9. Figures
# -----------------------------

# Figure 4: Monthly BA posting volume
plt.figure(figsize=(10.5, 6.5))
plt.plot(monthly["month"], monthly["ba_postings"], marker="o")
plt.xlabel("Month")
plt.ylabel("Distinct BA postings")
plt.title("Monthly Business Analyst Posting Volume")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    OUT / "Figure_4_Monthly_BA_Posting_Volume.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

# Figure 5: Monthly prevalence of top five skills
plt.figure(figsize=(11, 6.5))
for skill in top5:
    subset = monthly_skill[
        monthly_skill["individual_skill"] == skill
    ]
    plt.plot(
        subset["month"],
        subset["skill_share_pct"],
        marker="o",
        label=skill
    )

plt.xlabel("Month")
plt.ylabel("Share of BA postings (%)")
plt.title("Monthly Prevalence of the Five Most Common Associated Skills")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(
    OUT / "Figure_5_Monthly_Top5_Skill_Share.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

# Figure 6: Top ten companies
plot_companies = top10_companies.sort_values("ba_postings")
plt.figure(figsize=(10.5, 6.5))
plt.barh(plot_companies["company_clean"], plot_companies["ba_postings"])
plt.xlabel("Distinct BA postings")
plt.ylabel("Company")
plt.title("Top 10 Employers by Business Analyst Posting Count")
plt.tight_layout()
plt.savefig(
    OUT / "Figure_6_Top10_Companies.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

# Figure 7: Top company skill profile
heat = company_skill_matrix.pivot_table(
    index="company_clean",
    columns="individual_skill",
    values="skill_share_pct",
    fill_value=0
)

heat = heat.reindex(
    index=top10_companies["company_clean"],
    columns=top_company_skills
)

plt.figure(figsize=(13, 7))
plt.imshow(heat.values, aspect="auto")
plt.xticks(
    range(len(heat.columns)),
    heat.columns,
    rotation=60,
    ha="right"
)
plt.yticks(
    range(len(heat.index)),
    heat.index
)
plt.xlabel("Associated skill")
plt.ylabel("Top employer")
plt.title("Skill Prevalence Across the Top 10 BA Employers (%)")
plt.colorbar(label="Share of company's BA postings (%)")
plt.tight_layout()
plt.savefig(
    OUT / "Figure_7_Top10_Company_Skill_Heatmap.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

# Figure 8: Skill co-occurrence
top_pairs = cooccurrence.head(10).copy()
pair_labels = (
    top_pairs["skill_a"].str.title()
    + " + "
    + top_pairs["skill_b"].str.title()
)
plt.figure(figsize=(10.5, 6.5))
plt.barh(
    pair_labels.iloc[::-1],
    top_pairs["jaccard_similarity"].iloc[::-1]
)
plt.xlabel("Jaccard similarity")
plt.ylabel("Skill pair")
plt.title("Strongest Skill Pair Co-occurrence Among the Top 10 Skills")
plt.tight_layout()
plt.savefig(
    OUT / "Figure_8_Skill_Cooccurrence.png",
    dpi=300, bbox_inches="tight"
)
plt.close()

# ---------------------------------------------
# 10. Automated written findings
# ---------------------------------------------
lines = []
lines.append("GURUAKASH ADVANCED SECTION 4 RESULTS")
lines.append("=" * 45)
lines.append(f"Final BA postings: {N_BA:,}")
lines.append("")

if len(monthly):
    peak = monthly.loc[monthly["ba_postings"].idxmax()]
    low = monthly.loc[monthly["ba_postings"].idxmin()]
    lines.append(
        f"Monthly volume: highest month = {peak['month']} "
        f"({int(peak['ba_postings']):,} postings); "
        f"lowest month = {low['month']} "
        f"({int(low['ba_postings']):,} postings)."
    )

    valid_mom = monthly.dropna(subset=["mom_change"])
    if len(valid_mom):
        max_growth = valid_mom.loc[valid_mom["mom_change"].idxmax()]
        max_decline = valid_mom.loc[valid_mom["mom_change"].idxmin()]
        lines.append(
            f"Largest month-on-month increase: {max_growth['month']} "
            f"({max_growth['mom_change']:.1f}%)."
        )
        lines.append(
            f"Largest month-on-month decrease: {max_decline['month']} "
            f"({max_decline['mom_change']:.1f}%)."
        )

lines.append("")
lines.append("Top five associated skills:")
for _, row in associated.head(5).iterrows():
    lines.append(
        f"- {row['individual_skill']}: "
        f"{int(row['job_postings']):,} postings "
        f"({row['percentage']:.2f}%)."
    )

lines.append("")
if len(top10_companies):
    lines.append("Top ten employers by BA posting count:")
    for _, row in top10_companies.iterrows():
        lines.append(
            f"- {row['company_clean']}: "
            f"{int(row['ba_postings']):,} postings "
            f"({row['share_of_ba_pct']:.2f}% of BA sample)."
        )

lines.append("")
if len(cooccurrence):
    lines.append("Strongest skill-pair relationships:")
    for _, row in cooccurrence.head(5).iterrows():
        lines.append(
            f"- {row['skill_a']} + {row['skill_b']}: "
            f"{int(row['cooccurring_postings']):,} co-occurring postings; "
            f"Jaccard = {row['jaccard_similarity']:.3f}."
        )

lines.append("")
lines.append("Interpretation cautions:")
lines.append(
    "- Monthly changes describe posting volume in the 2024 LinkedIn dataset; "
    "they do not establish labour-market growth because scraping coverage can vary."
)
lines.append(
    "- Company comparisons describe advertised BA demand among the observed "
    "postings; they are not measures of total hiring or employer quality."
)
lines.append(
    "- Skill percentages measure co-presence in advertisements, not skill importance "
    "or causal influence on hiring."
)
lines.append(
    "- The company and first_seen analyses are supplementary exploratory variables. "
    "If included in the group report, update the integrated data dictionary accordingly."
)

(OUT / "advanced_analysis_summary.txt").write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print("=" * 60)
print("ADVANCED ANALYSIS COMPLETE")
print("=" * 60)
print(f"Final BA postings: {N_BA:,}")
print(f"Output folder: {OUT}")
print("Files created:")
for p in sorted(OUT.iterdir()):
    print(" -", p.name)
