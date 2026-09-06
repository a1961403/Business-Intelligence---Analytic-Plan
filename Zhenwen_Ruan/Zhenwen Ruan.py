import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import pandas as pd
import re


# Configuration

TITLE_PATTERN = re.compile(r"\bbusiness analyst\b", flags=re.IGNORECASE)

# Conservative synonym mapping used for clearly equivalent surface forms.
# Add mappings only when the group agrees they are genuinely equivalent.
SYNONYM_MAP = {
    "ms excel": "excel",
    "microsoft excel": "excel",
    "communication skills": "communication",
    "problem-solving": "problem solving",
    "problemsolving": "problem solving",
}

DEFAULT_CHUNK_SIZE = 100_000


# Cleaning functions

def normalise_text(value):
    """Basic text normalisation: string conversion, strip, lowercase."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalise_skill(skill):
    """
    Normalise one skill token.
    Uses conservative, explicit synonym mapping.
    """
    s = normalise_text(skill)
    s = re.sub(r"\s+", " ", s)
    return SYNONYM_MAP.get(s, s)


def split_and_normalise_skills(skill_string):
    """
    Split comma-separated skill text into a unique normalised list.
    Blank items are removed.
    """
    if pd.isna(skill_string):
        return []

    cleaned = []
    seen = set()

    for raw in str(skill_string).split(","):
        skill = normalise_skill(raw)
        if not skill:
            continue
        if skill not in seen:
            seen.add(skill)
            cleaned.append(skill)

    return cleaned


def title_is_business_analyst(title):
    """Boundary-aware title test for the phrase 'business analyst'."""
    if pd.isna(title):
        return False
    return bool(TITLE_PATTERN.search(str(title).strip().lower()))


# Step 1: Find postings containing exact 'business analysis' skill

def collect_business_analysis_skill_rows(skills_path, chunk_size):
    """
    Read job_skills.csv in chunks.

    Returns a DataFrame containing only rows whose normalised skill list
    contains exact token 'business analysis'.

    Columns returned:
        job_link
        job_skills
        normalised_skills
    """
    keep_rows = []

    usecols = ["job_link", "job_skills"]

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            skills_path,
            usecols=usecols,
            chunksize=chunk_size,
            low_memory=False
        ),
        start=1,
    ):
        chunk["normalised_skills"] = chunk["job_skills"].apply(
            split_and_normalise_skills
        )

        mask = chunk["normalised_skills"].apply(
            lambda skills: "business analysis" in skills
        )

        selected = chunk.loc[
            mask, ["job_link", "job_skills", "normalised_skills"]
        ].copy()

        if not selected.empty:
            keep_rows.append(selected)

        print(
            f"[skills] chunk {chunk_no}: "
            f"{len(chunk):,} rows read, {mask.sum():,} matched"
        )

    if not keep_rows:
        raise RuntimeError(
            "No rows containing exact normalised skill 'business analysis' "
            "were found."
        )

    skill_matches = pd.concat(keep_rows, ignore_index=True)

    # One row per posting link.
    # If duplicate skill rows exist, keep the first for reproducibility.
    skill_matches = skill_matches.drop_duplicates(subset=["job_link"])

    print(
        f"\nDistinct postings containing exact 'business analysis': "
        f"{skill_matches['job_link'].nunique():,}\n"
    )

    return skill_matches


# Step 2: Apply title rule and construct refined cohort

def collect_matching_postings(postings_path, target_links, chunk_size):
    """
    Read postings in chunks and keep only records that:
      1. have a job_link in target_links
      2. match boundary-aware 'business analyst' title rule
    """
    available_usecols = [
        "job_link",
        "job_title",
        "company",
        "first_seen",
    ]

    keep_rows = []

    for chunk_no, chunk in enumerate(
        pd.read_csv(
            postings_path,
            usecols=lambda col: col in available_usecols,
            chunksize=chunk_size,
            low_memory=False
        ),
        start=1,
    ):
        if "job_link" not in chunk.columns or "job_title" not in chunk.columns:
            raise RuntimeError(
                "The postings file must contain job_link and job_title."
            )

        link_mask = chunk["job_link"].isin(target_links)
        if not link_mask.any():
            print(
                f"[postings] chunk {chunk_no}: {len(chunk):,} rows read, "
                "0 candidate links"
            )
            continue

        candidate = chunk.loc[link_mask].copy()

        candidate["BA_title_match"] = candidate["job_title"].apply(
            title_is_business_analyst
        )

        candidate = candidate.loc[candidate["BA_title_match"]].copy()

        if not candidate.empty:
            keep_rows.append(candidate)

        print(
            f"[postings] chunk {chunk_no}: {len(chunk):,} rows read, "
            f"{link_mask.sum():,} skill-qualified candidates, "
            f"{len(candidate):,} also matched title"
        )

    if not keep_rows:
        raise RuntimeError(
            "No postings satisfied both the title and skill rules."
        )

    postings = pd.concat(keep_rows, ignore_index=True)
    postings = postings.drop_duplicates(subset=["job_link"])

    return postings


# Step 3: Build final cohort and exploded individual_skill table

def build_refined_cohort(postings, skill_matches):
    """
    Join filtered postings to the skill rows and construct BA_role_flag.
    """
    cohort = postings.merge(
        skill_matches[["job_link", "job_skills", "normalised_skills"]],
        on="job_link",
        how="inner",
        validate="one_to_one"
    )

    cohort["BA_role_flag"] = (
        cohort["job_title"].apply(title_is_business_analyst)
        & cohort["normalised_skills"].apply(
            lambda x: "business analysis" in x
        )
    )

    cohort = cohort.loc[cohort["BA_role_flag"]].copy()

    return cohort


def make_individual_skill_table(cohort):
    """
    Explode normalised skill lists into one skill per row and enforce
    one posting-skill pair at most once.
    """
    long_df = cohort[
        ["job_link", "job_title", "company", "first_seen", "normalised_skills"]
    ].copy()

    long_df = long_df.explode("normalised_skills")
    long_df = long_df.rename(
        columns={"normalised_skills": "individual_skill"}
    )

    long_df["individual_skill"] = (
        long_df["individual_skill"].fillna("").astype(str).str.strip()
    )

    long_df = long_df.loc[long_df["individual_skill"] != ""].copy()

    # Critical posting-level deduplication rule.
    long_df = long_df.drop_duplicates(
        subset=["job_link", "individual_skill"]
    )

    return long_df


# Step 4: Descriptive outputs

def make_top_skills(long_df, cohort_size, top_n=20):
    """
    Count distinct BA postings containing each associated skill.
    Excludes defining skill 'business analysis'.
    """
    associated = long_df.loc[
        long_df["individual_skill"] != "business analysis"
    ].copy()

    counts = (
        associated.groupby("individual_skill")["job_link"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .rename("distinct_ba_postings")
        .reset_index()
    )

    counts["share_of_ba_postings_pct"] = (
        counts["distinct_ba_postings"] / cohort_size * 100
    ).round(2)

    return counts


def make_employer_skill_prevalence(long_df, top_employers=10, top_skills=10):
    """
    Produce employer x skill prevalence table for leading employers.
    Percentages use number of distinct postings for each employer
    as the denominator.
    """
    base = long_df.dropna(subset=["company"]).copy()
    base["company"] = base["company"].astype(str).str.strip()
    base = base.loc[base["company"] != ""]

    employer_counts = (
        base.groupby("company")["job_link"]
        .nunique()
        .sort_values(ascending=False)
    )

    employers = employer_counts.head(top_employers).index.tolist()

    top_skill_names = (
        base.loc[base["individual_skill"] != "business analysis"]
        .groupby("individual_skill")["job_link"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_skills)
        .index.tolist()
    )

    subset = base[
        base["company"].isin(employers)
        & base["individual_skill"].isin(top_skill_names)
    ].copy()

    numerator = (
        subset.groupby(["company", "individual_skill"])["job_link"]
        .nunique()
        .rename("postings_with_skill")
        .reset_index()
    )

    denom = (
        base[base["company"].isin(employers)]
        .groupby("company")["job_link"]
        .nunique()
        .rename("employer_postings")
        .reset_index()
    )

    result = numerator.merge(denom, on="company", how="left")
    result["prevalence_pct"] = (
        result["postings_with_skill"] / result["employer_postings"] * 100
    ).round(2)

    return result.sort_values(
        ["company", "prevalence_pct"],
        ascending=[True, False]
    )


def make_cooccurrence(long_df, min_postings=20, top_pairs=50):
    """
    Compute posting-level skill co-occurrence and Jaccard similarity.

    To keep runtime manageable, first restrict to skills that occur in
    at least min_postings distinct BA postings.
    """
    df = long_df.loc[
        long_df["individual_skill"] != "business analysis"
    ].copy()

    skill_counts = (
        df.groupby("individual_skill")["job_link"].nunique()
    )

    eligible = set(
        skill_counts[skill_counts >= min_postings].index
    )

    df = df[df["individual_skill"].isin(eligible)]

    posting_skills = (
        df.groupby("job_link")["individual_skill"]
        .apply(lambda s: sorted(set(s)))
    )

    pair_counts = defaultdict(int)

    for skills in posting_skills:
        for a, b in combinations(skills, 2):
            pair_counts[(a, b)] += 1

    rows = []
    for (a, b), both in pair_counts.items():
        a_n = int(skill_counts[a])
        b_n = int(skill_counts[b])
        union = a_n + b_n - both
        jaccard = both / union if union else 0.0

        rows.append({
            "skill_a": a,
            "skill_b": b,
            "cooccurring_postings": both,
            "skill_a_postings": a_n,
            "skill_b_postings": b_n,
            "jaccard": round(jaccard, 4),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return (
        result.sort_values(
            ["jaccard", "cooccurring_postings"],
            ascending=[False, False]
        )
        .head(top_pairs)
        .reset_index(drop=True)
    )


def make_temporal_coverage(cohort):
    """Summarise month coverage from first_seen if available."""
    if "first_seen" not in cohort.columns:
        return pd.DataFrame(
            [{"note": "first_seen column was not present in postings file"}]
        )

    dates = pd.to_datetime(cohort["first_seen"], errors="coerce")

    months = (
        dates.dt.to_period("M")
        .astype("string")
        .value_counts(dropna=False)
        .rename_axis("month")
        .reset_index(name="postings")
    )

    return months


# Main

def main():
    parser = argparse.ArgumentParser(
        description="INFO5006 BA data preparation and reproducible analysis"
    )

    parser.add_argument(
        "--postings",
        default="linkedin_job_postings.csv",
        help="Path to linkedin_job_postings.csv"
    )
    parser.add_argument(
        "--skills",
        default="job_skills.csv",
        help="Path to job_skills.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="zhenwen_info5006_outputs",
        help="Directory for generated CSV outputs"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Rows per pandas chunk"
    )

    args = parser.parse_args()

    postings_path = Path(args.postings)
    skills_path = Path(args.skills)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not postings_path.exists():
        raise FileNotFoundError(
            f"Postings file not found: {postings_path}\n"
            "Use --postings to provide the correct path."
        )

    if not skills_path.exists():
        raise FileNotFoundError(
            f"Skills file not found: {skills_path}\n"
            "Use --skills to provide the correct path."
        )

    print("=" * 72)
    print("STEP 1 – Find postings containing exact skill 'business analysis'")
    print("=" * 72)

    skill_matches = collect_business_analysis_skill_rows(
        skills_path,
        args.chunk_size
    )

    target_links = set(skill_matches["job_link"])

    print("=" * 72)
    print("STEP 2 – Apply boundary-aware Business Analyst title rule")
    print("=" * 72)

    postings = collect_matching_postings(
        postings_path,
        target_links,
        args.chunk_size
    )

    print("=" * 72)
    print("STEP 3 – Build final AND-rule cohort")
    print("=" * 72)

    cohort = build_refined_cohort(postings, skill_matches)
    cohort_size = cohort["job_link"].nunique()

    print(f"Final refined BA cohort: {cohort_size:,} postings")

    if cohort_size != 1929:
        print(
            "\nWARNING: The current group report states that the final refined "
            "cohort contains 1,929 postings.\n"
            f"This run produced {cohort_size:,}. Check that you are using the "
            "same dataset version and the same agreed cleaning rules.\n"
        )

    long_df = make_individual_skill_table(cohort)

    print("=" * 72)
    print("STEP 4 – Generate analysis outputs")
    print("=" * 72)

    top_skills = make_top_skills(long_df, cohort_size, top_n=20)
    employer_skill = make_employer_skill_prevalence(
        long_df,
        top_employers=10,
        top_skills=10
    )
    cooccurrence = make_cooccurrence(
        long_df,
        min_postings=20,
        top_pairs=50
    )
    temporal = make_temporal_coverage(cohort)

    # Save cohort without Python-list object column for clean CSV output.
    cohort_export = cohort.copy()
    cohort_export["normalised_skills"] = cohort_export[
        "normalised_skills"
    ].apply(lambda x: ", ".join(x))

    cohort_export.to_csv(
        output_dir / "refined_ba_cohort.csv",
        index=False
    )
    long_df.to_csv(
        output_dir / "ba_individual_skills.csv",
        index=False
    )
    top_skills.to_csv(
        output_dir / "top_associated_skills.csv",
        index=False
    )
    employer_skill.to_csv(
        output_dir / "employer_skill_prevalence.csv",
        index=False
    )
    cooccurrence.to_csv(
        output_dir / "skill_cooccurrence_jaccard.csv",
        index=False
    )
    temporal.to_csv(
        output_dir / "first_seen_temporal_coverage.csv",
        index=False
    )

    print("\nTop associated skills:")
    print(top_skills.head(10).to_string(index=False))

    print("\nTemporal coverage:")
    print(temporal.to_string(index=False))

    if not cooccurrence.empty:
        print("\nTop Jaccard skill pairs:")
        print(cooccurrence.head(10).to_string(index=False))

    print("\nSaved outputs to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
