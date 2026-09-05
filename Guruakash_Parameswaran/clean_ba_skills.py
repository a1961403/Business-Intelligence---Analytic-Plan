import pandas as pd
import re

df = pd.read_csv("business_analyst_skills.csv")

skills = df["job_skills"].str.split(",").explode()
skills = skills.str.strip()
skills = skills[skills != ""]
skills = skills.str.lower()
skills = skills.str.replace(r"\s+", " ", regex=True)
skills = skills.str.title()

skill_counts = skills.value_counts().reset_index()
skill_counts.columns = ["Skill", "Number_of_Postings"]

skill_counts.to_csv(
    "business_analyst_skill_counts.csv",
    index=False
)

print("\nTop 30 Business Analyst Skills:")
print(skill_counts.head(30).to_string(index=False))

print("\nTotal unique skills:", len(skill_counts))
print("\nAnalysis complete!")
