import os
import time
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
load_dotenv()
import pandas as pd

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/final_code/statistics_output/cases/CASE_2_ONLY_NEI.parquet")

OUTPUT_PATH = Path(
    "/export/usuarios01/ivgomez/mind/final_mistral_results/final_code/statistics_output/case_2_gemini_results.parquet"
)

CHECKPOINT_EVERY = 20

# Leave some margin below the 500 daily-call limit
MAX_CALLS_PER_RUN = 1

MODEL = "gemini-3.1-pro-preview"


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class SearchResult(BaseModel):

    passage_web: str = Field(
        description=(
            "Short self-contained passage summarizing the "
            "best-supported information found on the web."
        )
    )

    discrepancy_label: Literal[
        "NO_DISCREPANCY",
        "CONTRADICTION",
        "CULTURAL_DISCREPANCY",
        "NOT_ENOUGH_INFO",
    ]

    sources: list[str]


# ============================================================
# PROMPT
# ============================================================

PROMPT = """
<role>
You are verifying and enriching a multilingual knowledge base. You check
whether an existing passage is still accurate by researching the web,
and you produce an updated, well-sourced passage.
</role>

<inputs>
<question>{question}</question>
<source_passage>{source_chunk}</source_passage>
</inputs>

<task>
Determine whether reliable web evidence supports, contradicts, or cannot
confirm the claims in SOURCE_PASSAGE that are relevant to answering
QUESTION, then produce an updated passage based on the web evidence.
</task>

<research_process>
1. Search the web before answering. Use multiple distinct searches when
   necessary to obtain sufficient and independent evidence.
2. Prioritize Italian sources and information applicable to Italy.
   Preference order for authority:
   a) Istituto Superiore di Sanità and Italian government institutions
   b) Italian public health institutions, hospitals, universities
   c) Recognized international scientific/medical organizations
   d) Peer-reviewed or evidence-based sources
3. Consult multiple independent reliable sources whenever possible.
   Do not treat all websites as equally authoritative — weigh conflicting
   claims by source authority, not just by count.
4. Focus only on information relevant to answering QUESTION — ignore
   parts of SOURCE_PASSAGE that are unrelated to it.
</research_process>

<comparison_logic>
Identify the claims in SOURCE_PASSAGE that are relevant to QUESTION, and
compare them against what the web evidence supports. Evaluate in this
exact order and stop at the first that applies:

1. NOT_ENOUGH_INFO — reliable web evidence is insufficient to confirm or
   refute the relevant claims in SOURCE_PASSAGE (this also applies if
   SOURCE_PASSAGE has no content relevant to QUESTION).
2. CULTURAL_DISCREPANCY — the difference between SOURCE_PASSAGE and the
   web evidence is meaningfully explained by Italian cultural, social,
   institutional, medical, or healthcare specificity (e.g., SOURCE_PASSAGE
   reflects a different country's practice).
3. CONTRADICTION — reliable web evidence factually contradicts a relevant
   claim in SOURCE_PASSAGE, and this is NOT explained by Italian
   cultural/institutional/healthcare context (e.g., the source is simply
   outdated or wrong).
4. NO_DISCREPANCY — web evidence is compatible with the relevant claims
   in SOURCE_PASSAGE.

A single question/passage pair gets exactly one label. If a factual difference is specifically explained by Italian cultural,
institutional, social, medical, or healthcare context, classify it as
CULTURAL_DISCREPANCY rather than CONTRADICTION.

Use CONTRADICTION when the disagreement is factual and cannot reasonably
be explained by such contextual differences.
</comparison_logic>

<writing_instructions>
- Write the passage in the same language as SOURCE_PASSAGE. If
  SOURCE_PASSAGE is empty or not provided, write in Italian.
- 80-150 words, self-contained (understandable without reading QUESTION
  or SOURCE_PASSAGE).
- Base the passage on the web evidence found, not on SOURCE_PASSAGE —
  do not simply repeat or lightly edit the original text.
- Include only information necessary to answer QUESTION.
- Do not include unsupported or invented information.
</writing_instructions>

<output_format>
Return ONLY valid JSON with this exact structure, no extra text:
{{
  "passage_web": "<string>",
  "discrepancy_label": "<one of: NO_DISCREPANCY | CONTRADICTION | CULTURAL_DISCREPANCY | NOT_ENOUGH_INFO>",
  "sources": ["<url or source name>", "..."]
}}
</output_format>
"""


# ============================================================
# GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)


# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_question(
    question: str,
    source_chunk: str
) -> SearchResult:

    prompt = PROMPT.format(
        question=question,
        source_chunk=source_chunk
    )


    interaction = client.interactions.create(

        model=MODEL,

        input=prompt,

        tools=[
            {
                "type": "google_search"
            }
        ],


        # Structured JSON response
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": SearchResult.model_json_schema(),
        },
    )

    result = SearchResult.model_validate_json(
        interaction.output_text
    )

    return result


# ============================================================
# LOAD CASE 2
# ============================================================

df = pd.read_parquet(
    INPUT_PATH
)

print(
    f"Rows in CASE 2 parquet: "
    f"{len(df):,}"
)


# ============================================================
# ONE ROW PER QUESTION
# ============================================================
#
# We only want one Gemini call for each question.
#
# The original parquet contains approximately 20 retrieved
# comparisons for each question.
# ============================================================

questions = (
    df[
        [
            "source_chunk_id",
            "question",
            "source_chunk",
        ]
    ]
    .drop_duplicates(
        subset=[
            "source_chunk_id",
            "question",
        ]
    )
    .reset_index(drop=True)
)


print(
    f"Unique questions to process: "
    f"{len(questions):,}"
)


# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

if OUTPUT_PATH.exists():

    previous_results = pd.read_parquet(
        OUTPUT_PATH
    )

    processed_keys = set(
        zip(
            previous_results["source_chunk_id"],
            previous_results["question"],
        )
    )

    print(
        f"Already processed: "
        f"{len(previous_results):,}"
    )

else:

    previous_results = pd.DataFrame()

    processed_keys = set()

    print(
        "No previous checkpoint found."
    )


# ============================================================
# PROCESS QUESTIONS
# ============================================================

new_results = []

calls_this_run = 0


for index, row in questions.iterrows():

    # ========================================================
    # STOP AFTER MAX CALLS
    # ========================================================

    if calls_this_run >= MAX_CALLS_PER_RUN:

        print("\n========================================")
        print("MAXIMUM CALLS FOR THIS RUN REACHED")
        print("========================================")

        print(
            f"Calls made this run: "
            f"{calls_this_run:,}"
        )

        print(
            "Stopping safely. Run the same script "
            "again when the API quota is available."
        )

        break


    source_chunk_id = row["source_chunk_id"]
    question = row["question"]
    source_chunk = row["source_chunk"]

    key = (
        source_chunk_id,
        question,
    )


    # ========================================================
    # SKIP ALREADY PROCESSED
    # ========================================================

    if key in processed_keys:
        continue


    print("\n========================================")

    print(
        f"Dataset position: "
        f"{index + 1:,}/{len(questions):,}"
    )

    print(
        f"API call this run: "
        f"{calls_this_run + 1:,}/{MAX_CALLS_PER_RUN:,}"
    )

    print(
        f"Already processed before this run: "
        f"{len(processed_keys):,}"
    )

    print(
        f"Anchor: {source_chunk_id}"
    )

    print(
        f"Question: {question}"
    )


    # ========================================================
    # GEMINI SEARCH
    # ========================================================

    try:

        # Count the attempt immediately.
        # Even failed API requests may consume quota.
        calls_this_run += 1

        result = search_question(
            question = question,
            source_chunk = source_chunk
        )

        new_results.append(
        {
            "source_chunk_id": source_chunk_id,
            "question": question,
            "source_chunk": source_chunk,

            "passage_web": result.passage_web,

            "web_discrepancy_label":
                result.discrepancy_label,

            "sources": result.sources,

            "error": None,
        }
    )

        print(
            f"Label: "
            f"{result.discrepancy_label}"
        )

        print(
            f"Passage: "
            f"{result.passage_web[:200]}..."
        )

        print(
            f"Sources: "
            f"{result.sources}"
        )


    except Exception as e:

        print(
            f"ERROR: {e}"
        )

        new_results.append(
            {
                "source_chunk_id": source_chunk_id,
                "question": question,
                "source_chunk": row["source_chunk"],
                "passage_web": None,
                "web_discrepancy_label": None,
                "sources": None,
                "error": str(e),
            }
        )


    # ========================================================
    # CHECKPOINT EVERY 20 CALLS
    # ========================================================

    if len(new_results) >= CHECKPOINT_EVERY:

        checkpoint_df = pd.DataFrame(
            new_results
        )

        if not previous_results.empty:

            output_df = pd.concat(
                [
                    previous_results,
                    checkpoint_df,
                ],
                ignore_index=True,
            )

        else:

            output_df = checkpoint_df


        # ----------------------------------------------------
        # Remove possible duplicates
        # ----------------------------------------------------

        output_df = (
            output_df
            .drop_duplicates(
                subset=[
                    "source_chunk_id",
                    "question",
                ],
                keep="last"
            )
            .reset_index(drop=True)
        )


        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        output_df.to_parquet(
            OUTPUT_PATH,
            index=False
        )


        # ----------------------------------------------------
        # Update processed questions
        # ----------------------------------------------------

        processed_keys.update(
            zip(
                checkpoint_df["source_chunk_id"],
                checkpoint_df["question"],
            )
        )


        print("\n========================================")
        print("CHECKPOINT")
        print("========================================")

        print(
            f"Calls this run: "
            f"{calls_this_run:,}/"
            f"{MAX_CALLS_PER_RUN:,}"
        )

        print(
            f"Total questions saved: "
            f"{len(output_df):,}/"
            f"{len(questions):,}"
        )

        print(
            f"Remaining questions: "
            f"{len(questions) - len(output_df):,}"
        )

        print(
            f"Path: {OUTPUT_PATH}"
        )


        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        previous_results = output_df

        new_results = []

# ============================================================
# SAVE REMAINING RESULTS
# ============================================================

if new_results:

    checkpoint_df = pd.DataFrame(
        new_results
    )

    if not previous_results.empty:

        output_df = pd.concat(
            [
                previous_results,
                checkpoint_df,
            ],
            ignore_index=True,
        )

    else:

        output_df = checkpoint_df

    output_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

else:

    output_df = previous_results


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("PROCESS FINISHED")
print("========================================")

print(
    f"Processed questions: "
    f"{len(output_df):,}"
)

print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)


if (
    not output_df.empty
    and "web_discrepancy_label" in output_df.columns
):

    print("\n========================================")
    print("WEB LABEL DISTRIBUTION")
    print("========================================")

    print(
        output_df[
            "web_discrepancy_label"
        ]
        .value_counts(
            dropna=False
        )
    )