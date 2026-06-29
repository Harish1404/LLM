import json
import logging
import asyncio
from app.ai.chatbot import call_llm_with_retry, resolve_model_name
from app.core.config import settings
from app.ai.llm_metrics import metrics
from app.prompts.chain_prompts import STAGE_1_EXTRACT, STAGE_2_ENRICH, STAGE_3_FORMAT

logger = logging.getLogger(__name__)


async def call_llm_non_streaming(model_key: str, messages: list) -> str:
    """
    Calls the LLM non-streaming with complete retry and provider fallback support.
    Runs the synchronous call_llm_with_retry in a separate thread to prevent blocking the event loop.
    """
    primary_model = resolve_model_name(model_key)
    gemini_key = settings.gemini_api_key
    groq_key = settings.groq_api_key

    # Dynamically build fallback list based on model provider
    fallback_chain = []
    if "gemini" in primary_model.lower():
        fallback_chain.append((primary_model, gemini_key))
        fallback_chain.append(("groq/llama-3.1-8b-instant", groq_key))
    else:
        fallback_chain.append((primary_model, groq_key))
        fallback_chain.append(("gemini/gemini-2.5-flash", gemini_key))

    print(f"\nfallback_chain = {fallback_chain}\n")
    last_error = None
    for model_name, api_key in fallback_chain:
        if not api_key:
            continue
        try:
            logger.info(f"Chain step calling model: {model_name}...")
            response = await asyncio.to_thread(
                call_llm_with_retry,
                model=model_name,
                messages=messages,
                api_key=api_key,
                temperature=0.3
            )
            # Log usage metrics
            metrics.log_metrics(model_name=model_name, response=response)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Chain step failed for {model_name}: {e}. Trying fallback...")
            last_error = e
            continue

    raise Exception(f"All models in fallback chain failed. Last error: {last_error}")


class ChainService:
    @staticmethod
    def clean_json_response(content: str) -> dict:
        """
        Cleans JSON string by stripping markdown fencing blocks and whitespace, then parses it.
        """
        content_clean = content.strip()
        if content_clean.startswith("```"):
            lines = content_clean.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content_clean = "\n".join(lines).strip()
        return json.loads(content_clean)

    @classmethod
    async def run_concept_chain(cls, raw_text: str, model_key: str = "fast") -> dict:
        """
        Executes the three-stage chain:
        1. Extract: Extracts structured concepts/context from raw text.
        2. Enrich: Enriches each concept with research details.
        3. Format: Formats the enriched data into a Markdown report.
        """
        logger.info("Starting multi-stage LLM chain execution...")
        execution_history = []

        # ─────────────────────────────────────────────
        # STAGE 1: Extract concepts
        # ─────────────────────────────────────────────
        messages_1 = [
            {"role": "system", "content": STAGE_1_EXTRACT},
            {"role": "user", "content": raw_text}
        ]

        try:
            logger.info("Stage 1: Extracting facts/concepts...")

            raw_extraction = await call_llm_non_streaming(model_key, messages_1)
            parsed_extraction = cls.clean_json_response(raw_extraction)
            
            concepts = parsed_extraction.get("concepts", [])
            execution_history.append({
                "stage": "extract",
                "raw_output": raw_extraction,
                "parsed_output": parsed_extraction,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Stage 1 Extraction failed: {e}")
            return {
                "success": False,
                "error": f"Stage 1 (Extraction) failed: {str(e)}",
                "execution_history": execution_history
            }

        if not concepts:
            return {
                "success": False,
                "error": "Stage 1 returned no concepts to process.",
                "execution_history": execution_history
            }

        # ─────────────────────────────────────────────
        # STAGE 2: Enrich each concept
        # ─────────────────────────────────────────────
        enriched_concepts = []
        logger.info(f"Stage 2: Enriching {len(concepts)} concepts...")

        for index, concept in enumerate(concepts):
            concept_name = concept.get("name", "Unknown")
            concept_context = concept.get("context", "")

            messages_2 = [
                {"role": "system", "content": STAGE_2_ENRICH},
                {
                    "role": "user",
                    "content": f"Concept: {concept_name}\nContext: {concept_context}"
                }
            ]

            try:
                logger.info(f"Enriching concept {index+1}/{len(concepts)}: '{concept_name}'...")
                enrichment_text = await call_llm_non_streaming(model_key, messages_2)
                enriched_concepts.append({
                    "name": concept_name,
                    "context": concept_context,
                    "enrichment": enrichment_text
                })
            except Exception as e:
                logger.warning(f"Failed to enrich concept '{concept_name}': {e}. Using fallback empty enrichment.")
                enriched_concepts.append({
                    "name": concept_name,
                    "context": concept_context,
                    "enrichment": "Enrichment failed due to an API error."
                })

        execution_history.append({
            "stage": "enrich",
            "enriched_concepts": enriched_concepts,
            "status": "success"
        })

        # ─────────────────────────────────────────────
        # STAGE 3: Format Report
        # ─────────────────────────────────────────────
        input_data_for_format = json.dumps(enriched_concepts, indent=2)
        messages_3 = [
            {"role": "system", "content": STAGE_3_FORMAT},
            {"role": "user", "content": input_data_for_format}
        ]

        try:
            logger.info("Stage 3: Formatting final Markdown report...")
            final_report = await call_llm_non_streaming(model_key, messages_3)
            execution_history.append({
                "stage": "format",
                "final_report": final_report,
                "status": "success"
            })

            return {
                "success": True,
                "final_report": final_report,
                "execution_history": execution_history
            }
        except Exception as e:
            logger.error(f"Stage 3 Formatting failed: {e}")
            return {
                "success": False,
                "error": f"Stage 3 (Formatting) failed: {str(e)}",
                "execution_history": execution_history
            }
