from typing import Dict, Optional
import logging
import time
import os
import json
from lpm_kernel.file_data.document import Document
from lpm_kernel.L0.l0_generator import L0Generator
from lpm_kernel.L0.models import (
    InsighterInput,
    SummarizerInput,
    FileInfo,
    BioInfo,
    DocumentType,
)
from lpm_kernel.configs.config import Config
from lpm_kernel.file_data.document_dto import DocumentDTO


logger = logging.getLogger(__name__)


def get_preferred_language():

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    local_params_path = os.path.join(base_dir, "data", "progress", "training_params.json")
    cloud_params_path = os.path.join(base_dir, "data", "cloud_progress", "cloud_training_params.json")
    
    try:
        if os.path.exists(local_params_path):
            with open(local_params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
                if "language" in params:
                    return params["language"]
        else:
            if os.path.exists(cloud_params_path):
                with open(cloud_params_path, 'r', encoding='utf-8') as f:
                    params = json.load(f)
                    if "language" in params:
                        return params["language"]
    except Exception as e:
        logging.warning(f"Failed to read training parameters: {str(e)}")
    
    return "en"


class InsightKernel:
    def __init__(self):
        self.generator = L0Generator()
        self.preferred_language = get_preferred_language()

    def analyze(self, doc: DocumentDTO) -> Dict:
        """Generate document insight"""
        try:
            self.generator.preferred_language = self.preferred_language
            document_type = DocumentType.from_mime_type(doc.mime_type)

            if document_type is DocumentType.TEXT:
                return {
                    "title": "",
                    "insight": doc.raw_content,
                }

            # Prepare input data
            file_info = FileInfo(
                data_type=document_type.value,
                filename=doc.name,
                content="",
                file_content={"content": doc.raw_content},
            )

            bio_info = BioInfo(global_bio="", status_bio="", about_me="")

            insighter_input = InsighterInput(file_info=file_info, bio_info=bio_info)

            insight_result = self.generator.insighter(insighter_input)

            return {
                "title": insight_result.get("title"),
                "insight": insight_result.get("insight"),
            }

        except Exception as e:
            logger.error(f"Failed to generate insight: {str(e)}", exc_info=True)
            raise Exception(f"Error generating insight: {e}")


class SummaryKernel:
    def __init__(self):
        self.generator = L0Generator()
        self.preferred_language = get_preferred_language()

    def analyze(self, doc: DocumentDTO, insight: str = "") -> Dict:
        """Generate document summary"""

        try:
            self.generator.preferred_language = self.preferred_language
            document_type = DocumentType.from_mime_type(doc.mime_type)

            if document_type is DocumentType.TEXT:
                return {
                    "title": "",
                    "summary": doc.raw_content,
                    "keywords": [],
                }

            # Prepare input data
            file_info = FileInfo(
                data_type=document_type.value,
                filename=doc.name,
                content="",
                file_content={"content": doc.raw_content},
            )

            summarizer_input = SummarizerInput(file_info=file_info, insight=insight)

            # Call LLM
            summary_result = self.generator.summarizer(summarizer_input)

            return {
                "title": summary_result.get("title"),
                "summary": summary_result.get("summary"),
                "keywords": summary_result.get("keywords", []),
            }

        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}", exc_info=True)
            raise Exception(f"Error generating summary: {e}")
