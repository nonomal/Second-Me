from itertools import islice
import concurrent.futures
import json
import logging
import os
import random
import re

from tqdm import tqdm
import openai
from enum import Enum
from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.configs.config import Config
from lpm_kernel.L2.data_pipeline.data_prep.preference.prompts import (
    CH_USR_TEMPLATES,
    EN_USR_TEMPLATES,
    CH_SYS_TEMPLATES,
    EN_SYS_TEMPLATES,
)
import traceback


class LowMode(Enum):
    cluster_nums = 2
    

class HighMode(Enum):
    cluster_nums = 1


class PreferenceQAGenerator:
    def __init__(self, filename: str, bio: str, preference_language: str):
        """Initialize the PreferenceQAGenerator class.
        
        Args:
            filename: Path to the input JSON file containing preference messages.
            bio: Biography or context information to use in prompt generation.
            preference_language: Language for prompts ("Chinese/中文" or otherwise English).
        """
        self.filename = filename
        
        with open(self.filename, "r", encoding="utf-8") as f:
            self.pre_msg = json.load(f)

        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        if user_llm_config is None:
            self.client = None
            self.model_name = None
        else:
            self.model_name = user_llm_config.chat_model_name
    
            self.client = openai.OpenAI(
                api_key=user_llm_config.chat_api_key,
                base_url=user_llm_config.chat_endpoint,
            )
        
        self.bio = bio
        self.question_list = []
        self.preference_language = preference_language
        self.prompt_templates = self._get_prompt_templates(preference_language)
        self.sys_templates = self._get_sys_templates(preference_language)
        self.max_workers = 1
        self.data_synthesis_mode = os.environ.get("DATA_SYNTHESIS_MODE", "standard")


    def generate_response(self, sys: str, prompt: str) -> str:
        """Generate a response using the OpenAI API.
        
        Args:
            sys: The system prompt to use.
            prompt: The user prompt to send to the API.
            
        Returns:
            The generated response text or None if an error occurred.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future = executor.submit(
                    self.client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": prompt},
                    ],
                )
                response = future.result()
                return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error generating response: {e}")
            return None


    def clean_chunk(self, chunk: str) -> str:
        """Clean and process a text chunk.
        
        Args:
            chunk: The text chunk to clean.
            
        Returns:
            Cleaned text after processing.
        """
        after = chunk.split(":")[1:]
        after = ":".join(after)
        return after[1:]


    def _get_prompt_templates(self, preference_language: str) -> dict:
        """Return a dictionary of prompt templates based on language preference.
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of prompt templates.
        """
        if preference_language == "Chinese":
            return CH_USR_TEMPLATES
        else:
            return EN_USR_TEMPLATES


    def _get_sys_templates(self, preference_language: str) -> dict:
        """Return a dictionary of system templates based on language preference.
        
        Args:
            preference_language: The language preference ("Chinese/中文" or otherwise English).
            
        Returns:
            A dictionary of system templates.
        """
        if preference_language == "Chinese":
            return CH_SYS_TEMPLATES
        else:
            return EN_SYS_TEMPLATES


    def process_clusters(self, output_filename: str) -> None:
        """Process clusters and generate questions and answers.
        
        Args:
            output_filename: Path to save the generated Q&A pairs.
        """
        cluster_items = list(self.pre_msg.items())
        count = 0
        new_cluster_items = random.sample(cluster_items, len(cluster_items) // HighMode.cluster_nums.value 
                                          if self.data_synthesis_mode == "standard" else len(cluster_items) // LowMode.cluster_nums.value)
        for _, cluster in tqdm(new_cluster_items):
            chunk_concat = self._get_chunk_concat(cluster["contents"])

            tags = " ".join(cluster["tags"])

            if len(chunk_concat) < 20:
                continue
            count += 1
            
            n_cluster = len(cluster["contents"])
            if n_cluster > 1:
                logging.info(f"Cluster has {str(n_cluster)} chunks")

            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
            except Exception as e:
                logging.error(traceback.format_exc())
                continue
            try:
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
            except Exception as e:
                logging.error(traceback.format_exc())
                continue
            
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
            if n_cluster >= 20:
                self._generate_multiple_questions(cluster["contents"], chunk_concat)
            if count % 5 == 0:
                logging.info(f"Processed {count} clusters")

        with open(output_filename, "w") as json_file:
            json.dump(self.question_list, json_file, indent=4, ensure_ascii=False)


    def _get_chunk_concat(self, contents: list) -> str:
        """Concatenate and clean chunks of text.
        
        Args:
            contents: List of content chunks to concatenate.
            
        Returns:
            Concatenated text with formatting.
        """
        chunk_concat = ""
        for content in contents:
            chunk_content = content
            chunk_concat += chunk_content
            chunk_concat += "\n\n"
        return chunk_concat


    def _generate_multiple_questions(self, contents: list, chunk_concat: str) -> None:
        """Generate multiple questions and answers for larger clusters.
        
        Args:
            contents: List of content chunks.
            chunk_concat: Concatenated text chunks.
        """
        num_chunk_referred = 30
        n_repeat = max(1, int(len(contents) * 1 / num_chunk_referred))
        chunk_content_list = [
            self.clean_chunk(content)
            for content in contents
            if len(self.clean_chunk(content)) >= 80
        ]

        logging.info(f"Big cluster: n_repeat = {n_repeat}")

        for i in range(n_repeat):
            if i % 5 == 0 and i > 0:
                logging.info(f"Repeat {i} times")
            selected_chunks = random.sample(
                chunk_content_list, min(len(chunk_content_list), num_chunk_referred)
            )
            chunk_concat = "\n".join(selected_chunks)
            prompt_question_template = self.prompt_templates["query"]
            prompt_answer_template = self.prompt_templates["answer"]
            sys_question = self.sys_templates["query"]
            sys_answer = self.sys_templates["answer"]

            try:
                gen_question = self.generate_response(
                    sys_question,
                    prompt_question_template.format(
                        bio=self.bio, chunks_concat=chunk_concat
                    ),
                )
                gen_answer = self.generate_response(
                    sys_answer,
                    prompt_answer_template.format(
                        question=gen_question, chunks_concat=chunk_concat, bio=self.bio
                    ),
                )
            except Exception as e:
                logging.error(traceback.format_exc())
                continue
            self.question_list.append({"user": gen_question, "assistant": gen_answer})
        return
