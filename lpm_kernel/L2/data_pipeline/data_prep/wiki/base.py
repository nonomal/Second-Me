import sys
import os
import statistics
import re
import json
import traceback
import logging

from tqdm import tqdm
import tiktoken
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import openai

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.L2.data_pipeline.data_prep.wiki.prompt import Prompts
from lpm_kernel.L2.data_pipeline.data_prep.wiki.utils import Note,Entity,Timeline,EntityWiki,Conversation,TimelineType,EntityType,MonthlyTimeline,MONTH_TIME_FORMAT
from lpm_kernel.L2.data_pipeline.data_prep.wiki.utils import (build_clusters,
                    find_neibor_entities, \
                    parse_monthly_timelines,
                    parse_daily_timelines,group_timelines_by_month,group_timelines_by_day, \
                    is_valid_note)

from lpm_kernel.configs.logging import get_train_process_logger
logger = get_train_process_logger()

PHRASES_LENGTH_THRESHOLD = 3
LONG_PHRASES_DISTANCE_THRESHOLD = 3
SHORT_PHRASES_DISTANCE_THRESHOLD = 1
RATE_THRESHOLD = 0.5
NEIBOR_ENTITY_N = 5
ENTITY_MERGE_BATCH_SIZE = CONCEPT_MERGE_BATCH_SIZE = 3

ENTITY_EXTRACTOR_DEFAULT_PROMPT = {
    "entity_extractor": Prompts.ENTITY_EXTRACT_SYSTEM_PROMPT_simplfied,
    "duplicate_entities": Prompts.DUPLICATE_ENTITY_SYSTEM_PROMPT,
    "merge_entities": Prompts.MERGE_ENTITY_SYSTEM_PROMPT,
    "extract_filter": Prompts.ENTITY_EXTRACT_FILTER_SYSTEM_PROMPT,
    "generate_timeline_by_note": Prompts.GENERATE_TIMELINE_BY_NOTE_SYSTEM_PROMPT
}

PERSONAL_WIKI_DEFAULT_PROMPT = {
    "personal_wiki_entity": Prompts.PERSONAL_WIKI_ENTITY_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_person": Prompts.PERSONAL_WIKI_PERSON_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_location": Prompts.PERSONAL_WIKI_LOCATION_SYSTEM_PROMPT_gt_v0,
    "personal_wiki_concept": Prompts.PERSONAL_WIKI_CONCEPT_SYSTEM_PROMPT,
    "timeline_generate": Prompts.TIMELINE_GENERATE_SYSTEM_PROMPT_gt_v0,
    "monthly_timeline_title_entity": Prompts.MONTHLY_TIMELINE_TITLE_ENTITY_SYSTEM_PROMPT,
    "monthly_timeline_title_person": Prompts.MONTHLY_TIMELINE_TITLE_PERSON_SYSTEM_PROMPT,
    "monthly_timeline_title_location": Prompts.MONTHLY_TIMELINE_TITLE_LOCATION_SYSTEM_PROMPT,
    "monthly_timeline_title_concept": Prompts.MONTHLY_TIMELINE_TITLE_CONCEPT_SYSTEM_PROMPT,
}


class EntityScorer:
    def __init__(self, langfuse_dict: Dict[str, Any]):
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
        self.llm = openai.OpenAI(api_key=user_llm_config.chat_api_key,base_url=user_llm_config.chat_endpoint)
        logger.info("Entity Scorer Initialized.")
        self.langfuse_dict = langfuse_dict

    def score_entities(self, entities: List[Entity], notes: List[Note], conversations: List[Conversation],
                       global_bio: str, user_name: str) -> List[Entity]:
        max_workers = max(1, min(2, len(entities)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._calculate_entity_score, entity, notes, conversations, global_bio, user_name)
                for entity in entities]
            scored_entities = [future.result() for future in futures]

        return scored_entities

    def update_entity_scores(self, entities: List[Entity], notes: List[Note], global_bio: str, user_name: str) -> List[
        Entity]:
        max_workers = max(1, min(2, len(entities)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._recalculate_entity_score, entity, notes, global_bio, user_name)
                       for entity in entities]
            updated_entities = [future.result() for future in futures]

        return updated_entities

    def _calculate_entity_score(self, entity: Entity, notes: List[Note], conversations: List[Conversation],
                                global_bio: str, user_name: str) -> Entity:
        note_ids = [int(timeline.note_id) for timeline in entity.timelines]
        entity_notes = [note for note in notes if note.id in note_ids]
        for entity_note in entity_notes:
            entity_note.content = next((note.content for note in notes if note.id == entity_note.id), "")

        # build prompt messages
        message = Prompts.return_introspection_extracted_entities_prompt(
            entity_name=entity.name,  
            input_notes=entity_notes,  
            user_name=user_name,  
            global_bio=global_bio,  
            system_prompt=self.langfuse_dict['extract_filter']['system_prompt']  # 系统提示
        )
        answer = self.llm.chat.completions.create(
            model=self.model_name,
            messages=message,
        )
        content = answer.choices[0].message.content
        # parse and update entity score
        self._parse_and_update_score(entity, content)
        return entity

    def _recalculate_entity_score(self, entity: Entity, notes: List[Note], global_bio: str, user_name: str) -> Entity:
        timeline_contents = [timeline.content for timeline in entity.timelines]
        # merge note and timelien contents
        combined_contents = timeline_contents + [note.content for note in notes]

        entity_notes = [note for note in notes if note.content in combined_contents]

        message = Prompts.return_introspection_extracted_entities_prompt(
            entity_name=entity.name,  
            input_notes=entity_notes,  
            user_name=user_name,  
            global_bio=global_bio,  
            system_prompt=self.langfuse_dict['extract_filter']['system_prompt']  # 系统提示
        )
        answer = self.llm.chat.completions.create(
            model=self.model_name,
            messages=message,
        )
        content = answer.choices[0].message.content
        self._parse_and_update_score(entity, content)
        return entity

    def _parse_and_update_score(self, entity: Entity, content: str):
        try:
            # extract json from content
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if not json_match:
                score_data = json.loads(content)
            else:
                score_data = json.loads(json_match.group(1))

            # update entity scores and reasons
            entity.personal_uniqueness_reason = score_data['personal_uniqueness']['reason']
            personal_uniqueness_score = score_data['personal_uniqueness']['score']
            entity.contextual_relevance_reason = score_data['contextual_relevance']['reason']
            contextual_relevance_score = score_data['contextual_relevance']['score']
            entity.personal_significance_reason = score_data['personal_significance']['reason']
            personal_significance_score = score_data['personal_significance']['score']
            entity.rarity_reason = score_data['rarity']['reason']
            rarity_score = score_data['rarity']['score']
            entity.time_relevance_reason = score_data['time_relevance']['reason']
            time_relevance_score = score_data['time_relevance']['score']
            entity.frequency_reason = score_data['frequency']['reason']
            frequency_score = score_data['frequency']['score']
            entity.emotional_connection_reason = score_data['emotional_connection']['reason']
            emotional_connection_score = score_data['emotional_connection']['score']

            # calculate average score
            scores = [
                personal_uniqueness_score,
                contextual_relevance_score,
                personal_significance_score,
                rarity_score,
                time_relevance_score,
                emotional_connection_score,
                frequency_score
            ]

            average_score = statistics.mean(scores)
            # entity.score = average_score
            # update when the new score is higher than the current score
            if average_score > entity.score and average_score > 0:
                entity.score = average_score

            # threshold of wiki generation
            if average_score > 0.4:
                entity.gen_wiki = True
            elif average_score > 0.2 and entity.freq >= 2:
                entity.gen_wiki = True
            else:
                entity.gen_wiki = False

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logging.error(f"Error parsing entity score response: {str(e)}\nContent: {content}")
            entity.score = 0
            entity.gen_wiki = False
            entity.personal_uniqueness_reason = ""
            entity.contextual_relevance_reason = ""
            entity.personal_significance_reason = ""
            entity.rarity_reason = ""
            entity.time_relevance_reason = ""
            entity.frequency_reason = ""
            entity.emotional_connection_reason = ""



class EntityExtractor:
    _input_keys: List = ["userName", "aboutMe", "notes", "entities", "preferredLanguage"]
    _output_keys: List = ["entities"]
    _must_keys: List = ["userName", "notes", "entities"]

    model_params = {
        "temperature": 0,
        "max_tokens": 8000,
        "top_p": 0,
        "frequency_penalty": 0,
        "seed": 42,
        "presence_penalty": 0,
        # "request_timeout": 60,
        # "max_retries": 1
    }

    def __init__(self, **kwargs):
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
        logger.info("Entity Extractor initialized.")
        self.max_threads = 2
        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.model_params.update(**kwargs)
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in ENTITY_EXTRACTOR_DEFAULT_PROMPT.items()
        }
        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=user_llm_config.chat_api_key,
                base_url=user_llm_config.chat_endpoint,
            )
            for k, _ in ENTITY_EXTRACTOR_DEFAULT_PROMPT.items()
        }

    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        system_prompt = default_prompt
        return { "system_prompt": system_prompt}


    def _call_(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用测试函数。
        Args:
            inputs (Dict[str, Any]): dictionary include user info.
        Returns:
            Dict[str, Any]: Entities dictionary.
        """

        # get input
        entity_extract_input = inputs
        user_name = entity_extract_input.get("userName", "")
        notes = entity_extract_input.get("notes", [])
        conversations = entity_extract_input.get("conversations", [])
        entities = entity_extract_input.get("entities", [])
        user_self_intro = entity_extract_input.get("aboutMe", "")
        preferred_language = entity_extract_input.get("preferredLanguage", "English/English")
        global_bio = entity_extract_input.get("globalBio", "")
        if global_bio:
            conclusion_match = re.search(r'Conclusion ###\s*([\s\S]*?)$', global_bio)
            if conclusion_match:
                global_bio = conclusion_match.group(1).strip()
        original_entities = entity_extract_input.get("entities", [])

        notes = [Note(**note, userName=user_name) for note in notes]
        conversations = [Conversation(**conversation, userName=user_name) for conversation in conversations]
        old_entities = [Entity(**entity) for entity in entities]

        # extract entities from input content
        raw_entities = self.extract_entities(notes, conversations, preferred_language)
        entity_scorer = EntityScorer(self.langfuse_dict)
        updated_entities = []

        # Initialization
        if not old_entities:
            logging.info(f"The user has no historical entity list, starting the entity list initial process!")
            entities = self.initialize_entity_list(raw_entities, preferred_language, notes, conversations, global_bio,
                                                   user_name)
            entities = entity_scorer.score_entities(entities, notes, conversations, global_bio, user_name)
            
            updated_entities = entities
            logging.info(
                f"The entity list initial process is completed! Successfully extracted {len(entities)} entities!")
        # If user has historical entity list, update the entity list
        else:
            logging.info(f"The user has historical entity list, starting the entity list update process!")
            new_entities = self.initialize_entity_list(raw_entities, preferred_language, notes, conversations,
                                                       global_bio, user_name)
            logging.info(f"After merge duplicate entity, Entity nums: {len(raw_entities)} -> {len(new_entities)}")

            # update entities list and get entities that need to score
            entities, entities_to_score = self.update_entity_list(old_entities, new_entities)
            # score entities
            scored_entities = entity_scorer.update_entity_scores(entities_to_score, notes, global_bio, user_name)

            # update list with new score
            entity_dict = {entity.name: entity for entity in entities}
            for scored_entity in scored_entities:
                if scored_entity.name not in entity_dict:
                    logging.warning(f"Entity name '{scored_entity.name}' not found in entity_dict. Adding it now.")
                    entity_dict[scored_entity.name] = scored_entity
                elif scored_entity.score > entity_dict[scored_entity.name].score:
                    entity_dict[scored_entity.name] = scored_entity
            entities = list(entity_dict.values())

            logging.info(
                f"The entity list update process is completed! Successfully extracted {len(entities)} entities!")
            updated_entities = self.compare_and_keep_max_scores([],scored_entities)

        final_entities = self.compare_and_keep_max_scores(original_entities, entities)
        
        return {
        "entities": [entity.to_dict() for entity in final_entities if entity.timelines],
        "updated_entities": [entity.to_dict() for entity in updated_entities] if updated_entities else []
    }

    def compare_and_keep_max_scores(self, original_entities: List[dict], updated_entities: List[Entity]) -> List[
        Entity]:
        entity_map = {entity.name: entity for entity in updated_entities}

        for entity_dict in original_entities:
            entity_name = entity_dict['name']
            original_score = float(entity_dict['score'])

            # if entity exists and has higher current score
            if entity_name in entity_map:
                if original_score > entity_map[entity_name].score:
                    # keep original score and update other fields
                    entity_map[entity_name].score = original_score
                    entity_map[entity_name].freq = entity_dict.get('freq', entity_map[entity_name].freq)
            else:
                new_entity = Entity(
                    name=entity_name,
                    entityType=EntityType(entity_dict['entity_type']),
                    score=original_score,
                    freq=entity_dict['freq'],
                    synonyms=entity_dict.get('synonyms', []),
                    timelines=[Timeline(**t) for t in entity_dict.get('timelines', [])]
                )
                entity_map[entity_name] = new_entity

        return list(entity_map.values())

    def extract_entities(self, notes: List[Note], conversations: List[Conversation], prefer_lang: str) -> List[Dict[str, Any]]:
        """
        Extract entities from the given notes and conversations.

        Args:
            notes (List[Note]): A list containing all notes.
            conversations (List[Conversation]): A list containing all conversations.
            prefer_lang (str): The user's preferred language.

        Returns:
            List[Dict[str, Any]]: A list of extracted entities, with each entity represented as a dictionary.
        """

        # solve item one by one 
        def process_item(item: Union[Note, Conversation]) -> list[dict[str, Any]]:
            return self.extract_entities_by_notes([item], prefer_lang)

        all_items = notes + conversations
        results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(process_item, item) for item in tqdm(all_items, desc="Extracting Entities")]
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error processing item: {e}")

        return results

    def extract_entities_by_notes(self, entity_input: List[Union[Note, Conversation]],prefer_lang: str) -> List[Dict[str, Any]]:
        message = Prompts.return_entity_extract_prompt(prefer_lang=prefer_lang,
                                                       entity_input=entity_input,
                                                       user_name=entity_input[0].user_name,
                                                       user_self_intro="",
                                                       global_bio="",
                                                       system_prompt=self.langfuse_dict['entity_extractor'][
                                                           'system_prompt'])
        answer = self.llms["entity_extractor"].chat.completions.create(messages=message,model=self.model_name,**self.model_params)
        content = answer.choices[0].message.content
        return self.extract_entity_postprocess_new(content)

    def extract_entity_postprocess_new(self, raw_result: str) -> List[Dict[str, Any]]:
        print(raw_result)
        json_data_str = re.search(r'```json\n([\s\S]+?)\n```', raw_result).group(1)
        json_data = json.loads(json_data_str)
        final_entities = []
        for entity_data in json_data:
            entity_name = self.clean_entity_name(entity_data['entity_name'])
            entity_type = entity_data['entity_type']
            entity_desc = entity_data['entity_description']
            entity_notes = entity_data['notes']
            final_entities.append({
                "name": entity_name,
                "entity_type": entity_type,
                "description": entity_desc,
                "notes": entity_notes
            })
        return final_entities

    def clean_entity_name(self, entity_name: str) -> str:
        # deal with upperclass and lowerclass, and some special characters 
        entity_name = entity_name.strip()
        entity_name = re.sub(r'\s+', ' ', entity_name)
        entity_name = re.sub(r'(\w+)\.(\w+)', r'\1\2', entity_name)
        
        return entity_name

    def initialize_entity_list(self, raw_entities: List[Dict[str, Any]], preferred_language, notes: List[Note],
                               conversations: List[Conversation], global_bio: str, user_name: str):
        """
        Initialize the entity list.

        Args:
            raw_entities (List[Dict[str, Any]]): The raw entity list, where each entity is a dictionary containing the entity name and other information.
            preferred_language (str): The user's preferred language.
            notes (List[Note]): A list of notes, where each note is an object containing note content.
            conversations (List[Conversation]): A list of conversations, where each conversation is an object containing conversation content.
            global_bio (str): Global biographical information, represented as a string.
            user_name (str): The username, represented as a string.

        Returns:
            List[Entity]: A list of entities, where each entity is an object containing the entity name, type, frequency, synonyms, and timelines.
        """
        # calculate the frequency of each entity
        entity_freqs = Counter(entity["name"] for entity in raw_entities)

        # get the entity name list that is longer than the threshold
        long_phrase_entities = list(set(entity["name"] for entity in raw_entities
                                        if len(entity["name"]) > PHRASES_LENGTH_THRESHOLD))

        # get the entity name list that is shorter than the threshold
        short_phrase_entities = list(set(entity["name"] for entity in raw_entities
                                         if len(entity["name"]) <= PHRASES_LENGTH_THRESHOLD))

        # build cluster based on the entity name list
        long_phrase_clusters = build_clusters(long_phrase_entities, LONG_PHRASES_DISTANCE_THRESHOLD)
        short_phrase_clusters = build_clusters(short_phrase_entities, SHORT_PHRASES_DISTANCE_THRESHOLD)

        # merge the results from the two clusters
        raw_entity_list = self.clustered_entities(long_phrase_clusters + short_phrase_clusters, entity_freqs)

        # build synonym map
        synonym_map = {
            synonym: entity_name
            for entity_name, entity_info in raw_entity_list.items()
            for synonym in entity_info["synonyms"]
        }
        note_dict = {note.id: note for note in notes}

        for entity in raw_entities:
            if entity["name"] not in synonym_map:
                synonym_map[entity["name"]] = entity["name"]
                raw_entity_list[entity["name"]] = {
                    "synonyms": [entity["name"]],
                    "timelines": []
                }
            for entity_note in entity.get("notes", []):
                try:
                    note_id = int(entity_note.get("note_id", ""))
                    origin_note = note_dict.get(note_id)
                    if not origin_note:
                        logging.warning(f"Note ID {note_id} not found")
                        continue
                    
                    content = origin_note.content
                    # generate timeline based on the note 
                    entity_timeline = self.gen_timeline_by_note(user_name, preferred_language, entity, entity_note, notes,
                                                                debug=False)
                    raw_entity_list[synonym_map[entity["name"]]]["timelines"].append(
                        Timeline(createTime=entity_note.get("create_time", ""),
                                noteId=str(note_id),
                                content=content,
                                description=entity_timeline,
                                timelineType=TimelineType(entity["entity_type"]))
                    )
                except (ValueError, KeyError, TypeError) as e:
                    logging.warning(f"Skip invalid note for entity {entity['name']}: {str(e)}")
                    continue
         

        # judge entity type
        for entity_name, entity_info in raw_entity_list.items():
            raw_entity_list[entity_name]["entity_type"] = self.judge_entity_type(entity_info["timelines"])

        logging.info(
            f"After duplicated entities processing, Entity nums: {len(raw_entities)} -> {len(raw_entity_list)}")

        entities = [
            Entity(name=entity_name,
                   entityType=entity_info["entity_type"],
                   freq=len(entity_info["timelines"]),
                   synonyms=entity_info["synonyms"],
                   timelines=entity_info["timelines"])
            for entity_name, entity_info in raw_entity_list.items()
        ]
        return entities

    def gen_timeline_by_note(self, user_name, preferred_language, entity, entity_note, notes,
                             debug: bool = False) -> str:

        # If we are in debug mode, return an empty string
        if debug == True:
            return ""

        origin_note = None
        for note in notes:
            # find match note
            if note.id == int(entity_note["note_id"]):
                origin_note = note

        if origin_note is None:
            return ""
        else:
            message = Prompts.generate_timeline_by_notes(user_name, preferred_language, entity, origin_note,
                                                         system_prompt=self.langfuse_dict['generate_timeline_by_note'][
                                                             'system_prompt'])
            answer = self.llms["entity_extractor"].chat.completions.create(messages=message,model=self.model_name,**self.model_params)
            content = answer.choices[0].message.content

            content_matches = re.search(r"### Entity Timeline(.*?)(###|$)", content, re.DOTALL)

            if not content_matches:
                logging.warning(f"can not extract timeline from content: {content[:100]}...")
                return ""
                
            raw_result = content_matches.group(1)

            content_section = raw_result.replace("\n", " ").strip()
            
            if ": " in content_section:
                timeline = content_section.split(": ", 1)[1]
            else:
                timeline = content_section

            entity_note["content"] = origin_note.content

            return timeline

    def clustered_entities(self, clusters: List[List[str]], entity_freqs: Counter):
        """
        Map the entities within each cluster to their synonym sets and calculate the frequency and synonyms for each entity.

        Args:
            clusters (List[List[str]]): A list of entity clusters, where each sublist represents a cluster containing all entity names in that cluster.
            entity_freqs (Counter): Entity frequency statistics, indicating how often each entity name appears.

        Returns:
            dict: A dictionary containing each entity name along with its synonym set and frequency. The dictionary keys are entity names, and the values are dictionaries containing "freq" (frequency), "synonyms" (list of synonyms), and "timelines" (list of timelines, unused in this function).
        """
        entity_cluster_dict = {}

        for cluster in clusters:
            # if the number of entity in cluster less than 2, build map directly.
            if len(cluster) < 2:
                entity_maps = {e: [e] for e in cluster}
            else:
                raw_entity_maps = self.deep_duplicated_entities(cluster)
                entity_maps = {
                    # chose the entity with the highest frequency as the main entity
                    max(s, key=lambda x: entity_freqs[x]): s
                    for s in raw_entity_maps.values()
                }

            for entity_name, synonyms in entity_maps.items():
                if entity_name not in entity_cluster_dict:
                    entity_cluster_dict[entity_name] = {
                        "freq": 0,
                        "synonyms": [],
                        "timelines": []
                    }
                for synonym in synonyms:
                    entity_cluster_dict[entity_name]["freq"] += entity_freqs[synonym]
                    entity_cluster_dict[entity_name]["synonyms"].append(synonym)

        return entity_cluster_dict

    def deep_duplicated_entities(self, entity_list: List[str]) -> Dict[str, List[str]]:
        """
        Deeply detect and process entity duplication issues.

        Args:
            entity_list (List[str]): The list of entities to be checked.

        Returns:
            Dict[str, List[str]]: A dictionary containing the processed entity mapping relationships.
        """
        default_entity_maps = {
            e: [e] for e in entity_list
        }

        duplicate_entity_pattern = r'\{.*\}'

        message = Prompts.return_duplicate_entity_prompt(entity_list=entity_list,
                                                         system_prompt=self.langfuse_dict["duplicate_entities"][
                                                             "system_prompt"])

        answer = self.llms["duplicate_entities"].chat.completions.create(messages=message, model=self.model_name,**self.model_params)

        content = answer.choices[0].message.content

        entity_maps = self.parse_json_response(content, default_entity_maps, duplicate_entity_pattern)

        return entity_maps

    def update_entity_list(self, old_entities: List[Entity], new_entities: List[Entity]) -> Tuple[
        List[Entity], List[Entity]]:
        """
        Update the existing entity list by merging newly extracted entities.
        Args:
            old_entities (List[Entity]): The existing list of entities.
            new_entities (List[Entity]): The newly extracted list of entities.
        Returns:
            Tuple[List[Entity], List[Entity]]: The updated list of entities and the list of entities that need to be re-scored.
        """
        old_entity_dict = {entity.name: entity for entity in old_entities}

        synonym_dict = {
            synonym: entity.name for entity in old_entities for synonym in entity.synonyms
        }

        old_entity_synonym_names = set(synonym_dict.keys())

        unmerged_entities: List[Entity] = []
        entities_to_score: List[Entity] = []  # entity list that need to be re-scored

        for entity in new_entities:
            if union_name := set(entity.synonyms) & old_entity_synonym_names:
                union_entity_dict = {}

                for name in union_name:
                    match_old_entity = old_entity_dict[synonym_dict[name]]
                    union_entity_dict[match_old_entity.name] = union_entity_dict.get(match_old_entity.name,
                                                                                     0) + match_old_entity.freq

                # find the entity with the highest frequency as the main entity
                max_entity_name = max(union_entity_dict, key=union_entity_dict.get)
                old_entity = old_entity_dict[max_entity_name]

                existing_note_ids = {timeline.note_id for timeline in old_entity.timelines}
                
                new_timelines = []
                for timeline in entity.timelines:
                    if timeline.note_id not in existing_note_ids:
                        new_timelines.append(timeline)
                        existing_note_ids.add(timeline.note_id)
                
                entity.timelines = new_timelines
                
                old_entity.merge_entity(entity)
                entities_to_score.append(old_entity) 
            else:
                unmerged_entities.append(entity)
                entities_to_score.append(entity)

        unmered_entity_dict = {entity.name: entity for entity in unmerged_entities}
        logging.info(f"After merge into old entities, Entity nums: {len(new_entities)} -> {len(unmerged_entities)}")
        old_entities = list(old_entity_dict.values())

        ## merge same object entity into old entities from new entities
        if unmerged_entities:
            # find the neibor entity that unmerged
            neibor_entity_dict = find_neibor_entities(unmerged_entities, old_entities, NEIBOR_ENTITY_N)
            entities_batch_list = self.batch_merge_preprocess(neibor_entity_dict, old_entities, unmered_entity_dict)

            with ThreadPoolExecutor(max_workers=min(2, len(entities_batch_list))) as executor:
                futures = [executor.submit(self.batch_merge_entity, entities_batch) for entities_batch in
                           entities_batch_list]
                results = [merge_res for future in futures for merge_res in future.result()]

            logging.info(f"unmered_entity_dict.keys: {list(unmered_entity_dict.keys())}")
            logging.info(f"new_entity_name_list: {[res['new_entity_name'] for res in results]}")
            logging.info(f"Merge Result Num: {len(results)}")

            for merge_res in results:
                new_entity_name = merge_res["new_entity_name"]
                if new_entity_name not in unmered_entity_dict:
                    logging.warning(f"Entity {new_entity_name} not in new entity dict!, merge model performance issue!")
                    continue

                if merge_res["merged"] and merge_res["merge_target"] in old_entity_dict:
                    old_entity = old_entity_dict[merge_res["merge_target"]]
                    old_entity.merge_entity(unmered_entity_dict[new_entity_name])
                    old_entity.entity_type = self.judge_entity_type(old_entity.timelines)
                    entities_to_score.append(old_entity)
                else:
                    new_entity = unmered_entity_dict[new_entity_name]
                    old_entities.append(new_entity)
                    entities_to_score.append(new_entity)

        return old_entities, entities_to_score

    def batch_merge_preprocess(self, neibor_entity_dict: Dict[str, Entity], old_entities: List[Entity],
                               new_entities: Dict[str, Entity]) -> List[Dict[str, Any]]:
        """
        Batch merge entities by combining newly extracted entities with old entities and return the merge results.

        Args:
            neibor_entity_dict (Dict[str, Entity]): Neighbor entity dictionary, where keys are entity names and values are entity objects.
            old_entities (List[Entity]): List of old entities.
            new_entities (Dict[str, Entity]): Dictionary of newly extracted entities, where keys are entity names and values are entity objects.

        Returns:
            List[Dict[str, Any]]: Returns a list of merged entity information, where each element is a dictionary containing the merge results.
        """

        old_entities_dict = {entity.name: entity for entity in old_entities}


        neibor_entity_list = [
            {
                "new_entity_name": {
                    "name": new_entity_name,
                    "description": new_entities[new_entity_name].description if new_entity_name in new_entities else ""
                },
                "candidate_entity_names": [
                    {
                        "name": candidate_entity_name,
                        "similar_names": old_entities_dict[candidate_entity_name].synonyms,
                        "description": old_entities_dict[candidate_entity_name].description
                    } for candidate_entity_name in candidate_entity_names if candidate_entity_name in old_entities_dict
                ]
            } for new_entity_name, candidate_entity_names in neibor_entity_dict.items()
        ]

        entities_batch_list = [
            neibor_entity_list[i: i + ENTITY_MERGE_BATCH_SIZE]
            for i in range(0, len(neibor_entity_list), ENTITY_MERGE_BATCH_SIZE)
        ]

        input_states_list = []

        for entities_batch in entities_batch_list:
            input_statements = ""
            for i, entity in enumerate(entities_batch):
                input_statements += f"Group {i + 1}:\n"
                input_statements += f"  New Entity Name: {entity['new_entity_name']['name']}\n"
                input_statements += f"  New Entity Description: {entity['new_entity_name']['description']}\n\n"

                for j, candidate_entity in enumerate(entity["candidate_entity_names"]):
                    input_statements += f"Candidate Entity {j + 1}:\n"
                    input_statements += f"  Name: {candidate_entity['name']}\n"
                    input_statements += f"  Similar Names: {candidate_entity['similar_names']}\n"
                    input_statements += f"  Description: \n  {candidate_entity['description']}\n"
                input_statements += "\n"
            input_states_list.append(input_statements)

        return input_states_list

    def batch_merge_entity(self, entities_batch: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        default_res = [{
            "merged": False,
            "merge_target": None
        } for _ in entities_batch]

        merge_entity_pattern = r'\[.*\]'
        messages = Prompts.return_merge_entity_prompt(merge_entity_json=entities_batch,
                                                      system_prompt=self.langfuse_dict["merge_entities"][
                                                          "system_prompt"])

        answer = self.llms["merge_entities"].chat.completions.create(messages=messages,model=self.model_name, **self.model_params)
        content = answer.choices[0].message.content
        
        merge_res_list = self.parse_json_response(content, default_res, merge_entity_pattern)
        
        if not merge_res_list:
            return []
            
        return [
            {
                "merged": merge_res.get("merged", False),
                "merge_target": merge_res.get("merge_target") if merge_res.get("merged", False) else merge_res.get("new_entity_name", ""),
                "new_entity_name": merge_res.get("new_entity_name", "")
            } for merge_res in merge_res_list if isinstance(merge_res, dict)
        ]

    def judge_entity_type(self, entity_timelines: List[Timeline]) -> EntityType:
        if not entity_timelines:
            return EntityType.NORMAL_ENTITY
        person_rate = sum(timeline.timeline_type in [TimelineType.PERSON]
                          for timeline in entity_timelines) / len(entity_timelines)
        location_rate = sum(timeline.timeline_type in [TimelineType.GEO]
                            for timeline in entity_timelines) / len(entity_timelines)

        if person_rate > RATE_THRESHOLD:
            return EntityType.PERSON

        if location_rate > RATE_THRESHOLD:
            return EntityType.LOCATION

        return EntityType.NORMAL_ENTITY

    def parse_json_response(self, response: str, default_res: Optional[Union[Dict, List]], pattern: str) -> Dict[
        str, Any]:
        matches = re.findall(pattern, response, re.DOTALL)
        if not matches:
            logging.error(f"No Json Found: {response}")
            return default_res
        try:
            json_res = json.loads(matches[0])
            if isinstance(json_res, list):
                for item in json_res:
                    if isinstance(item, dict) and 'new_entity_name' not in item:
                        item['new_entity_name'] = None    
        except Exception as e:
            logging.error(f"Json Parse Error: {traceback.format_exc()}-{response}")
            return default_res
        return json_res


class PersonalWiki:
    _input_keys: List = ["userName", "oldWiki", "wikiType", "entityName", "timelines", "preferredLanguage"]
    _output_keys: List = ["entityWiki"]
    _must_keys: List = ["userName", "oldWiki", "wikiType", "entityName", "timelines", "preferredLanguage"]

    model_params = {
        "temperature": 0,
        "max_tokens": 3000,
        "top_p": 0,
        # "frequency_penalty": 0,
        # "seed": 42,
        # "presence_penalty": 0,
        # "request_timeout": 45,
        # "max_retries": 1,
        "extra_body": {
                "metadata": {
                    "tags": ["lpm_personal_wiki"]
                }
            }
    }


    def __init__(self, **kwargs):
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
        self.max_threads = 2
        self.class_name = self.__class__.__name__
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.model_params.update(**kwargs)
        self.langfuse_dict = {
            k: self._get_langfuse_prompt(k, v) for k, v in PERSONAL_WIKI_DEFAULT_PROMPT.items()
        }
        self.llms: Dict[str, openai.OpenAI] = {
            k: openai.OpenAI(
                api_key=user_llm_config.chat_api_key,
                base_url=user_llm_config.chat_endpoint,
            )
            for k, _ in PERSONAL_WIKI_DEFAULT_PROMPT.items()
        }
        logger.info("PersonalWiki init success")


    def _get_langfuse_prompt(self, prompt_key, default_prompt) -> Dict[str, Any]:
        try:
            system_prompt = default_prompt
            logging.info(f"Get prompt success: {prompt_key}")
        except Exception as e:
            logging.error(f"Failed to get prompt [{prompt_key}]: {traceback.format_exc()}")
            system_prompt = default_prompt
        return {"system_prompt": system_prompt,"model":self.model_name}

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_name = inputs.get("userName", "")
        old_entity_wiki = inputs.get("oldEntityWiki", {})
        wiki_type = EntityType(inputs.get("wikiType", "NORMAL_ENTITY"))
        entity_name = inputs.get("entityName", "")
        user_self_intro = inputs.get("aboutMe", "")
        timelines = [Timeline(**timeline) for timeline in inputs.get("timelines", [])]

        preferred_language = inputs.get("preferredLanguage", "English/English")
        old_entity_wiki = EntityWiki(**old_entity_wiki) if old_entity_wiki else None
        

        if not user_name:
            raise Exception(f"The user name is empty! Please check the user name!")

        if not entity_name or not timelines:
            raise Exception(f"The entity name or timelines is empty! Please check the entity name or timelines!")

        if wiki_type == EntityType.PERSON:
            entity_type = "person"
        elif wiki_type == EntityType.LOCATION:
            entity_type = "location"
        elif wiki_type == EntityType.CONCEPT:
            entity_type = "concept"
        else:
            entity_type = "entity"

        new_entity_wiki = self.update_entity_wiki_text(user_name, user_self_intro, entity_name, entity_type, timelines,
                                                       old_entity_wiki, preferred_language)
        # Do not need timeline for now
        # new_entity_wiki = self.update_entity_wiki_timelines(user_name, entity_name, entity_type, user_self_intro,
        #                                                     timelines, new_entity_wiki, preferred_language)

        return {
            "entityWiki": new_entity_wiki.to_dict()
        }

    def update_entity_wiki_text(self,
                                user_name: str,
                                user_self_intro: str,
                                entity_name: str,
                                entity_type: str,
                                timelines: List[Timeline],
                                old_entity_wiki: Optional[EntityWiki],
                                preferred_language: str):
        prompt_type = f"personal_wiki_{entity_type}"
        desc_type = "Concept Name" if entity_type == "concept" else "Entity Name"

        system_prompt = self.langfuse_dict[prompt_type]["system_prompt"].format(
            user_name=user_name,
            entity_name=entity_name,
            preview_version_wiki=old_entity_wiki.wiki_text if old_entity_wiki else "",
            prefer_lang=preferred_language,
            self_intro=user_self_intro
        )

        description_list = "\n".join([timeline._desc_() for timeline in timelines])

        user_prompt = f"""
        -Data-

# {desc_type}: {entity_name}

# Impression Flow:
# {description_list}
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if self.client:
            llm = self.client
        else:
            print("There is no valid client can be used.")
        answer = llm.chat.completions.create(messages=messages,model=self.model_name,**self.model_params)

        wiki_text = answer.choices[0].message.content

        logging.info(f"Generated wiki: {wiki_text}")

        if old_entity_wiki:
            old_entity_wiki.wiki_text = wiki_text
            return old_entity_wiki

        return EntityWiki(wikiText=wiki_text, monthlyTimelines=[])

    def update_entity_wiki_timelines(self,
                                     user_name: str,
                                     entity_name: str,
                                     entity_type: str,
                                     user_self_intro: str,
                                     timelines: List[Timeline],
                                     old_entity_wiki: Optional[EntityWiki],
                                     preferred_language: str):

        # Generate monthly timelines month by month, w/o cold start to avoid performance gap
        monthly_timeline_dict = group_timelines_by_month(timelines)

        system_prompt = self.langfuse_dict["timeline_generate"]["system_prompt"].format(
            user_self_intro=user_self_intro,
            user_name=user_name,
            prefer_lang=preferred_language,
            entity_name=entity_name
        )

        entity_wiki_month_date = {
            timeline.month_date: timeline.id for timeline in old_entity_wiki.monthly_timelines
        }

        max_month_idx = old_entity_wiki.max_month_idx if old_entity_wiki else 0

        missing_month_dates = set(monthly_timeline_dict.keys()) - set(entity_wiki_month_date.keys())

        for month_date in missing_month_dates:
            max_month_idx += 1
            entity_wiki_month_date[month_date] = max_month_idx

        with ThreadPoolExecutor(max_workers=min(self.max_threads, len(monthly_timeline_dict))) as executor:
            futures = [executor.submit(self.generate_monthly_timeline_byDay, entity_wiki_month_date[month_date], month_date,
                                       monthly_timelines, system_prompt)
                       for month_date, monthly_timelines in monthly_timeline_dict.items()]
            new_wiki_timelines = [future.result() for future in futures]

        # filter month that without daily timelien 
        new_wiki_timelines = [timeline for timeline in new_wiki_timelines if timeline.daily_timelines]

        new_month_date = [timeline.month_date for timeline in new_wiki_timelines if timeline.month_date]
        origin_monthly_timelines = [timeline for timeline in old_entity_wiki.monthly_timelines if
                                    timeline.month_date not in new_month_date]
        origin_monthly_timelines.extend(new_wiki_timelines)
        monthly_timelines = sorted(origin_monthly_timelines,
                                   key=lambda x: datetime.strptime(x.month_date, MONTH_TIME_FORMAT))

        for month_idx, monthly_timeline in enumerate(monthly_timelines):
            if monthly_timeline.title:
                continue
            history_monthly_timelines = monthly_timelines[:month_idx]
            monthly_timelines[month_idx] = self.generate_monthly_timeline_title(history_monthly_timelines,
                                                                                monthly_timeline,
                                                                                entity_name,
                                                                                entity_type,
                                                                                preferred_language)

        old_entity_wiki.monthly_timelines = monthly_timelines
        return old_entity_wiki
    
    def generate_monthly_timeline_byDay(self,
                                    month_idx: int,
                                    month_date: str,
                                    timelines: List[Timeline],
                                    system_prompt: str,
                                    ) -> MonthlyTimeline:

        # merge based on day
        daily_timeline_dict = group_timelines_by_day(timelines)

        def generate_daily_timeline(day: str, daily_timelines: List[Timeline], day_idx: int) -> Dict[str, Any]:
            user_prompt = "\n".join([timeline._desc_(with_note_id=True) for timeline in daily_timelines])

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            if self.client:
                llm = self.client
            else:
                logger.info("There is no valid client can be used.")
            
            answer = llm.chat.completions.create(messages=messages,model=self.model_name,**self.model_params)
            new_timeline_str = answer.choices[0].message.content
            print(f"Before processing:\n{new_timeline_str}")
            daily_timeline = parse_daily_timelines(new_timeline_str, day)
            daily_timeline["id"] = day_idx
            return daily_timeline

        with ThreadPoolExecutor(max_workers=min(self.max_threads, len(daily_timeline_dict))) as executor:
            futures = [executor.submit(generate_daily_timeline, day, daily_timelines, idx) 
                    for idx, (day, daily_timelines) in enumerate(daily_timeline_dict.items())]
            daily_timelines = [future.result() for future in futures]
        daily_timelines = [timeline for timeline in daily_timelines if timeline['dateTime']]
        print(f"After processing and merging:\n{daily_timelines}")
        monthly_timeline = MonthlyTimeline(
            id=month_idx,
            monthDate=month_date,
            title="",
            dailyTimelines=daily_timelines
        )

        return monthly_timeline

    def generate_monthly_timeline_title(self,
                                        history_monthly_timelines: List[MonthlyTimeline],
                                        monthly_timeline: MonthlyTimeline,
                                        entity_name: str,
                                        entity_type: str,
                                        prefer_lang: str):
        prompt_type = f"monthly_timeline_title_{entity_type}"
        system_prompt = self.langfuse_dict[prompt_type]["system_prompt"].format(
            prefer_lang=prefer_lang
        )

        history_monthly_timelines_str = "\n".join([timeline.title for timeline in history_monthly_timelines])

        if entity_type == "entity":
            prefix_prompt = ""
        elif entity_type == "person":
            prefix_prompt = f"Here are some of my experiences with {entity_name}."
        elif entity_type == "location":
            prefix_prompt = f"Current Location is {entity_name}."
        elif entity_type == "concept":
            prefix_prompt = f"Current Concept is {entity_name}."
        else:
            raise Exception(f"Invalid entity type: {entity_type}")

        user_prompt = f"""{prefix_prompt}
Historical development stages and events:
{history_monthly_timelines_str}

Current events:
{monthly_timeline._desc_()}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        if self.client:
            llm = self.client
        else:
            logger.info("There is no valid client can be used.")
        answer = llm.chat.completions.create(messages=messages,model=self.model_name,**self.model_params)
        raw_monthly_timeline_title = answer.choices[0].message.content

        logging.info(f"Generated monthly timeline title:\n{raw_monthly_timeline_title}")

        date_str, title_str = self.parse_monthly_timeline_title(raw_monthly_timeline_title)
        if date_str != monthly_timeline.month_date:
            logging.warning(f"Date not match: {date_str} != {monthly_timeline.month_date}")

        monthly_timeline.title = title_str
        return monthly_timeline

    def parse_monthly_timeline_title(self, raw_monthly_timeline_title: str):
        date_pattern = r"Date:\s*([0-9]{4}-[0-9]{2})"
        title_pattern = r"Title:\s*([\s\S]*?)\s*(?:\n|$)"
        date_match = re.search(date_pattern, raw_monthly_timeline_title)
        title_match = re.search(title_pattern, raw_monthly_timeline_title)
        date_str = date_match.group(1) if date_match else ""
        title_str = title_match.group(1) if title_match else ""
        return date_str, title_str

